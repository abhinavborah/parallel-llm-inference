"""A stage's per-request KV store: position rules and release (R3.5, R3.6, R3.9).

Each request id has its own cache for this stage's layers, created by its first message
at position 0. For a message that starts at start_pos with the request's cache holding
`length` positions:

- start_pos == length: the new keys and values are appended.
- start_pos < length: the cache is truncated to start_pos and then the new ones are
  appended. The gateway does this to replay from an earlier message. Truncating at the end
  of an earlier message and appending is bitwise equal to a fresh run (R2.7); stages only
  truncate at message boundaries, because inside a message it is only agreement.
- start_pos > length, or a request this stage has never seen that does not start at 0:
  the message is refused with Gap, which becomes a GAP Nack so the gateway replays.

free releases a request's cache; freeing an unknown request does nothing, so a Free that
arrives twice is harmless.

Memory cost: a request holds, per position, layers x 2 (keys and values) x kv_heads x
head_dim x 4 bytes (fp32). SmolLM2-135M has 3 kv heads of 64 and 15 layers per stage, so
23,040 bytes per position; a request of 528 positions (512 prompt, 16 new) holds 11.6 MiB
on each stage. That is the steady state: each step copies a layer's keys and values when it
appends, so a layer's cache is briefly about twice its size during a step. Caches of requests
that never receive a Free stay until the process ends; the gateway frees them (R4.6).
"""

import threading
from collections.abc import Callable

from pllm.model import KVCache, ModelConfig

FLOAT_BYTES = 4


class Gap(Exception):
    """A message that starts past what this stage has cached for its request."""

    def __init__(self, rid: str, start_pos: int, length: int | None):
        self.rid, self.start_pos, self.length = rid, start_pos, length
        have = "has no cache for it" if length is None else f"caches {length} positions"
        super().__init__(f"request {rid!r} starts at position {start_pos} but this stage {have}")


class KVStore:
    def __init__(self, config: ModelConfig, num_layers: int, new_cache: Callable[[], KVCache]):
        self.bytes_per_position = (
            num_layers * 2 * config.num_kv_heads * config.head_dim * FLOAT_BYTES
        )
        self._new_cache = new_cache
        self._caches: dict[str, KVCache] = {}
        self._lock = threading.Lock()

    def acquire(self, rid: str, start_pos: int) -> KVCache:
        """The cache a message starting at start_pos writes to, positioned for it.

        Raises Gap if the message starts past the cache (nothing is changed then).
        """
        with self._lock:
            cache = self._caches.get(rid)
            if cache is None:
                if start_pos != 0:
                    raise Gap(rid, start_pos, None)
                cache = self._caches[rid] = self._new_cache()
                return cache
            length = len(cache)
            if start_pos > length:
                raise Gap(rid, start_pos, length)
            if start_pos < length:
                # clone: a slice is a view and would keep the longer storage alive
                cache.keys = [k[:, :start_pos].clone() for k in cache.keys]
                cache.values = [v[:, :start_pos].clone() for v in cache.values]
            return cache

    def free(self, rid: str) -> bool:
        """Release a request's cache; False if there was none."""
        with self._lock:
            return self._caches.pop(rid, None) is not None

    def free_cache(self, rid: str, cache: KVCache) -> bool:
        """Release rid only if the cache stored for it is this one. A failed step uses this,
        so it cannot drop a cache that was freed and started again while the step ran."""
        with self._lock:
            if self._caches.get(rid) is not cache:
                return False
            del self._caches[rid]
            return True

    def __len__(self) -> int:
        """The number of requests cached (the Beat's cached_requests)."""
        with self._lock:
            return len(self._caches)

    def __contains__(self, rid: str) -> bool:
        with self._lock:
            return rid in self._caches

    def rids(self) -> set[str]:
        with self._lock:
            return set(self._caches)

    def length(self, rid: str) -> int:
        """Positions cached for rid; 0 if there is none."""
        with self._lock:
            cache = self._caches.get(rid)
            return len(cache) if cache is not None else 0

    def nbytes(self, rid: str) -> int:
        """Bytes of storage held for rid's keys and values."""
        with self._lock:
            cache = self._caches.get(rid)
            if cache is None:
                return 0
            return sum(t.untyped_storage().nbytes() for t in cache.keys + cache.values)
