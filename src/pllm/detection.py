"""Failure detection at the gateway: which stage incarnations are dead, and when (R6.3, R6.4).

Everything is keyed by epoch, which is unique across incarnations, so a new incarnation
never inherits the last beat of an old one. An epoch is declared dead once and never
revived. The clock for an epoch starts at its first Beat, so a registered stage that has
not beaten yet cannot time out.

Every detection and registration is kept as a flat JSON-ready record (events()) and logged
as one JSON line. Reported times are integers from time.perf_counter_ns on the gateway, the
one reporting clock, in fields ending in _ns:

- detected: path (stream_break, heartbeat_timeout, step_deadline or replaced), stage, epoch,
  declared, t_ns, time, last_beat_ns, since_last_beat_ns, rid, step, detail
- registered: stage, old_epoch, new_epoch, t_ns, time, waited_ns, route_queued_ns

time is the wall clock, a label only: it is never subtracted. waited_ns runs from the
declaration of old_epoch to the new registration. route_queued_ns is when the route update
was queued for the neighbour, and None if the neighbour had no stream.

A step deadline declares nothing. The gateway cannot tell which stage holds a late step, and
a slow stage keeps beating, so its supervisor would never send the Register a declaration
waits for. A dead stage is declared by its stream break or heartbeat timeout anyway. The
record names as suspect (stage and epoch) the stage with the oldest beat past half the
heartbeat timeout, if there is one.

HEARTBEAT_TIMEOUT_SECONDS and STEP_TIMEOUT_SECONDS are provisional: the timeout is measured
on host A only. Tests set heartbeat_timeout_ns and step_timeout_ns on the instance.
"""

import collections
import json
import logging
import threading
import time
from collections.abc import Callable

from pllm.heartbeat import HEARTBEAT_SECONDS

log = logging.getLogger("pllm.detection")

# Three intervals leave no slack for one late beat; each further one adds to recovery time.
HEARTBEAT_TIMEOUT_SECONDS = 4 * HEARTBEAT_SECONDS
STEP_TIMEOUT_SECONDS = 10.0  # well above a measured 512-token prefill (0.16 s at most)
MAX_EVENTS = 4096
NS = 1_000_000_000


class Detector:
    """The live and dead epochs and the records of what happened. Safe from many threads."""

    def __init__(
        self,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT_SECONDS,
        step_timeout: float = STEP_TIMEOUT_SECONDS,
        clock: Callable[[], int] = time.perf_counter_ns,
        wall: Callable[[], float] = time.time,
    ):
        self.heartbeat_timeout_ns = round(heartbeat_timeout * NS)
        self.step_timeout_ns = round(step_timeout * NS)
        self.now = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._live: dict[int, tuple[int, int]] = {}  # epoch: stage, last beat
        self._dead: dict[int, int] = {}  # epoch: when it was declared
        self._events: collections.deque = collections.deque(maxlen=MAX_EVENTS)

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def is_dead(self, epoch: int) -> bool:
        with self._lock:
            return epoch in self._dead

    def beat(self, stage: int, epoch: int) -> bool:
        """Note a Beat's arrival. False if the epoch is dead: the Beat does not count."""
        with self._lock:
            if epoch in self._dead:
                return False
            self._live[epoch] = (stage, self.now())
            return True

    def declare(self, stage: int, epoch: int, path: str, detail: str = "") -> bool:
        """Declare an epoch dead. The first declaration wins; True if this was it."""
        with self._lock:
            record = self._declare(stage, epoch, path, detail)
        if record is not None:
            self._log(record)
        return record is not None

    def expire(self) -> None:
        """Declare every epoch whose last beat is older than the heartbeat timeout."""
        with self._lock:
            now = self.now()
            late = [
                (stage, epoch)
                for epoch, (stage, last) in self._live.items()
                if now - last >= self.heartbeat_timeout_ns
            ]
            records = [self._declare(stage, epoch, "heartbeat_timeout") for stage, epoch in late]
        for record in records:
            self._log(record)

    def step_deadline(self, rid: str, step: int) -> None:
        """Record a step that outlived the step timeout. It declares nothing."""
        with self._lock:
            now = self.now()
            stale = [
                (last, stage, epoch)
                for epoch, (stage, last) in self._live.items()
                if (now - last) * 2 > self.heartbeat_timeout_ns
            ]
            last, stage, epoch = min(stale, default=(None, None, None))
            record = self._detected("step_deadline", stage, epoch, False, now, last, rid, step)
        self._log(record)

    def registered(
        self, stage: int, old_epoch: int | None, new_epoch: int, route_queued_ns: int | None
    ) -> None:
        """Record a registration, after any declaration of the epoch it replaces."""
        with self._lock:
            now = self.now()
            declared = self._dead.get(old_epoch)
            record = {
                "event": "registered",
                "stage": stage,
                "old_epoch": old_epoch,
                "new_epoch": new_epoch,
                "t_ns": now,
                "time": self._wall(),
                "waited_ns": None if declared is None else now - declared,
                "route_queued_ns": route_queued_ns,
            }
            self._events.append(record)
        self._log(record)

    def _declare(self, stage: int, epoch: int, path: str, detail: str = "") -> dict | None:
        if epoch in self._dead:
            return None
        now = self.now()
        last = self._live.pop(epoch, (stage, None))[1]
        self._dead[epoch] = now
        return self._detected(path, stage, epoch, True, now, last, detail=detail)

    def _detected(self, path, stage, epoch, declared, now, last, rid=None, step=None, detail=""):
        record = {
            "event": "detected",
            "path": path,
            "stage": stage,
            "epoch": epoch,
            "declared": declared,
            "t_ns": now,
            "time": self._wall(),
            "last_beat_ns": last,
            "since_last_beat_ns": None if last is None else now - last,
            "rid": rid,
            "step": step,
            "detail": detail,
        }
        self._events.append(record)
        return record

    @staticmethod
    def _log(record: dict) -> None:
        log.warning("%s", json.dumps(record))
