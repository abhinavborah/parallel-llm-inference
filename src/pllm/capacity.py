"""The capacity-aware split point, from each host's measured free memory and speed (R8.6).

The split point k gives Host B (stage 0) layers 0 to k - 1 and Host A (stage 1) the other
n - k. Each host is described by a straight line in its number of layers, for memory and
for the time of one decode step, both measured (pllm.x9 probe and pllm.hostmem). Nothing
here has a default or reads anything: the same numbers give the same k.

A split is allowed when

1. both stages have at least one layer, and Host B has fewer than an even split would
   give it: k < n / 2;
2. each host's predicted worker memory is at most the memory it has free.

Among the allowed splits the one whose slower stage is fastest is chosen, since the
slower stage limits a pipeline's throughput; on a tie, the smaller k, which asks less of
Host B. If no split is allowed, NoSplit says why and no k is returned.

Rule 1 is a requirement, not a measurement. When the best split by speed and memory alone
would give Host B half the layers or more, forced_below_even is True: the k returned then
comes from rule 1 and must not be presented as what capacity alone chose.
"""

import math
from dataclasses import dataclass


class NoSplit(ValueError):
    """No split satisfies the constraints."""


def _finite(value) -> bool:
    # bool is a subclass of int, so rule it out explicitly
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class HostCapacity:
    """One host, all measured. Memory is in bytes, time in seconds per decode step."""

    free_bytes: float  # memory the worker may use on this host
    fixed_bytes: float  # the worker's memory apart from its layers
    bytes_per_layer: float  # memory each layer adds, with any KV cache reserved for it
    fixed_seconds: float  # step time apart from the layers (embedding, or norm and LM head)
    seconds_per_layer: float  # step time each layer adds

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not _finite(value):
                raise ValueError(f"{name} must be a measured, finite number, got {value!r}")
        if self.free_bytes < 0 or self.fixed_bytes < 0:
            raise ValueError("free_bytes and fixed_bytes must not be negative")
        if self.bytes_per_layer <= 0 or self.seconds_per_layer <= 0:
            raise ValueError("bytes_per_layer and seconds_per_layer must be positive")

    def bytes_for(self, layers: int) -> float:
        return self.fixed_bytes + layers * self.bytes_per_layer

    def seconds_for(self, layers: int) -> float:
        """The predicted time of one step. fixed_seconds is a fitted intercept, which noise
        can make negative, so a prediction that is not positive is refused."""
        seconds = self.fixed_seconds + layers * self.seconds_per_layer
        if seconds <= 0:
            raise ValueError(
                f"the fitted step time for {layers} layers is {seconds} s, not positive: "
                "measure the host again"
            )
        return seconds


@dataclass(frozen=True)
class Split:
    k: int  # Host B runs layers 0 to k - 1
    num_layers: int
    even_k: int  # n // 2, the even split
    allowed: tuple[int, ...]  # every k that satisfies both rules
    unconstrained_k: int  # the choice without rule 1
    forced_below_even: bool  # rule 1, not capacity, decided
    host_b_bytes: float
    host_a_bytes: float
    host_b_step_seconds: float
    host_a_step_seconds: float


def even_split(num_layers: int) -> int:
    return num_layers // 2


def _fits(num_layers: int, k: int, host_b: HostCapacity, host_a: HostCapacity) -> bool:
    return (
        host_b.bytes_for(k) <= host_b.free_bytes
        and host_a.bytes_for(num_layers - k) <= host_a.free_bytes
    )


def choose_split(num_layers: int, host_b: HostCapacity, host_a: HostCapacity) -> Split:
    """The split point for a model of num_layers layers. Raises NoSplit if none is allowed
    and ValueError for numbers that cannot be used."""
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 2:
        raise ValueError(f"num_layers must be a whole number of at least 2, got {num_layers!r}")

    def bottleneck(k: int) -> float:
        return max(host_b.seconds_for(k), host_a.seconds_for(num_layers - k))

    def best(candidates) -> int:
        return min(candidates, key=lambda k: (bottleneck(k), k))

    fitting = [k for k in range(1, num_layers) if _fits(num_layers, k, host_b, host_a)]
    allowed = tuple(k for k in fitting if 2 * k < num_layers)
    if not allowed:
        below = (num_layers - 1) // 2  # the largest k under an even split
        if below < 1:
            raise NoSplit(f"a model of {num_layers} layers has no split below an even one")
        mib = 1024 * 1024
        reasons = []
        if host_b.bytes_for(1) > host_b.free_bytes:
            reasons.append(
                f"Host B cannot hold one layer: needs {host_b.bytes_for(1) / mib:.1f} MiB, "
                f"has {host_b.free_bytes / mib:.1f} MiB"
            )
        most_a = host_a.bytes_for(num_layers - below)
        if most_a > host_a.free_bytes:
            reasons.append(
                f"Host A cannot hold the {num_layers - below} layers left when Host B has "
                f"{below}: needs {most_a / mib:.1f} MiB, has {host_a.free_bytes / mib:.1f} MiB"
            )
        if not reasons:
            reasons.append("no k below the even split fits both hosts at the same time")
        raise NoSplit("no split satisfies the constraints: " + "; ".join(reasons))
    k = best(allowed)
    unconstrained = best(fitting)
    return Split(
        k=k,
        num_layers=num_layers,
        even_k=even_split(num_layers),
        allowed=allowed,
        unconstrained_k=unconstrained,
        forced_below_even=2 * unconstrained >= num_layers,
        host_b_bytes=host_b.bytes_for(k),
        host_a_bytes=host_a.bytes_for(num_layers - k),
        host_b_step_seconds=host_b.seconds_for(k),
        host_a_step_seconds=host_a.seconds_for(num_layers - k),
    )


def sweep_points(even_k: int, chosen_k: int, points: int = 4) -> list[int]:
    """The split points X9 runs, from the even split down to the chosen one, spread evenly,
    both ends included. Fewer than points when the range is narrower."""
    if points < 2:
        raise ValueError("a sweep needs at least its two ends")
    if not 1 <= chosen_k < even_k:
        raise ValueError(f"the chosen k {chosen_k} must be from 1 to below the even {even_k}")
    step = (even_k - chosen_k) / (points - 1)
    return sorted({round(even_k - i * step) for i in range(points)}, reverse=True)
