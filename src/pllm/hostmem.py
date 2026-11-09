"""Measure available memory on a host and propose the worker budget.

Close every other program first, plug in the charger, then run:

    uv run python -m pllm.hostmem

It samples available memory once a second for a minute and takes the minimum, since
Windows background work makes it move. The proposed worker budget is that minimum
less a fixed headroom, rounded down to 0.1 GiB. The record also lists the processes
using the most memory, so it shows what was still running. Note the "Available"
figure in Task Manager (Performance, Memory) at the same time and record it with
the result.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import psutil

from pllm import runrecord

GIB = 1024**3
STEP = GIB // 10
DEFAULT_HEADROOM_GIB = 0.5


def sample(seconds: float, interval: float, read: Callable[[], int] | None = None) -> list[int]:
    """Available memory in bytes, read every `interval` seconds for `seconds`."""
    if read is None:

        def read() -> int:
            return psutil.virtual_memory().available

    count = max(1, round(seconds / interval))
    values = []
    for i in range(count):
        values.append(read())
        if i < count - 1:
            time.sleep(interval)
    return values


def budget(available: int, headroom: int) -> int:
    """Available bytes less headroom, rounded down to 0.1 GiB, never negative."""
    return max(0, (available - headroom) // STEP * STEP)


def top_processes(count: int = 10) -> list[dict]:
    """The processes with the largest resident memory, largest first."""
    found = []
    for proc in psutil.process_iter(["name", "memory_info"]):
        info = proc.info.get("memory_info")
        if info is not None:
            found.append({"name": proc.info["name"], "rss_bytes": info.rss})
    return sorted(found, key=lambda p: p["rss_bytes"], reverse=True)[:count]


def gib(value: int) -> str:
    return f"{value / GIB:.2f} GiB"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--headroom-gib", type=float, default=DEFAULT_HEADROOM_GIB)
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.interval <= 0 or args.headroom_gib < 0:
        parser.error("--seconds and --interval must be positive, --headroom-gib not negative")

    headroom = int(args.headroom_gib * GIB)
    with runrecord.record("host-memory") as rec:
        values = sample(args.seconds, args.interval)
        memory = psutil.virtual_memory()
        rec.update(
            method="psutil.virtual_memory().available, minimum of the samples",
            samples=len(values),
            interval_s=args.interval,
            total_bytes=memory.total,
            available_min_bytes=min(values),
            available_median_bytes=int(statistics.median(values)),
            available_max_bytes=max(values),
            swap_used_bytes=psutil.swap_memory().used,
            headroom_bytes=headroom,
            budget_bytes=budget(min(values), headroom),
            top_processes=top_processes(),
        )
    print(f"total             {gib(rec['total_bytes'])}")
    print(f"available min     {gib(rec['available_min_bytes'])} over {len(values)} samples")
    print(f"available median  {gib(rec['available_median_bytes'])}")
    print(f"swap in use       {gib(rec['swap_used_bytes'])}")
    print(f"worker budget     {gib(rec['budget_bytes'])} (headroom {gib(headroom)})")
    print("largest processes:")
    for proc in rec["top_processes"]:
        print(f"  {gib(proc['rss_bytes']):>10}  {proc['name']}")


if __name__ == "__main__":
    main()
