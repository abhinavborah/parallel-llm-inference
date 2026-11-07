"""A stage worker: one process's share of the model, loaded from configuration.

The worker loads only its layer range (R3.1) and logs the weight load time and the
resident memory of the process. The first stage reads embedding rows through the
checkpoint's memory map, so only the rows it is asked for enter RAM (R3.2). Before each
step it calls the step delay once, which adds nothing unless a sleep is set (R6.1).

Nothing here depends on the operating system (R3.15). The network service that hosts a
worker is a separate story.

Usage (loads the stage, logs the load time and RSS, and exits):
uv run python -m pllm.worker smollm2-135m --layers 0:15
"""

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch

from pllm.faults import StepDelay
from pllm.model import KVCache
from pllm.models import MODELS_DIR, PINNED
from pllm.split import Stage

log = logging.getLogger("pllm.worker")

MIB = 1024 * 1024


def parse_layers(text: str) -> tuple[int, int]:
    """Read 'START:END', layers START to END - 1, as two whole numbers."""
    start, sep, end = text.partition(":")
    if not sep or not start.isdigit() or not end.isdigit():
        raise ValueError(f"layers must be START:END with whole numbers, got {text!r}")
    return int(start), int(end)


@dataclass(frozen=True)
class WorkerConfig:
    model_dir: Path
    layer_start: int
    layer_end: int
    step_delay: float = 0.0


def rss_mib() -> float:
    return psutil.Process().memory_info().rss / MIB


class StageWorker:
    def __init__(self, config: WorkerConfig, delay: Callable[[], None] | None = None):
        self.config = config
        self.delay = delay if delay is not None else StepDelay(config.step_delay)
        began = time.perf_counter()
        self.stage = Stage.load(
            config.model_dir, config.layer_start, config.layer_end, mapped_embedding=True
        )
        self.load_seconds = time.perf_counter() - began
        self.rss_after_load_mib = rss_mib()
        log.info(
            "stage layers %d to %d loaded in %.2f s, rss %.0f MiB",
            config.layer_start,
            config.layer_end - 1,
            self.load_seconds,
            self.rss_after_load_mib,
        )

    def new_cache(self) -> KVCache:
        return self.stage.new_cache()

    def step(self, inputs: torch.Tensor, start_pos: int, cache: KVCache) -> torch.Tensor:
        """Run one step: the delay, called once, then the stage."""
        self.delay()
        return self.stage.forward(inputs, start_pos, cache)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load a stage worker and report its memory.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--layers", required=True, help="START:END, layers START to END - 1")
    parser.add_argument("--step-delay", type=float, default=0.0, help="seconds slept per step")
    args = parser.parse_args(argv)
    try:
        start, end = parse_layers(args.layers)
        StepDelay(args.step_delay)  # rejects a negative value
    except ValueError as err:
        parser.error(str(err))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    StageWorker(WorkerConfig(MODELS_DIR / args.model, start, end, args.step_delay))


if __name__ == "__main__":
    main()
