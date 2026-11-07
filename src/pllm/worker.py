"""A stage worker: one process's share of the model, loaded from configuration.

The worker loads only its layer range (R3.1) and logs the weight load time and the
resident memory of the process. The first stage reads embedding rows through the
checkpoint's memory map, so only the rows it is asked for enter RAM (R3.2). Before each
step it calls the step delay once, which adds nothing unless a sleep is set (R6.1).

Nothing here depends on the operating system (R3.15). The network service that hosts a
worker is a separate story.

The layer range comes from the split point in the hosts file ([split], R3.1);
--layers overrides it for tests and split sweeps.

Usage (loads the stage, logs the load time and RSS, and exits):
uv run python -m pllm.worker smollm2-135m --stage 0
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

from pllm import config
from pllm.faults import StepDelay
from pllm.model import KVCache, ModelConfig
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


def stage_layers(model_dir: Path, model: str, stage: int) -> tuple[int, int]:
    """Layer range of stage 0 or 1 from the split point in the hosts file."""
    n = ModelConfig.from_file(model_dir / "config.json").num_layers
    k = config.split_point(model)
    if not 1 <= k <= n - 1:
        raise ValueError(f"split point {k} for {model} outside 1 to {n - 1}")
    return (0, k) if stage == 0 else (k, n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load a stage worker and report its memory.")
    parser.add_argument("model", choices=sorted(PINNED))
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--stage", type=int, choices=(0, 1), help="layers from the hosts file")
    which.add_argument("--layers", help="START:END, layers START to END - 1, overrides it")
    parser.add_argument("--step-delay", type=float, default=0.0, help="seconds slept per step")
    args = parser.parse_args(argv)
    model_dir = MODELS_DIR / args.model
    try:
        if args.layers is not None:
            start, end = parse_layers(args.layers)
        else:
            start, end = stage_layers(model_dir, args.model, args.stage)
        StepDelay(args.step_delay)  # rejects a negative value
    except (ValueError, OSError) as err:
        parser.error(str(err))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    StageWorker(WorkerConfig(model_dir, start, end, args.step_delay))


if __name__ == "__main__":
    main()
