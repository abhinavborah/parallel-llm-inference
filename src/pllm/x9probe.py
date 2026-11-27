"""One measurement process of `pllm.x9 probe`: loads a stage and times its decode steps.

It is a module of its own so that it imports only what a serving worker imports: its
memory then stands for a worker's. It loads layers START to END - 1 as the worker does,
runs one prefill and then single-token decode steps, prints one JSON line and waits for a
line on stdin, so the parent can read its process tree's counters while it is alive.

The inputs are random token ids (first stage) or random hidden states (last stage). The
times are of the stage's own compute: no serialisation, network or argmax.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import psutil
import torch

from pllm import stage_service  # noqa: F401  the imports of a serving worker
from pllm.worker import StageWorker, WorkerConfig, parse_layers

WARMUP_STEPS = 4  # decode steps run before the timed ones


def time_steps(worker: StageWorker, prompt: int, steps: int, warmup: int = WARMUP_STEPS):
    """The seconds each of `steps` single-token decode steps took, after a prefill of
    `prompt` positions and `warmup` untimed steps."""
    stage = worker.stage
    generator = torch.Generator().manual_seed(0)

    def inputs(count: int) -> torch.Tensor:
        if stage.first:
            return torch.randint(stage.config.vocab_size, (count,), generator=generator)
        return torch.randn(count, stage.config.hidden_size, generator=generator)

    cache = worker.new_cache()
    worker.step(inputs(prompt), 0, cache)
    out = []
    for i in range(warmup + steps):
        one = inputs(1)
        began = time.perf_counter()
        worker.step(one, prompt + i, cache)
        if i >= warmup:
            out.append(time.perf_counter() - began)
    return out


def own_peak_rss() -> int:
    """This process's lifetime peak resident set in bytes."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset
    import resource  # not on Windows

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux gives KiB


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One probe process of pllm.x9.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layers", required=True, help="START:END")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--prompt", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    start, end = parse_layers(args.layers)
    worker = StageWorker(WorkerConfig(args.model_dir, start, end))
    load_peak = own_peak_rss()
    seconds = time_steps(worker, args.prompt, args.steps)
    result = {
        "layer_start": start,
        "layer_end": end,
        "step_seconds": seconds,
        "load_seconds": worker.load_seconds,
        "own_load_peak_rss_bytes": load_peak,
        "own_peak_rss_bytes": own_peak_rss(),
    }
    print(json.dumps(result), flush=True)
    sys.stdin.readline()


if __name__ == "__main__":
    main()
