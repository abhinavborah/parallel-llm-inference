"""A stage worker: one process's share of the model, loaded from configuration.

The worker loads only its layer range (R3.1) and logs the weight load time and the
resident memory of the process. The first stage reads embedding rows through the
checkpoint's memory map, so only the rows it is asked for enter RAM (R3.2). Before each
step it calls the step delay once, which adds nothing unless a sleep is set (R6.1).

Nothing here depends on the operating system (R3.15). The network service that hosts a
worker is pllm.stage_service; --serve starts it.

The layer range comes from the split point in the hosts file ([split], R3.1);
--layers overrides it for tests and split sweeps.

Usage (loads the stage, logs the load time and RSS, and exits):
uv run python -m pllm.worker smollm2-135m --stage 0
uv run python -m pllm.worker smollm2-135m --layers 0:15

Usage (loads the stage, then serves it until Ctrl-C):
uv run python -m pllm.worker smollm2-135m --stage 0 --serve --threads 6
"""

import argparse
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import grpc
import psutil
import torch

from pllm import config, wire
from pllm.control import register
from pllm.faults import StepDelay
from pllm.model import KVCache, ModelConfig
from pllm.models import MODELS_DIR, PINNED, sha256_of
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


PARENT_ENV = "PLLM_SUPERVISOR_PID"  # set by pllm.supervisor
PARENT_POLL_SECONDS = 1.0


def exit_when_parent_gone(
    pid: int, interval: float = PARENT_POLL_SECONDS, leave: Callable[[int], None] = os._exit
) -> threading.Thread:
    """Call leave(1) once process pid no longer exists, polling every interval seconds.

    A supervisor that is force killed cannot stop its worker; this keeps the worker from
    outliving it and holding its port, which would make the next supervisor's worker fail.
    """

    def watch():
        while psutil.pid_exists(pid):
            time.sleep(interval)
        log.error("supervisor %d is gone; exiting", pid)
        leave(1)

    thread = threading.Thread(target=watch, name="parent-guard", daemon=True)
    thread.start()
    return thread


REGISTER_RETRY_SECONDS = 1.0
RETRY_CODES = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)


def register_with_gateway(
    gateway: str,
    stage: int,
    layers: tuple[int, int],
    model: str,
    model_dir: Path,
    addr: str,
    servicer,
    stop: threading.Event,
    retry_seconds: float = REGISTER_RETRY_SECONDS,
    options: list[tuple[str, int]] | None = None,
) -> int | None:
    """Register this stage with the gateway (R3.3) and keep the epoch it returns.

    Tries again every retry_seconds while the gateway is not reachable, so a stage that
    starts first, or comes back after a restart, registers as soon as the gateway is up. A
    gateway that rejects the registration (a different model, weights or layer range), or
    answers with any error other than being unreachable, is raised and not retried. Returns
    the epoch, or None if stop was set first. options are the channel's gRPC options;
    main passes wire.channel_options, as every channel does (decision #20).
    """
    began = time.perf_counter()
    weights = sha256_of(model_dir / "model.safetensors")
    log.info("weights hashed in %.2f s", time.perf_counter() - began)
    with grpc.insecure_channel(gateway, options=options) as channel:
        while not stop.is_set():
            try:
                epoch = register(channel, stage, layers, model, weights, addr)
            except grpc.RpcError as err:
                if err.code() not in RETRY_CODES:
                    raise  # a rejection or a protocol error will not fix itself
                log.warning("gateway %s not ready (%s); retrying", gateway, err.code().name)
                stop.wait(retry_seconds)
                continue
            servicer.own_epoch = epoch
            log.info(
                "registered stage %d with epoch %d after %.2f s",
                stage,
                epoch,
                time.perf_counter() - began,
            )
            return epoch
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load a stage worker and report its memory.")
    parser.add_argument("model", choices=sorted(PINNED))
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--stage", type=int, choices=(0, 1), help="layers from the hosts file")
    which.add_argument("--layers", help="START:END, layers START to END - 1, overrides it")
    parser.add_argument("--step-delay", type=float, default=0.0, help="seconds slept per step")
    parser.add_argument(
        "--threads", type=int, help="torch threads for this process (fix it for equality runs)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="serve Stage.Steps on this stage's port (needs --stage)",
    )
    args = parser.parse_args(argv)
    model_dir = MODELS_DIR / args.model
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.serve and args.stage is None:
        parser.error("--serve needs --stage, which names this process's port and next hop")
    try:
        if args.layers is not None:
            start, end = parse_layers(args.layers)
        else:
            start, end = stage_layers(model_dir, args.model, args.stage)
        StepDelay(args.step_delay)  # rejects a negative value
        hosts = config.load() if args.serve else None
    except (ValueError, OSError) as err:
        parser.error(str(err))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    worker = StageWorker(WorkerConfig(model_dir, start, end, args.step_delay))
    if not args.serve:
        return

    from pllm.stage_service import serve  # imports this module, so not at the top

    server, servicer, port = serve(
        worker,
        f"0.0.0.0:{getattr(hosts, f'stage{args.stage}').port}",
        next_address=hosts.stage1.address if args.stage == 0 else None,
        results_address=hosts.gateway.address if args.stage == 1 else None,
    )
    log.info("serving, rss %.0f MiB; Ctrl-C to stop", rss_mib())
    stop = threading.Event()
    failure: list[BaseException] = []
    if os.environ.get(PARENT_ENV, "").isdigit():
        exit_when_parent_gone(int(os.environ[PARENT_ENV]))

    def register_in_background():
        try:
            register_with_gateway(
                hosts.gateway.address,
                args.stage,
                (start, end),
                args.model,
                model_dir,
                getattr(hosts, f"stage{args.stage}").address,
                servicer,
                stop,
                options=wire.channel_options(worker.stage.config.hidden_size),
            )
        except BaseException as err:  # a rejected registration ends the worker, loudly
            failure.append(err)
            log.error("registration failed: %s", err)
            server.stop(grace=0)

    threading.Thread(target=register_in_background, daemon=True).start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=1).wait()
    finally:
        stop.set()
        servicer.close()
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
