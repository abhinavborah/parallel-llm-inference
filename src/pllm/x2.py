"""X2: throughput against concurrency, the pipeline and single-process round-robin (R7.9).

Both modes serve the same requests in a closed loop: c requests are outstanding at once and
a finished request is replaced by the next at once, N requests per cell. Prompts are the
prompt set's passages of one length in order, greedy decoding stops at max tokens or end of
sequence. One untimed warm-up request runs before each cell. Cells run repeat by repeat,
concurrency inside, so slow drift spreads over every cell.

- pipeline: a client of a gateway already started with pllm.launch (address from the hosts
  file). --metrics names the gateway's metrics file. After each cell the runner waits for
  a heartbeat past the window, then keeps that file's request lines of the cell's requests
  and every beat, and pllm.report computes the numbers. Latencies come from the gateway's
  one clock only.
- single: one process with the whole model serves the requests round-robin without
  batching: each turn every running request takes one step, a prefill is one step, at most
  gateway.MAX_RUNNING run and the rest wait in arrival order, as at the gateway. It writes
  its own metrics file in the gateway's format (request lines, and stage 0 beats with the
  time spent in forward), so pllm.report reads it the same way.

Throughput, TTFT, TPOT and stage utilisation are as defined in pllm.report. Ordering is
checked two ways in pipeline mode: the steps the gateway sent to Stage 0 and received from
Stage 1 rise per (rid, attempt), read from its hop lines, and each client stream carries
steps 0 to n - 1. The stages' own ordering counters (R3.7) are not exported, so they are not
read here. Single mode has no hops and records no ordering counts.

One JSON line per cell, one aggregate per mode and concurrency with the spread over the
repeats (n, mean, stdev, min, max), then a summary, go to a .jsonl file next to the run
record in runs/ and to stdout. The exit code is 0 only with zero ordering violations and no
failed request. Single mode records the process's lifetime peak RSS (after load and at the
end), its RSS just after load, and the largest RSS sampled at its beats while serving, so a
load peak and the settled size are told apart. --budget-mib names the budget they are read
against; the record states it and asserts nothing (decision 57).

Usage:
uv run python -m pllm.x2 pipeline smollm2-135m --metrics runs/<gateway>.metrics.jsonl
uv run python -m pllm.x2 single smollm2-135m --threads 12
"""

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import grpc
import psutil
import torch

from pllm import client, config, launch, metrics, ordering, prompts, report, runrecord
from pllm.gateway import MAX_RUNNING, eos_ids
from pllm.models import MODELS_DIR, PINNED, verify
from pllm.proto import protocol_pb2

CONCURRENCY = (1, 2, 4, 8, 16)
REPEATS = 5
REQUESTS = 32
NEW_TOKENS = 64
LENGTH = 128
SETTLE_S = 1.5  # after a cell, long enough for a stage beat (every 0.5 s) past its window
BEAT_NS = 500_000_000  # single mode writes a beat this often, as a stage does
NOT_EXPORTED = (
    "stage-internal ordering counters (R3.7) are not exported; ordering is checked at the "
    "gateway's hops and per client stream"
)
METRICS = (
    "throughput_tok_s",
    *(f"{key}_p{p}_ms" for key in ("ttft", "tpot") for p in report.PERCENTILES),
    *(f"stage{stage}_utilisation" for stage in report.STAGES),
)


@dataclass
class Job:
    """One request of a cell: its prompt and, once served, what came back."""

    prompt: int  # index in the prompt set
    ids: list[int]
    rid: str | None = None
    tokens: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    error: str | None = None
    # single mode only, on its recorder's clock
    arrived_ns: int | None = None
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    cache: object = None


def jobs(prompt_ids: list[list[int]], count: int) -> list[Job]:
    """The cell's requests: the prompts in order, from the first again if count is larger."""
    return [Job(i % len(prompt_ids), prompt_ids[i % len(prompt_ids)]) for i in range(count)]


def tokens_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ single process


class SingleProcess:
    """The whole model in this process, serving requests round-robin without batching.

    model needs new_cache() and forward(tokens, start_pos, cache) -> logits per position.
    Busy time and step totals run across cells, like a stage's heartbeat totals.
    """

    def __init__(self, model, recorder: metrics.Recorder, eos: list[int], cap: int = MAX_RUNNING):
        self.model = model
        self.recorder = recorder
        self.eos = frozenset(eos)
        self.cap = cap
        self.busy_ns = 0
        self.steps = 0
        self._last_beat: int | None = None
        self.rss_serving_max = 0  # largest RSS sampled at a beat, in bytes

    def beat(self, force: bool = False) -> None:
        now = self.recorder.now()
        if force or self._last_beat is None or now - self._last_beat >= BEAT_NS:
            self._last_beat = now
            self.rss_serving_max = max(self.rss_serving_max, rss())
            beat = protocol_pb2.Beat(stage=0, epoch=0, busy_ns=self.busy_ns, steps=self.steps)
            self.recorder.beat(0, beat)

    def step(self, job: Job, new_tokens: int) -> bool:
        """One forward for job, prefill or one decode step; True when job has ended."""
        if job.cache is None:
            job.cache = self.model.new_cache()
            inputs, start = torch.tensor(job.ids), 0
        else:
            inputs, start = torch.tensor([job.tokens[-1]]), len(job.ids) + len(job.tokens) - 1
        began = self.recorder.now()
        logits = self.model.forward(inputs, start, job.cache)[-1]
        self.busy_ns += self.recorder.now() - began
        self.steps += 1
        _, best = logits.max(dim=-1)  # as pllm.reference.decode
        job.tokens.append(best.item())
        job.steps.append(len(job.tokens) - 1)
        job.last_token_ns = self.recorder.now()
        if job.first_token_ns is None:
            job.first_token_ns = job.last_token_ns
        return job.tokens[-1] in self.eos or len(job.tokens) >= new_tokens

    def end(self, job: Job, new_tokens: int) -> None:
        job.cache = None  # frees its KV cache
        outcome = "end of sequence" if job.tokens[-1] in self.eos else "max_tokens"
        self.recorder.request(
            outcome,
            None,
            rid=job.rid,
            attempt=0,
            prompt_tokens=len(job.ids),
            max_tokens=new_tokens,
            output_tokens=len(job.tokens),
            arrived_ns=job.arrived_ns,
            first_token_ns=job.first_token_ns,
            last_token_ns=job.last_token_ns,
        )

    def serve(self, cell: list[Job], concurrency: int, new_tokens: int, prefix: str) -> None:
        """Serve cell in a closed loop of concurrency outstanding requests."""
        pending = deque(cell)
        waiting: deque[Job] = deque()
        running: deque[Job] = deque()
        number = 0

        def arrive() -> None:
            nonlocal number
            if pending:
                job = pending.popleft()
                job.rid, job.arrived_ns = f"{prefix}-{number}", self.recorder.now()
                number += 1
                waiting.append(job)
            while waiting and len(running) < self.cap:
                running.append(waiting.popleft())  # joins the end of the turn

        self.beat(force=True)
        for _ in range(concurrency):
            arrive()
        while running:
            job = running.popleft()
            if self.step(job, new_tokens):
                self.end(job, new_tokens)
                arrive()
            else:
                running.append(job)
            self.beat()
        self.beat(force=True)


def rss() -> int:
    """This process's resident set now, in bytes."""
    return psutil.Process().memory_info().rss


def peak_rss() -> int:
    """This process's peak resident set in bytes, on every operating system."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset
    import resource  # not on Windows

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux gives KiB


# ------------------------------------------------------------------ pipeline


def serve_pipeline(call: Callable[[Job], Iterator], cell: list[Job], concurrency: int) -> None:
    """Serve cell through the gateway with concurrency client threads in a closed loop.

    call(job) yields the gateway's Tokens for one request.
    """
    pending = deque(cell)
    lock = threading.Lock()

    def worker() -> None:
        while True:
            with lock:
                if not pending:
                    return
                job = pending.popleft()
            try:
                for token in call(job):
                    job.rid = token.rid
                    job.tokens.append(token.token)
                    job.steps.append(token.step)
            except grpc.RpcError as err:
                job.error = f"{err.code().name}: {err.details()}"

    with ThreadPoolExecutor(concurrency) as pool:
        for future in [pool.submit(worker) for _ in range(concurrency)]:
            future.result()


def read_hops(path: Path, rids: set[str]) -> list[dict]:
    """The hop lines of these requests, in file order. A cut line is skipped."""
    out = []
    with Path(path).open(encoding="utf-8") as file:
        for text in file:
            try:
                row = json.loads(text)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("type") == "hop" and row.get("rid") in rids:
                out.append(row)
    return out


def hop_violations(hops: list[dict]) -> int:
    """Messages whose step does not rise for their (rid, attempt), per gateway hop."""
    checks: dict[str, ordering.StepOrder] = {}
    for row in hops:
        hop = row["hop"]
        order = checks.setdefault(hop, ordering.StepOrder(f"x2 {hop}"))
        order.check(row["rid"], row["attempt"], row["step"])
    return sum(order.violations for order in checks.values())


def stream_violations(cell: list[Job]) -> int:
    """Requests whose streamed steps are not exactly 0 to n - 1."""
    return sum(job.steps != list(range(len(job.steps))) for job in cell)


# ------------------------------------------------------------------ results


def cell_run(path: Path, rids: set[str]) -> tuple[report.Run, int]:
    """The metrics file reduced to these requests (all beats kept), and how many of the
    requests it has no line for."""
    run = report.load(Path(path))
    run.requests = [r for r in run.requests if r.get("rid") in rids]
    return run, len(rids - {r.get("rid") for r in run.requests})


def cell_line(mode: str, concurrency: int, repeat: int, cell: list[Job], run, missing: int):
    s = report.summary(run)
    line = {
        "type": "cell",
        "mode": mode,
        "concurrency": concurrency,
        "repeat": repeat,
        "requests": len(cell),
        "completed": s["completed"],
        "outcomes": s["outcomes"],
        "failed_requests": sum(job.error is not None for job in cell),
        "errors": sorted({job.error for job in cell if job.error}),
        "missing_in_metrics": missing,
        "output_tokens": s["output_tokens"],
        "window_s": s["window_s"],
        "throughput_tok_s": s["throughput_tok_s"],
    }
    for key in ("ttft", "tpot"):
        row = s["latency"][key]
        line[f"{key}_n"] = row["n"]
        line[f"{key}_mean_ms"] = row.get("mean")
        for p in report.PERCENTILES:
            line[f"{key}_p{p}_ms"] = row.get(f"p{p}")
    for stage in report.STAGES:
        v = s["stages"].get(stage, {})
        line[f"stage{stage}_utilisation"] = v.get("utilisation")
        line[f"stage{stage}_max_beat_gap_s"] = v.get("max_beat_gap_s")
    if mode == "pipeline":
        line["hop_order_violations"] = hop_violations(read_hops(run.path, {j.rid for j in cell}))
        line["stream_step_violations"] = stream_violations(cell)
    else:
        line["hop_order_violations"] = line["stream_step_violations"] = None
    # Same host, fp32, batch 1: the two modes should give the same tokens per prompt.
    line["outputs"] = {str(job.prompt): tokens_sha256(job.tokens) for job in cell}
    return line


def spread(values: list) -> dict:
    """n, mean, stdev, min and max of the values that exist; stdev needs two."""
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else None,
        "min": min(xs),
        "max": max(xs),
    }


def aggregate(cells: list[dict]) -> list[dict]:
    """One line per mode and concurrency with the spread of every metric over the repeats."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for line in cells:
        groups.setdefault((line["mode"], line["concurrency"]), []).append(line)
    out = []
    for (mode, concurrency), lines in sorted(groups.items()):
        row = {
            "type": "aggregate",
            "mode": mode,
            "concurrency": concurrency,
            "repeats": len(lines),
            **{name: spread([line[name] for line in lines]) for name in METRICS},
            "failed_requests": sum(line["failed_requests"] for line in lines),
        }
        for key in ("hop_order_violations", "stream_step_violations"):
            counts = [line[key] for line in lines if line[key] is not None]
            row[key] = sum(counts) if counts else None
        out.append(row)
    return out


def summary_line(cells: list[dict]) -> dict:
    violations = sum(
        line[key] or 0
        for line in cells
        for key in ("hop_order_violations", "stream_step_violations")
    )
    failed = sum(line["failed_requests"] + line["missing_in_metrics"] for line in cells)
    return {
        "type": "summary",
        "cells": len(cells),
        "ordering_violations": violations,
        "failed_or_missing_requests": failed,
        "passed": violations == 0 and failed == 0,
        "ordering_note": NOT_EXPORTED,
    }


# ------------------------------------------------------------------ command line


def sweep(serve_cell, settle, metrics_path, args, prompt_ids, out) -> list[dict]:
    """Run every cell, write its line, then the aggregates and the summary."""
    cells = []

    def emit(line: dict) -> None:
        text = json.dumps(line, allow_nan=False)
        out.write(text + "\n")
        print(text, flush=True)

    for repeat in range(args.repeats):
        for concurrency in args.concurrency:
            serve_cell(jobs(prompt_ids, 1), 1, f"warmup-{repeat}-{concurrency}")
            cell = jobs(prompt_ids, args.requests)
            serve_cell(cell, concurrency, f"x2-{repeat}-{concurrency}")
            settle()
            run, missing = cell_run(metrics_path(), {job.rid for job in cell if job.rid})
            cells.append(cell_line(args.mode, concurrency, repeat, cell, run, missing))
            emit(cells[-1])
    for line in aggregate(cells):
        emit(line)
    emit(summary_line(cells))
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X2: throughput against concurrency.")
    parser.add_argument("mode", choices=("pipeline", "single"))
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--concurrency", type=int, nargs="+", default=list(CONCURRENCY))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--requests", type=int, default=REQUESTS, help="per cell")
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument("--length", type=int, choices=prompts.LENGTHS, default=LENGTH)
    parser.add_argument("--metrics", type=Path, help="pipeline: the gateway's metrics file")
    parser.add_argument("--settle", type=float, default=SETTLE_S, help="pipeline: seconds")
    parser.add_argument("--timeout", type=float, default=300.0, help="pipeline: per request")
    parser.add_argument("--threads", type=int, help="single: torch threads (default: cores)")
    parser.add_argument(
        "--budget-mib", type=float, help="single: the budget peak RSS is read against"
    )
    args = parser.parse_args(argv)
    if min(args.concurrency) < 1 or args.repeats < 1 or args.new_tokens < 1:
        parser.error("--concurrency, --repeats and --new-tokens must be positive")
    if args.requests < 2 * max(args.concurrency):
        parser.error("--requests must be at least twice the largest concurrency")
    if args.mode == "pipeline":
        if args.metrics is None or not args.metrics.is_file():
            parser.error("pipeline needs --metrics, the running gateway's metrics file")
        served = report.load(args.metrics).info.get("model")
        if served != args.model:
            parser.error(f"the gateway serves {served}, not {args.model}")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")

    spec = PINNED[args.model]
    model_dir = MODELS_DIR / args.model
    prompt_ids = prompts.prompts(args.model, args.length)
    settings = {
        "mode": args.mode,
        "concurrency": args.concurrency,
        "repeats": args.repeats,
        "requests": args.requests,
        "new_tokens": args.new_tokens,
        "length": args.length,
        "running_cap": MAX_RUNNING,
        "ordering_note": NOT_EXPORTED,
    }
    if args.mode == "pipeline":
        settings.update(settle_s=args.settle, timeout_s=args.timeout)
        record = runrecord.record(
            "x2", model=args.model, revision=spec.revision, split=config.split_point(args.model)
        )
    else:
        threads = args.threads or launch.default_threads(["stage0"])
        torch.set_num_threads(threads)
        record = runrecord.record("x2", model=args.model, revision=spec.revision, threads=threads)

    with record as rec:
        rec.update(settings)
        lines_path = runrecord.output_path(rec, ".jsonl")
        rec["results_file"] = lines_path.name
        if args.mode == "pipeline":
            rec["metrics_file"] = args.metrics.name
            tokenizer = prompts.load_tokenizer(args.model)
            address = config.load().gateway.address

            def call(job):
                text = tokenizer.decode(job.ids, skip_special_tokens=False)
                return client.generate(address, text, args.new_tokens, args.timeout)

            def serve_cell(cell, concurrency, _prefix):
                serve_pipeline(call, cell, concurrency)

            def settle():
                time.sleep(args.settle)

            metrics_path = lambda: args.metrics  # noqa: E731
        else:
            for filename, expected in spec.files.items():
                verify(model_dir / filename, expected)
            metrics_file = runrecord.output_path(rec, ".metrics.jsonl")
            rec["metrics_file"] = metrics_file.name
            recorder = metrics.Recorder(
                metrics_file, {"model": args.model, "commit": rec["commit"], "mode": "single"}
            )
            from pllm.model import Model

            server = SingleProcess(Model.load(model_dir), recorder, eos_ids(model_dir))
            rec["peak_rss_after_load_bytes"] = peak_rss()
            rec["rss_after_load_bytes"] = rss()

            def serve_cell(cell, concurrency, prefix):
                server.serve(cell, concurrency, args.new_tokens, prefix)

            def settle():
                pass

            metrics_path = lambda: metrics_file  # noqa: E731
        try:
            with lines_path.open("w", encoding="utf-8") as out:
                cells = sweep(serve_cell, settle, metrics_path, args, prompt_ids, out)
        finally:
            if args.mode == "single":
                recorder.finish()
                rec["peak_rss_bytes"] = peak_rss()
                rec["rss_serving_max_bytes"] = server.rss_serving_max
                rec["budget_mib"] = args.budget_mib
        rec["summary"] = summary_line(cells)
    return 0 if rec["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
