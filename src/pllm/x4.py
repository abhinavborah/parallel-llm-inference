"""X4: per-token latency and throughput against injected hop delay (R7.7, R7.11, decision #48).

Delay d is added to every message of the two hops that cross the LAN: gateway to Stage 0
(at the gateway's send) and Stage 0 to Stage 1 (at Stage 1's receive), both on Host A. The
Stage 1 to gateway hop stays on Host A and is not delayed. The delay is pipelined and counts
as transit (pllm.faults.HopDelay). The expectation, written as the first line before any
result, is about 2 x d added per token, so the slope of per-token latency against d is 2.0.

The gateway and Stage 1 must run with --hop-delay-control (pllm.launch passes it to both).
They read d from runs/hop_delay_ms and each acks the value it applied; the gateway also
writes a hop_delay line to its metrics file per change. Before each cell this runner writes
d and waits for both acks, so every cell's delay is confirmed, not assumed. Host B starts
once. Delays are interleaved inside each repeat, in an order rotated per repeat, so slow
drift of the network spreads over every delay instead of following d.

Each cell is an x2 pipeline cell (closed loop, c requests outstanding, one untimed warm-up
request first) at one delay and one concurrency; its line carries delay_ms and the x2
fields, TPOT mean and p50 among them. Per-token latency is TPOT on the gateway's clock.
Then one aggregate per delay and concurrency with the spread over the repeats, one fit per
concurrency (ordinary least squares of the per-cell TPOT mean against d, slope with a 95%
confidence interval, and whether it holds 2.0; stated, not asserted; the expectation is
checked at concurrency 1 only, since queueing at higher concurrency hides part of a small
delay) and the x2 summary.
The exit code is 0 only with zero ordering violations and no failed request. The control
file is set back to 0 at the end.

Usage (Host A, gateway and Stage 1 started with pllm.launch --hop-delay-control):
uv run python -m pllm.x4 smollm2-135m --metrics runs/<gateway>.metrics.jsonl
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

from pllm import client, config, faults, prompts, report, runrecord, x2
from pllm.models import PINNED

DELAYS_MS = (0, 1, 5, 20, 50)
CONCURRENCY = (1, 4)
REPEATS = 5
REQUESTS = 16
NEW_TOKENS = 32
LENGTH = 128
BASE_MS = 9  # round trip of the two LAN hops measured before #48
EXPECTED_SLOPE = 2.0
ACK_SECONDS = 10.0  # how long a delay change may take to be acked by both processes
METRICS = (
    "throughput_tok_s",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "ttft_mean_ms",
    "ttft_p50_ms",
)
# Student t at 0.975 by degrees of freedom (the default sweep has 23 per concurrency).
# shortcut: above 30 df it uses 1.96, which narrows the interval by under 4%; compute the
# real quantile if a sweep ever has more than 32 cells per concurrency.
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
    17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}  # fmt: skip


def expectation(delays: list[float], concurrency: list[int]) -> dict:
    """What decision #48 predicts, written before any result."""
    return {
        "type": "expectation",
        "decision": "#48 option 1, agreed 2026-11-07",
        "delayed_hops": ["gateway_stage0", "stage0_stage1"],
        "not_delayed": ["stage1_gateway"],
        "added_per_token_ms": {str(d): 2 * d for d in delays},
        "expected_slope": EXPECTED_SLOPE,
        "check": "95% CI of the OLS slope of per-cell TPOT mean against d holds 2.0",
        "checked_concurrency": 1,
        "above_concurrency_1": "queueing at small d hides part of the delay, so a slope "
        "below 2.0 is expected there; it is reported, not checked",
        "throughput": "about min(c / T, 1 / slowest stage time), T = compute + base + 2 x d",
        "concurrency": list(concurrency),
    }


def fit(xs: list[float], ys: list[float]) -> dict:
    """Least-squares slope of ys against xs with its 95% confidence interval."""
    n = len(xs)
    out = {"n": n, "slope": None, "intercept": None, "se": None, "ci95": None}
    if n < 3 or len(set(xs)) < 2:
        return out
    slope, intercept = statistics.linear_regression(xs, ys)
    mx = statistics.fmean(xs)
    sxx = sum((x - mx) ** 2 for x in xs)
    ssr = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys, strict=True))
    se = math.sqrt(ssr / (n - 2) / sxx)
    t = T975.get(n - 2, 1.96)
    out.update(slope=slope, intercept=intercept, se=se, ci95=[slope - t * se, slope + t * se])
    return out


def fit_lines(cells: list[dict]) -> list[dict]:
    """One fit per concurrency of TPOT mean (ms) against delay (ms)."""
    out = []
    for c in sorted({line["concurrency"] for line in cells}):
        rows = [
            line for line in cells if line["concurrency"] == c and line["tpot_mean_ms"] is not None
        ]
        result = fit([r["delay_ms"] for r in rows], [r["tpot_mean_ms"] for r in rows])
        ci = result["ci95"]
        out.append(
            {
                "type": "fit",
                "concurrency": c,
                "x": "delay_ms",
                "y": "tpot_mean_ms",
                **result,
                "expected_slope": EXPECTED_SLOPE,
                "checked": c == 1,  # the expectation of #48 is for concurrency 1
                "holds_expected": None if ci is None else ci[0] <= EXPECTED_SLOPE <= ci[1],
            }
        )
    return out


def aggregate(cells: list[dict]) -> list[dict]:
    """One line per delay and concurrency with the spread of each metric over the repeats."""
    groups: dict[tuple[float, int], list[dict]] = {}
    for line in cells:
        groups.setdefault((line["delay_ms"], line["concurrency"]), []).append(line)
    return [
        {
            "type": "aggregate",
            "delay_ms": d,
            "concurrency": c,
            "repeats": len(lines),
            **{name: x2.spread([line[name] for line in lines]) for name in METRICS},
            "failed_requests": sum(line["failed_requests"] for line in lines),
        }
        for (d, c), lines in sorted(groups.items())
    ]


def rotated(delays: list[float], repeat: int) -> list[float]:
    """The delays in the order of one repeat: each repeat starts one further along."""
    k = repeat % len(delays)
    return [*delays[k:], *delays[:k]]


def set_delay(path: Path, ms: float, timeout: float = ACK_SECONDS) -> None:
    """Write d to the control file and wait until the gateway and Stage 1 both ack it."""
    faults.write_atomic(path, f"{ms!r}\n")
    end = time.monotonic() + timeout
    while True:
        acked = [faults.read_delay_ms(p) for p in faults.ack_paths(path)]
        if all(path_ms == ms for path_ms in acked):
            return
        if time.monotonic() > end:
            names = [n for n, a in zip(faults.CONTROLLED, acked, strict=True) if a != ms]
            raise TimeoutError(
                f"no ack of {ms} ms from {', '.join(names)} in {timeout} s: "
                "were they started with --hop-delay-control?"
            )
        time.sleep(faults.CONTROL_SECONDS / 2)


def sweep(call, metrics_path: Path, args, prompt_ids, emit, control: Path) -> list[dict]:
    """Every cell, repeat by repeat, delays rotated inside, then aggregates, fits, summary."""
    cells = []
    emit(expectation(args.delays, args.concurrency))
    try:
        for repeat in range(args.repeats):
            for d in rotated(args.delays, repeat):
                set_delay(control, d)
                for c in args.concurrency:
                    x2.serve_pipeline(call, x2.jobs(prompt_ids, 1), 1)  # warm-up
                    cell = x2.jobs(prompt_ids, args.requests)
                    x2.serve_pipeline(call, cell, c)
                    time.sleep(args.settle)
                    run, missing = x2.cell_run(metrics_path, {j.rid for j in cell if j.rid})
                    line = x2.cell_line("pipeline", c, repeat, cell, run, missing)
                    cells.append({"type": "cell", "delay_ms": d, **line})
                    emit(cells[-1])
    finally:
        faults.write_atomic(control, "0\n")  # back to no delay; acks follow by themselves
    for line in [*aggregate(cells), *fit_lines(cells)]:
        emit(line)
    emit(x2.summary_line(cells))
    return cells


def timeout_floor_s(new_tokens: int, delays: list[float]) -> float:
    """Twice the network time of one request at the largest delay, prefill included."""
    return 2 * (new_tokens + 1) * (2 * max(delays) + BASE_MS) / 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X4: latency against injected hop delay.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--metrics", type=Path, required=True, help="the gateway's metrics file")
    parser.add_argument("--delays", type=float, nargs="+", default=list(DELAYS_MS), help="ms")
    parser.add_argument("--concurrency", type=int, nargs="+", default=list(CONCURRENCY))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--requests", type=int, default=REQUESTS, help="per cell")
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument("--length", type=int, choices=prompts.LENGTHS, default=LENGTH)
    parser.add_argument("--settle", type=float, default=x2.SETTLE_S, help="seconds")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    parser.add_argument("--control-file", type=Path, default=faults.HOP_DELAY_FILE)
    args = parser.parse_args(argv)
    if min(args.concurrency) < 1 or args.repeats < 1 or args.new_tokens < 1:
        parser.error("--concurrency, --repeats and --new-tokens must be positive")
    if not all(0 <= d < float("inf") for d in args.delays) or len(set(args.delays)) < 2:
        parser.error("--delays needs at least two different non-negative values")
    if args.requests < 2 * max(args.concurrency):
        parser.error("--requests must be at least twice the largest concurrency")
    floor = timeout_floor_s(args.new_tokens, args.delays)
    if args.timeout < floor:
        parser.error(f"--timeout must cover the delay: at least {floor:.1f} s here")
    if not args.metrics.is_file():
        parser.error("--metrics must be the running gateway's metrics file")
    info = report.load(args.metrics).info
    if info.get("model") != args.model:
        parser.error(f"the gateway serves {info.get('model')}, not {args.model}")
    if not info.get("hop_delay_control"):
        parser.error("the gateway was not started with --hop-delay-control")

    spec = PINNED[args.model]
    prompt_ids = prompts.prompts(args.model, args.length)
    with runrecord.record(
        "x4", model=args.model, revision=spec.revision, split=config.split_point(args.model)
    ) as rec:
        rec.update(
            delays_ms=args.delays,
            concurrency=args.concurrency,
            repeats=args.repeats,
            requests=args.requests,
            new_tokens=args.new_tokens,
            length=args.length,
            settle_s=args.settle,
            timeout_s=args.timeout,
            metrics_file=args.metrics.name,
        )
        lines_path = runrecord.output_path(rec, ".jsonl")
        rec["results_file"] = lines_path.name
        tokenizer = prompts.load_tokenizer(args.model)
        address = config.load().gateway.address

        def call(job):
            text = tokenizer.decode(job.ids, skip_special_tokens=False)
            return client.generate(address, text, args.new_tokens, args.timeout)

        with lines_path.open("w", encoding="utf-8") as out:

            def emit(line: dict) -> None:
                text = json.dumps(line, allow_nan=False)
                out.write(text + "\n")
                print(text, flush=True)

            cells = sweep(call, args.metrics, args, prompt_ids, emit, args.control_file)
        rec["fits"] = fit_lines(cells)
        rec["summary"] = x2.summary_line(cells)
    return 0 if rec["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
