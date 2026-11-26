"""Beat gaps at the gateway, for choosing the heartbeat timeout (R6.3, #124).

    uv run python -m pllm.beatgaps METRICS_FILE ... [--margin 1.5]

A gap is the time between two consecutive beats of one stage incarnation (stage and epoch)
on the gateway's clock, read from the gateway's metrics file; a restart starts a new
series. Gaps ending inside the request window (pllm.report.window: first arrival to last
token of a completed request) are "load", the rest "idle". For each stage and phase the
command prints the count, median, p99 and maximum, then the heartbeat timeout the run used,
the stage declarations the gateway made, and a verdict: pass when nothing was declared and
the longest gap is below the timeout.

It also prints a suggested timeout, the longest gap times --margin rounded up to 0.5 s. It
is a suggestion only: the members choose the value and state its margin. It must stay
below the keepalive timeout (pllm.wire), which the gateway enforces.

Each call writes a run record with the numbers and the files read.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from pllm import report, runrecord
from pllm.wire import KEEPALIVE_TIMEOUT_SECONDS

PHASES = ("idle", "load", "all")
STEP_S = 0.5  # the suggestion is rounded up to this


def gaps(run: report.Run) -> dict[int, list[tuple[int, float]]]:
    """Per stage, (end time in ns, gap in seconds) between consecutive beats of one epoch."""
    last: dict[tuple[int, int], int] = {}
    out: dict[int, list[tuple[int, float]]] = {}
    for b in sorted(run.beats, key=lambda b: b["t_ns"]):
        key = (b["stage"], b["epoch"])
        if key in last:
            out.setdefault(b["stage"], []).append((b["t_ns"], (b["t_ns"] - last[key]) / 1e9))
        last[key] = b["t_ns"]
    return out


def declarations(path: Path) -> list[dict]:
    """The detected lines of a metrics file that declared a stage dead. report.load skips
    them, so they are read here."""
    out = []
    with path.open(encoding="utf-8") as file:
        for text in file:
            try:
                row = json.loads(text)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("type") == "detected" and row.get("declared"):
                out.append(row)
    return out


def stats(values: list[float]) -> dict:
    if not values:
        return {"gaps": 0, "median_s": None, "p99_s": None, "max_s": None}
    return {
        "gaps": len(values),
        "median_s": report.percentile(values, 50),
        "p99_s": report.percentile(values, 99),
        "max_s": max(values),
    }


def summary(run: report.Run, declared: list[dict], margin: float) -> dict:
    span = report.window(run)
    stages = {}
    for stage, series in sorted(gaps(run).items()):
        load = [g for t, g in series if span is not None and span[0] <= t <= span[1]]
        idle = [g for t, g in series if span is None or not span[0] <= t <= span[1]]
        stages[stage] = {"idle": stats(idle), "load": stats(load), "all": stats(idle + load)}
    longest = max((s["all"]["max_s"] for s in stages.values()), default=None)
    timeout = run.info.get("heartbeat_timeout_s")
    suggested = None if longest is None else math.ceil(longest * margin / STEP_S) * STEP_S
    return {
        "file": str(run.path),
        "network": run.info.get("network"),
        "heartbeat_timeout_s": timeout,
        "window": span,
        "stages": stages,
        "max_gap_s": longest,
        "declared": len(declared),
        "declared_paths": [d.get("path") for d in declared],
        "margin": margin,
        "suggested_timeout_s": suggested,
        "passed": longest is not None
        and not declared
        and timeout is not None
        and longest < timeout,
    }


def _ms(seconds: float | None) -> str:
    return "" if seconds is None else f"{seconds * 1000:.1f}"


def render(s: dict) -> str:
    lines = [
        f"## {s['file']}",
        "",
        f"network {s['network']}, heartbeat timeout {s['heartbeat_timeout_s']} s, "
        f"load window {'none' if s['window'] is None else 'present'}",
        "",
        "| stage | phase | gaps | median ms | p99 ms | max ms |",
        "|---|---|---|---|---|---|",
    ]
    for stage, phases in s["stages"].items():
        for phase in PHASES:
            p = phases[phase]
            lines.append(
                f"| {stage} | {phase} | {p['gaps']} | {_ms(p['median_s'])} | "
                f"{_ms(p['p99_s'])} | {_ms(p['max_s'])} |"
            )
    lines += ["", f"stages declared dead: {s['declared']} {s['declared_paths'] or ''}".rstrip()]
    if s["max_gap_s"] is None:
        lines.append("verdict: no beat gaps in this file")
    else:
        verdict = "pass" if s["passed"] else "fail"
        lines.append(
            f"verdict: {verdict} (max gap {_ms(s['max_gap_s'])} ms, "
            f"timeout {s['heartbeat_timeout_s']} s, {s['declared']} declared)"
        )
        suggestion = s["suggested_timeout_s"]
        lines.append(
            f"suggested --heartbeat-timeout (a suggestion only): {suggestion} s "
            f"= max gap x {s['margin']}, rounded up to {STEP_S} s"
        )
        if suggestion >= KEEPALIVE_TIMEOUT_SECONDS:
            lines.append(
                f"warning: the suggestion is not below the keepalive timeout "
                f"({KEEPALIVE_TIMEOUT_SECONDS} s), which the gateway refuses"
            )
    return "\n".join(lines) + "\n"


def margin_type(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text}") from None
    if not 1 < value < float("inf"):
        raise argparse.ArgumentTypeError("the margin must be above 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Beat gaps in gateway metrics files.")
    parser.add_argument("logs", nargs="+", type=Path, help="gateway metrics files")
    parser.add_argument("--margin", type=margin_type, default=1.5, help="default 1.5")
    args = parser.parse_args(argv)
    missing = [str(p) for p in args.logs if not p.is_file()]
    if missing:
        parser.error(f"not a file: {', '.join(missing)}")
    with runrecord.record("beatgaps") as rec:
        rec["argv"] = sys.argv
        rec["results"] = []
        for path in args.logs:
            s = summary(report.load(path), declarations(path), args.margin)
            rec["results"].append(s)
            print(render(s))
    return 0 if all(s["passed"] for s in rec["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
