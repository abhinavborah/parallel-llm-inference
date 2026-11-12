"""Every plot and table from the gateway's metrics files (R7.3 to R7.6).

    uv run python -m pllm.report                 every runs/*.metrics.jsonl
    uv run python -m pllm.report LOG_OR_DIR ...  the files given, or those in each folder
    --out DIR                                    where to write (default runs/report)

Nothing is typed in by hand: each run's tables and plots are rebuilt from its file, and the
output folder is written again from scratch, so a fresh checkout and the logs give the same
files. Reported numbers come from here, never from the dashboard.

Definitions (as in vLLM's serving benchmark, read 2026-11-10). Every time comes from the
gateway's one clock (pllm.metrics), so no two machines' clocks are ever compared.

- A request is completed when it ended at end of sequence or at max_tokens. Only completed
  requests give latencies and throughput; the others are counted by outcome.
- TTFT: first token minus arrival at the gateway, so time in the admission queue counts.
- TPOT: (last token minus first token) / (output tokens - 1), for requests with at least
  two tokens.
- End-to-end: last token minus arrival.
- Percentiles interpolate linearly between the two nearest ranks (numpy's default); each
  table gives n, since a p99 of fewer than 100 values is an interpolation.
- Throughput: output tokens of completed requests over the window, which runs from the
  first arrival of any request to the last token of a completed one.
- Busy time per stage: each stage reports running totals of the time it spent handling
  steps in its heartbeats (every 0.5 s), counting a step still running, and the gateway
  logs each beat on arrival. The busy time in the window is read off those totals by
  straight-line interpolation between beats, per stage incarnation (epoch), since a
  restarted stage counts from zero again. The error at each end of the window is at most
  the time between the two beats around it; max_beat_gap_s gives the longest stretch of the
  window with no beat from that stage (a late first beat, a heartbeat that reconnected,
  beats that stopped), and the summary warns when it is over 1 s. Utilisation is busy time
  over the window; two streams working at once on one stage can push it above 1.
"""

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

GAP_WARNING_S = 1.0  # a beat gap longer than this is called out in the summary
RUNS = Path("runs")
OUT = RUNS / "report"
PATTERN = "*.metrics.jsonl"
MARKER = ".pllm-report"  # marks a folder this script made, so only such a folder is replaced
COMPLETED = ("end of sequence", "max_tokens")
PERCENTILES = (50, 95, 99)
LATENCIES = (("ttft", "TTFT"), ("tpot", "TPOT"), ("e2e", "End-to-end"))
STAGES = (0, 1)
REQUIRED = {  # a line missing one of these is counted as unreadable
    "request": ("outcome", "output_tokens", "arrived_ns", "first_token_ns", "last_token_ns"),
    "beat": ("t_ns", "stage", "epoch", "busy_ns", "steps"),
    "bytes": ("hop", "msg", "phase", "messages", "bytes", "payload_bytes", "positions"),
}


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between the closest ranks, as numpy.percentile by default."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    h = (len(ordered) - 1) * p / 100
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


@dataclass
class Run:
    """One metrics file, read."""

    path: Path
    info: dict = field(default_factory=dict)
    requests: list[dict] = field(default_factory=list)
    beats: list[dict] = field(default_factory=list)
    bytes: list[dict] = field(default_factory=list)
    end: dict | None = None
    bad_lines: int = 0

    @property
    def name(self) -> str:
        return self.path.name.removesuffix(".jsonl").removesuffix(".metrics")


def load(path: Path) -> Run:
    """Read a metrics file. A line that is not a JSON object (a run cut short mid-write)
    is counted, not fatal."""
    run = Run(path)
    with path.open(encoding="utf-8") as file:
        for text in file:
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except ValueError:
                run.bad_lines += 1
                continue
            if not isinstance(row, dict):
                run.bad_lines += 1
                continue
            kind = row.get("type")
            if any(key not in row for key in REQUIRED.get(kind, ())):
                run.bad_lines += 1
                continue
            if kind == "run":
                run.info = row
            elif kind == "request":
                run.requests.append(row)
            elif kind == "beat":
                run.beats.append(row)
            elif kind == "bytes":
                run.bytes.append(row)
            elif kind == "run_end":
                run.end = row
    return run


def completed(run: Run) -> list[dict]:
    return [
        r
        for r in run.requests
        if r.get("outcome") in COMPLETED
        and r.get("output_tokens", 0) >= 1
        and r.get("first_token_ns") is not None
        and r.get("last_token_ns") is not None
    ]


def latencies(run: Run) -> dict[str, list[float]]:
    """Per completed request, in milliseconds."""
    out = {"ttft": [], "tpot": [], "e2e": []}
    for r in completed(run):
        arrived, first, last = r["arrived_ns"], r["first_token_ns"], r["last_token_ns"]
        out["ttft"].append((first - arrived) / 1e6)
        out["e2e"].append((last - arrived) / 1e6)
        if r["output_tokens"] >= 2:
            out["tpot"].append((last - first) / 1e6 / (r["output_tokens"] - 1))
    return out


def window(run: Run) -> tuple[int, int] | None:
    """From the first arrival of any request to the last token of a completed one."""
    done = completed(run)
    arrivals = [r["arrived_ns"] for r in run.requests if r.get("arrived_ns") is not None]
    if not done or not arrivals:
        return None
    start, end = min(arrivals), max(r["last_token_ns"] for r in done)
    return (start, end) if end > start else None


def _at(series: list[tuple[int, int]], t: int) -> float:
    """A running total at time t, interpolated between beats and held flat outside them."""
    if t <= series[0][0]:
        return series[0][1]
    if t >= series[-1][0]:
        return series[-1][1]
    for (t0, v0), (t1, v1) in zip(series, series[1:], strict=False):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return series[-1][1]


def series(run: Run, stage: int, key: str) -> dict[int, list[tuple[int, int]]]:
    """Per epoch, the (gateway time, running total) points of one stage's beats."""
    out: dict[int, list[tuple[int, int]]] = {}
    for b in sorted(run.beats, key=lambda b: b["t_ns"]):
        if b.get("stage") == stage:
            out.setdefault(b["epoch"], []).append((b["t_ns"], b[key]))
    return out


def max_gap(run: Run, stage: int, start: int, end: int) -> float | None:
    """The longest stretch of the window not between two beats of this stage, in seconds,
    counting the beats just outside the window; None with no beats."""
    times = sorted(b["t_ns"] for b in run.beats if b.get("stage") == stage)
    if not times:
        return None
    before = [t for t in times if t <= start]
    after = [t for t in times if t >= end]
    points = [before[-1] if before else start]
    points += [t for t in times if start < t < end]
    points.append(after[0] if after else end)
    return max(b - a for a, b in zip(points, points[1:], strict=False)) / 1e9


def busy(run: Run, stage: int, start: int, end: int, key: str = "busy_ns") -> float | None:
    """The stage's busy nanoseconds (or steps) between start and end; None with no beats."""
    epochs = series(run, stage, key)
    if not epochs:
        return None
    return sum(_at(points, end) - _at(points, start) for points in epochs.values())


def summary(run: Run) -> dict:
    """The numbers of one run, ready for its tables."""
    done = completed(run)
    lat = latencies(run)
    span = window(run)
    outcomes: dict[str, int] = {}
    for r in run.requests:
        outcomes[r.get("outcome", "unknown")] = outcomes.get(r.get("outcome", "unknown"), 0) + 1
    out = {
        "run": run.name,
        "model": run.info.get("model"),
        "commit": run.info.get("commit"),
        "network": run.info.get("network"),
        "requests": len(run.requests),
        "completed": len(done),
        "outcomes": outcomes,
        "output_tokens": sum(r["output_tokens"] for r in done),
        "window_s": None,
        "throughput_tok_s": None,
        "latency": {},
        "stages": {},
        "complete_run": run.end is not None,
        "stage0_stage1_complete": (run.end or {}).get("stage0_stage1_complete"),
        "bad_lines": run.bad_lines,
    }
    for key, _ in LATENCIES:
        values = lat[key]
        row = {"n": len(values)}
        if values:
            row["mean"] = sum(values) / len(values)
            for p in PERCENTILES:
                row[f"p{p}"] = percentile(values, p)
        out["latency"][key] = row
    if span is not None:
        seconds = (span[1] - span[0]) / 1e9
        out["window_s"] = seconds
        out["throughput_tok_s"] = out["output_tokens"] / seconds
        for stage in STAGES:
            ns = busy(run, stage, *span)
            steps = busy(run, stage, *span, key="steps")
            out["stages"][stage] = {
                "busy_s": None if ns is None else ns / 1e9,
                "utilisation": None if ns is None else ns / 1e9 / seconds,
                "steps": None if steps is None else round(steps),
                "max_beat_gap_s": max_gap(run, stage, *span),
            }
    return out


# ------------------------------------------------------------------ tables


def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return lines


def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows([["" if v is None else v for v in row] for row in rows])


LATENCY_HEADER = ["metric", "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms"]
STAGE_HEADER = ["stage", "busy_s", "utilisation", "steps", "max_beat_gap_s"]
BYTES_HEADER = [
    "hop",
    "msg",
    "phase",
    "messages",
    "bytes",
    "payload_bytes",
    "positions",
    "bytes_per_position",
    "source",
]


def latency_rows(s: dict) -> list[list]:
    rows = []
    for key, label in LATENCIES:
        row = s["latency"][key]
        rows.append([label, row["n"], row.get("mean"), *(row.get(f"p{p}") for p in PERCENTILES)])
    return rows


def stage_rows(s: dict) -> list[list]:
    return [[stage, *(v[key] for key in STAGE_HEADER[1:])] for stage, v in s["stages"].items()]


def bytes_rows(run: Run) -> list[list]:
    rows = []
    for b in run.bytes:
        per = b["bytes"] / b["positions"] if b.get("positions") else None
        rows.append([*(b[key] for key in BYTES_HEADER[:7]), per, b["source"]])
    return rows


def write_run(run: Run, s: dict, folder: Path) -> None:
    folder.mkdir(parents=True)
    _csv(folder / "latency.csv", LATENCY_HEADER, latency_rows(s))
    _csv(folder / "stages.csv", STAGE_HEADER, stage_rows(s))
    _csv(folder / "bytes.csv", BYTES_HEADER, bytes_rows(run))
    outcome_rows = sorted(s["outcomes"].items())
    text = [f"# {s['run']}", ""]
    text += [
        f"Model {_fmt(s['model'])}, commit {_fmt(s['commit'])}, network {_fmt(s['network'])}.",
        f"Requests {s['requests']}, completed {s['completed']}, output tokens "
        f"{s['output_tokens']}, window {_fmt(s['window_s'])} s, throughput "
        f"{_fmt(s['throughput_tok_s'])} output tokens/s.",
    ]
    if not s["complete_run"]:
        text.append("The run has no end line: it was cut short, and byte totals are missing.")
    if s["stage0_stage1_complete"] is False:
        text.append("A Stage 0 stream gave no byte report: the stage0_stage1 totals are low.")
    for stage, v in s["stages"].items():
        gap = v["max_beat_gap_s"]
        if gap is not None and gap > GAP_WARNING_S:
            text.append(
                f"Stage {stage} sent no beat for {gap:.1f} s within the window, so its busy "
                f"time may be off by up to that much."
            )
    if s["bad_lines"]:
        text.append(f"{s['bad_lines']} unreadable lines were skipped.")
    text += ["", "## Latency (ms)", ""] + _markdown(LATENCY_HEADER, latency_rows(s))
    text += ["", "## Outcomes", ""] + _markdown(["outcome", "requests"], outcome_rows)
    text += ["", "## Stages (busy time within the window)", ""]
    text += _markdown(STAGE_HEADER, stage_rows(s)) if s["stages"] else ["No window."]
    text += ["", "## Bytes per hop", ""] + _markdown(BYTES_HEADER, bytes_rows(run))
    (folder / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    plot_run(run, folder)


RUNS_HEADER = [
    "run",
    "model",
    "requests",
    "completed",
    "output_tokens",
    "window_s",
    "throughput_tok_s",
    *[f"{key}_p{p}_ms" for key, _ in LATENCIES for p in PERCENTILES],
    *[f"stage{stage}_utilisation" for stage in STAGES],
]


def runs_row(s: dict) -> list:
    return [
        s["run"],
        s["model"],
        s["requests"],
        s["completed"],
        s["output_tokens"],
        s["window_s"],
        s["throughput_tok_s"],
        *[s["latency"][key].get(f"p{p}") for key, _ in LATENCIES for p in PERCENTILES],
        *[s["stages"].get(stage, {}).get("utilisation") for stage in STAGES],
    ]


# ------------------------------------------------------------------ plots


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")  # files only, no display: works on every laptop and in CI
    from matplotlib import pyplot

    return pyplot


def plot_run(run: Run, folder: Path) -> None:
    """latency_cdf.png and, with beats, utilisation.png."""
    plt = _pyplot()
    lat = latencies(run)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (key, label) in zip(axes, LATENCIES, strict=True):
        values = sorted(lat[key])
        if values:
            ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post")
        ax.set(title=f"{label} (n={len(values)})", xlabel="ms", ylabel="fraction of requests")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "latency_cdf.png", dpi=120)
    plt.close(fig)
    if not run.beats:
        return
    origin = run.info.get("start_ns", min(b["t_ns"] for b in run.beats))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for stage in STAGES:
        for epoch, points in series(run, stage, "busy_ns").items():
            pairs = list(zip(points, points[1:], strict=False))
            xs = [(t1 - origin) / 1e9 for (_, _), (t1, _) in pairs]
            ys = [(v1 - v0) / (t1 - t0) if t1 > t0 else 0 for (t0, v0), (t1, v1) in pairs]
            if xs:
                ax.step(xs, ys, where="pre", label=f"stage {stage} (epoch {epoch})")
    ax.set(title="Stage utilisation between beats", xlabel="s since run start", ylabel="busy")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "utilisation.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ command line


def find(paths: list[Path]) -> list[Path]:
    """The metrics files named, and those directly in each folder named, in name order."""
    found: set[Path] = set()
    for path in paths:
        if path.is_dir():
            found.update(path.glob(PATTERN))
        elif path.is_file():
            found.add(path)
        else:
            raise FileNotFoundError(path)
    return sorted(found, key=lambda p: (p.name, str(p.resolve())))


def check_out(out: Path, logs: list[Path]) -> None:
    """Refuse an output folder whose replacement could lose something: one that holds a
    log, a file, or a folder this script did not make. A folder it made is replaced whole,
    with anything added to it since."""
    if any(log.resolve().is_relative_to(out.resolve()) for log in logs):
        raise ValueError(f"{out} holds a log; choose another output folder")
    if out.exists() and not out.is_dir():
        raise ValueError(f"{out} is a file, not a folder")
    if out.is_dir() and not (out / MARKER).is_file() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty and was not made by this script")


def build(logs: list[Path], out: Path) -> list[dict]:
    """Write every table and plot for these logs into out, replacing what was there. The
    report is built beside out first, so a failure leaves the old one in place."""
    final, out = out, out.with_name(out.name + ".new")
    for folder in (final, out):
        check_out(folder, logs)
    if out.exists():
        shutil.rmtree(out)  # left by a build that failed
    out.mkdir(parents=True)
    (out / MARKER).write_text("made by pllm.report; replaced on every run\n", encoding="utf-8")
    summaries = []
    names: set[str] = set()
    for path in logs:
        run = load(path)
        name, n = run.name, 1
        while name in names:  # two logs of the same name in different folders
            n += 1
            name = f"{run.name}-{n}"
        names.add(name)
        s = summary(run)
        s["run"] = name
        write_run(run, s, out / name)
        summaries.append(s)
    rows = [runs_row(s) for s in summaries]
    _csv(out / "runs.csv", RUNS_HEADER, rows)
    text = ["# Runs", "", "Built by `uv run python -m pllm.report`; one folder per run.", ""]
    (out / "index.md").write_text(
        "\n".join(text + _markdown(RUNS_HEADER, rows)) + "\n", encoding="utf-8"
    )
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build every plot and table from run logs.")
    parser.add_argument("logs", nargs="*", type=Path, help=f"files or folders (default {RUNS})")
    parser.add_argument("--out", type=Path, default=OUT, help=f"output folder (default {OUT})")
    args = parser.parse_args(argv)
    try:
        logs = find(args.logs or [RUNS])
    except FileNotFoundError as err:
        parser.error(f"no such file or folder: {err}")
    if not logs:
        print(f"no {PATTERN} files found", file=sys.stderr)
        return 1
    try:
        summaries = build(logs, args.out)
    except ValueError as err:
        parser.error(str(err))
    print(f"{len(summaries)} runs written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
