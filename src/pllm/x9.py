"""X9: the layer split from even to balanced (R7.16), with the capacity-aware choice of k
(R8.6) and Host B's memory against its budget (R8.7).

Four steps, in order:

1. probe (each laptop, for its own stage): fresh processes (pllm.x9probe) load the stage
   with two layer counts, `--repeats` times each, in turn. For each count the step time is
   the median over the processes of their median decode step, and the memory is the
   largest peak RSS of the process tree. A straight line through the two counts gives the
   host's time and memory per layer and the part that does not depend on the layers.
2. choose (Host A): reads each host's pllm.hostmem record and probe record and calls
   pllm.capacity.choose_split. Host B's free memory is the smaller of its measured budget
   and --budget-b-mib, the stated one. Each layer also reserves the KV cache of
   --requests requests of --positions positions. It writes the chosen k and the split
   points to sweep from the even one down to it, or says why there is no split (exit 1).
3. run (Host A, once per split point): the deployment must have been started at that k
   with --split-point on both laptops. The runner reads k from the gateway's metrics file;
   the gateway accepts a stage only with the layer range of its own k, so once both stages
   beat they run the same k. Cells are X2 pipeline cells (closed loop, one untimed
   warm-up request first), repeat by repeat with the concurrencies inside. Throughput and
   stage busy time come from the gateway's metrics file, on its clock alone (pllm.report
   defines them). Host B's numbers come from pllm.x9monitor on Host B:
   - load peak: the worker's lifetime peak RSS read before any request. It counts only if
     the gateway has seen no request and Stage 0 no step; otherwise it is null.
   - decode: around each cell, the growth of the page fault and disk fault counters and
     the largest sampled RSS; at the end, the lifetime peak. If the lifetime peak grew
     over the load peak, decoding set it; if not, decoding stayed at or under the load
     peak, and the sampled maximum, which can miss a short peak, is a lower bound.
   A record is valid when the measurement is complete (see CHECKS); the exit code says
   so. Whether Host B kept to the budget is a result, reported apart: true, false or
   null for unknown. null never counts as true.
4. summary (Host A): puts the run records side by side and judges the chosen split. It
   passes only if the choice gave Host B fewer layers than the even split, every split of
   the sweep has valid records, and every record of the chosen split has Host B within
   the budget with no disk fault in its measured windows. A split may have several
   records, one per deployment; their cells are pooled.

What "no disk fault" shows is in pllm.procmem: the worker read no page from disk while
its measured cells ran. A count above 0 does not say whether the page file or a mapped
file was read, so it is reported as not shown, never excused. Faults during the load,
which reads the program and the weights from disk, and during the warm-ups are reported
beside it and not judged.

Usage:
uv run python -m pllm.x9 probe smollm2-135m --stage 0 --threads 6
uv run python -m pllm.x9 choose smollm2-135m --budget-b-mib 716.8 \
    --hostmem-b runs/<B>-host-memory.json --probe-b runs/<B>-x9-probe.json \
    --hostmem-a runs/<A>-host-memory.json --probe-a runs/<A>-x9-probe.json
uv run python -m pllm.x9 run smollm2-135m --metrics runs/<gateway>.metrics.jsonl \
    --budget-mib 716.8 --rtt-ms <netcheck round trip>
uv run python -m pllm.x9 summary --choice runs/<A>-x9-choice.json runs/*-x9.json
"""

import argparse
import dataclasses
import json
import math
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from pllm import (
    capacity,
    client,
    config,
    procmem,
    prompts,
    report,
    runrecord,
    x1b,
    x2,
    x9monitor,
)
from pllm.model import ModelConfig
from pllm.models import MODELS_DIR, PINNED, verify

REPEATS = 5
CONCURRENCY = (1, 4)
REQUESTS = 16
NEW_TOKENS = 32
LENGTH = 128
PROBE_PROMPT = 128
PROBE_STEPS = 64
PROBE_TIMEOUT_S = 600.0
MIB = 1024 * 1024

# What makes a run record a complete measurement. Those in TRIAL_EXEMPT may be false in a
# --trial record, which the summary never passes.
CHECKS = (
    "requests_ok",
    "stage_busy_reported",
    "worker_unchanged",
    "worker_matches",
    "load_separable",
    "counters_available",
    "decode_sampled",
    "same_commit",
    "two_hosts",
    "enough_repeats",
    "rtt_recorded",
)
TRIAL_EXEMPT = ("same_commit", "two_hosts", "enough_repeats", "rtt_recorded")
# Settings every run record of one summary must share.
SHARED = (
    "model",
    "revision",
    "commit",
    "network",
    "concurrency",
    "repeats",
    "requests",
    "new_tokens",
    "length",
    "budget_bytes",
    "host",
    "host_b_host",
    "host_b_threads",
)
CELL_METRICS = (
    "throughput_tok_s",
    "tpot_mean_ms",
    "ttft_mean_ms",
    *(f"stage{stage}_{key}" for stage in report.STAGES for key in ("busy_s", "busy_ms_per_step")),
    *(f"stage{stage}_utilisation" for stage in report.STAGES),
)


class BadInput(ValueError):
    """A record that cannot be used."""


def load_record(path: Path, script: str) -> dict:
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise BadInput(f"{path}: not a readable record ({err})") from None
    if not isinstance(rec, dict) or rec.get("script") != script or rec.get("status") != "ok":
        raise BadInput(f"{path}: not a {script} record with status ok")
    return rec


# ------------------------------------------------------------------ probe


def probe_counts(num_layers: int, stage: int) -> tuple[int, int]:
    """The two layer counts probed by default. They span what the stage holds from the
    even split to well past it: Host B shrinks from half, Host A grows from half."""
    sixth, half = max(1, num_layers // 6), num_layers // 2
    return (sixth, half) if stage == 0 else (num_layers - half, num_layers - sixth)


def layer_range(num_layers: int, stage: int, count: int) -> tuple[int, int]:
    return (0, count) if stage == 0 else (num_layers - count, num_layers)


def run_probe_process(
    command: list[str], snapshot: Callable[[int], dict] = procmem.tree_snapshot
) -> dict:
    """Run one probe process and add the peak RSS of its tree, read while it is alive."""
    process = subprocess.Popen(
        command, cwd=config.REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        text = process.stdout.readline()
        if not text:
            raise RuntimeError(f"the probe process gave no result, exit code {process.wait()}")
        result = json.loads(text)
        result["tree_peak_rss_bytes"] = snapshot(process.pid)["peak_rss_bytes"]
        process.communicate("\n", timeout=PROBE_TIMEOUT_S)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"the probe process ended with exit code {process.returncode}")
    return result


def probe_row(result: dict) -> dict:
    """One probe process's numbers. The tree's peak is used where the system reports it
    (as for the worker X9 judges); on macOS, with no launcher process, the process's own."""
    tree = result["tree_peak_rss_bytes"]
    return {
        "layers": result["layer_end"] - result["layer_start"],
        "step_seconds_median": statistics.median(result["step_seconds"]),
        "steps": len(result["step_seconds"]),
        "load_seconds": result["load_seconds"],
        "load_peak_rss_bytes": result["own_load_peak_rss_bytes"],
        "peak_rss_bytes": tree if tree is not None else result["own_peak_rss_bytes"],
        "peak_rss_source": "process tree" if tree is not None else "the process itself",
    }


def fit_host(rows: list[dict]) -> dict:
    """The straight lines of one host through its two layer counts: the step time through
    the median over each count's processes, the memory through the largest peak of each.
    Raises ValueError when the measurements cannot give them."""
    counts = sorted({row["layers"] for row in rows})
    if len(counts) != 2:
        raise ValueError(f"the probe needs two layer counts, got {counts}")
    groups = [[row for row in rows if row["layers"] == count] for count in counts]
    seconds = [statistics.median(r["step_seconds_median"] for r in group) for group in groups]
    peaks = [max(r["peak_rss_bytes"] for r in group) for group in groups]

    def line(name: str, low: float, high: float) -> tuple[float, float]:
        slope = (high - low) / (counts[1] - counts[0])
        if slope <= 0:
            raise ValueError(
                f"the {name} did not grow from {counts[0]} to {counts[1]} layers "
                f"({low} to {high}): close other programs and probe again"
            )
        return low - slope * counts[0], slope

    fixed_seconds, seconds_per_layer = line("median step time", *seconds)
    fixed_bytes, bytes_per_layer = line("peak memory", *peaks)
    if fixed_bytes < 0:
        raise ValueError(f"the fitted memory with no layers is {fixed_bytes} bytes, below 0")
    return {
        "layer_counts": counts,
        "per_count": [
            {
                "layers": count,
                "step_seconds": x2.spread([r["step_seconds_median"] for r in group]),
                "peak_rss_bytes": x2.spread([r["peak_rss_bytes"] for r in group]),
            }
            for count, group in zip(counts, groups, strict=True)
        ],
        "fixed_seconds": fixed_seconds,
        "seconds_per_layer": seconds_per_layer,
        "fixed_bytes": fixed_bytes,
        "bytes_per_layer": bytes_per_layer,
    }


def probe(args) -> int:
    spec = PINNED[args.model]
    model_dir = MODELS_DIR / args.model
    for filename, expected in spec.files.items():
        verify(model_dir / filename, expected)
    n = ModelConfig.from_file(model_dir / "config.json").num_layers
    counts = args.layer_counts or probe_counts(n, args.stage)
    if len(set(counts)) != 2 or not all(1 <= c <= n - 1 for c in counts):
        raise BadInput(f"--layer-counts needs two different counts from 1 to {n - 1}")
    with runrecord.record(
        "x9-probe", model=args.model, revision=spec.revision, threads=args.threads
    ) as rec:
        rec.update(
            stage=args.stage,
            num_layers=n,
            repeats=args.repeats,
            prompt=args.prompt,
            steps=args.steps,
            processes=[],
        )
        for repeat in range(args.repeats):
            for count in sorted(counts, reverse=bool(repeat % 2)):
                start, end = layer_range(n, args.stage, count)
                command = [sys.executable, "-m", "pllm.x9probe", str(model_dir)]
                command += ["--layers", f"{start}:{end}", "--threads", str(args.threads)]
                command += ["--prompt", str(args.prompt), "--steps", str(args.steps)]
                row = {"repeat": repeat, **probe_row(run_probe_process(command))}
                rec["processes"].append(row)
                print(json.dumps(row), flush=True)
        rec["fit"] = fit_host(rec["processes"])
        print(json.dumps(rec["fit"], indent=2), flush=True)
    return 0


# ------------------------------------------------------------------ choose


def kv_bytes_per_layer(model: ModelConfig, positions: int) -> int:
    """One request's keys and values in one layer (pllm.kvstore)."""
    return positions * 2 * model.num_kv_heads * model.head_dim * 4


def host_capacity(fit: dict, free_bytes: float, kv_layer: int, requests: int):
    """A HostCapacity from a probe's fit. Each layer reserves the cache of `requests`
    requests; one layer of one request is added once, for the copy an append makes."""
    return capacity.HostCapacity(
        free_bytes=free_bytes,
        fixed_bytes=fit["fixed_bytes"] + kv_layer,
        bytes_per_layer=fit["bytes_per_layer"] + requests * kv_layer,
        fixed_seconds=fit["fixed_seconds"],
        seconds_per_layer=fit["seconds_per_layer"],
    )


def check_choice_inputs(records: dict, model: str, trial: bool = False) -> None:
    """records has hostmem_a, hostmem_b, probe_a and probe_b. Anything missing, from the
    wrong host or for another model is refused."""
    revision = PINNED[model].revision
    for side, stage in (("b", 0), ("a", 1)):
        memory, probed = records[f"hostmem_{side}"], records[f"probe_{side}"]
        value = memory.get("budget_bytes")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise BadInput(f"hostmem_{side} has no measured budget_bytes")
        if (probed.get("model"), probed.get("revision")) != (model, revision):
            raise BadInput(f"probe_{side} is not of {model} at the pinned revision")
        if probed.get("stage") != stage:
            raise BadInput(f"probe_{side} must be of stage {stage}")
        if not isinstance(probed.get("fit"), dict):
            raise BadInput(f"probe_{side} has no fit")
        if not memory.get("host") or memory.get("host") != probed.get("host"):
            raise BadInput(f"hostmem_{side} and probe_{side} are not from the same host")
    if trial:
        return
    if records["probe_a"]["host"] == records["probe_b"]["host"]:
        raise BadInput("both hosts' records are from one host (use --trial to allow it)")
    commits = {records["probe_a"].get("commit"), records["probe_b"].get("commit")}
    if len(commits) != 1 or not _clean_commit(next(iter(commits))):
        raise BadInput(f"the probes need one clean commit, got {sorted(map(str, commits))}")


def _clean_commit(commit) -> bool:
    return isinstance(commit, str) and bool(commit) and not commit.endswith("-dirty")


def decide(
    records: dict,
    model: ModelConfig,
    budget_b_bytes: int,
    positions: int,
    requests: int,
    points: int = 4,
) -> dict:
    """The inputs, capacities and result of the choice. result is None, with the reason in
    no_split, when no split is allowed."""
    kv_layer = kv_bytes_per_layer(model, positions)
    free = {
        "b": min(records["hostmem_b"]["budget_bytes"], budget_b_bytes),
        "a": records["hostmem_a"]["budget_bytes"],
    }
    hosts = {
        side: host_capacity(records[f"probe_{side}"]["fit"], free[side], kv_layer, requests)
        for side in ("a", "b")
    }
    out = {
        "num_layers": model.num_layers,
        "budget_b_bytes": budget_b_bytes,
        "measured_budget_b_bytes": records["hostmem_b"]["budget_bytes"],
        "measured_budget_a_bytes": records["hostmem_a"]["budget_bytes"],
        "positions": positions,
        "requests": requests,
        "kv_bytes_per_layer_per_request": kv_layer,
        "host_b": records["probe_b"]["host"],
        "host_a": records["probe_a"]["host"],
        "threads_b": records["probe_b"].get("threads"),
        "threads_a": records["probe_a"].get("threads"),
        "capacity_b": dataclasses.asdict(hosts["b"]),
        "capacity_a": dataclasses.asdict(hosts["a"]),
        "result": None,
        "no_split": None,
        "sweep": None,
        "extrapolated": None,
    }
    try:
        split = capacity.choose_split(model.num_layers, hosts["b"], hosts["a"])
    except capacity.NoSplit as err:
        out["no_split"] = str(err)
        return out
    out["result"] = dataclasses.asdict(split)
    out["sweep"] = capacity.sweep_points(split.even_k, split.k, points)
    held = {"b": split.k, "a": model.num_layers - split.k}
    # a count outside the probed range rests on extending the line, not on a measurement
    out["extrapolated"] = {
        side: not min(counts) <= held[side] <= max(counts)
        for side in ("a", "b")
        for counts in [records[f"probe_{side}"]["fit"]["layer_counts"]]
    }
    return out


def choose(args) -> int:
    paths = {key: getattr(args, key) for key in ("hostmem_a", "hostmem_b", "probe_a", "probe_b")}
    records = {
        key: load_record(path, "host-memory" if key.startswith("hostmem") else "x9-probe")
        for key, path in paths.items()
    }
    check_choice_inputs(records, args.model, args.trial)
    model = ModelConfig.from_file(MODELS_DIR / args.model / "config.json")
    with runrecord.record(
        "x9-choice", model=args.model, revision=PINNED[args.model].revision
    ) as rec:
        rec["trial"] = args.trial
        rec["input_files"] = {key: path.name for key, path in paths.items()}
        rec.update(
            decide(
                records,
                model,
                round(args.budget_b_mib * MIB),
                args.positions,
                args.requests,
                args.points,
            )
        )
    print(json.dumps({key: rec[key] for key in ("result", "no_split", "sweep", "extrapolated")}))
    if rec["result"] is None:
        print(f"no split: {rec['no_split']}")
        return 1
    result = rec["result"]
    print(f"chosen k = {result['k']} (even split {result['even_k']}); sweep {rec['sweep']}")
    if result["forced_below_even"]:
        print(
            f"warning: speed and memory alone would choose k = {result['unconstrained_k']}; "
            "k is below the even split only because the rule requires it"
        )
    if any(rec["extrapolated"].values()):
        print("warning: the chosen split is outside a probed range; probe nearer to it")
    return 0


# ------------------------------------------------------------------ Host B memory


def monitor_get(url: str, path: str, timeout: float = 30.0) -> dict:
    """One answer of pllm.x9monitor. No proxy is used: the monitor is on the LAN."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url.rstrip("/") + path, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        reason = err.read().decode(errors="replace")
        raise RuntimeError(f"monitor {path}: {err.code} {reason}") from None
    except (OSError, ValueError) as err:
        raise RuntimeError(f"monitor at {url} cannot be read ({err}): is it running?") from None


def same_worker(first: dict, other: dict) -> bool:
    """Whether two snapshots read the same worker processes, with no restart between."""

    def key(s: dict):
        pids = sorted(row["pid"] for row in s.get("processes") or [])
        return s.get("host"), s.get("pid"), s.get("restarts"), pids

    return bool(first.get("processes")) and key(first) == key(other)


def window_fields(loaded: dict, start: dict, stop: dict) -> dict:
    """Host B's memory over one cell, from the snapshots at its two ends."""
    same = same_worker(loaded, start) and same_worker(loaded, stop)
    sampled = stop.get("window") or {}
    return {
        "same_worker": same,
        "disk_faults": procmem.delta(start, stop, "disk_faults") if same else None,
        "page_faults": procmem.delta(start, stop, "page_faults") if same else None,
        "sampled_rss_max_bytes": sampled.get("rss_max_bytes"),
        "samples": sampled.get("samples"),
        "max_sample_gap_s": sampled.get("max_gap_s"),
        "sample_read_errors": sampled.get("read_errors"),
        "peak_rss_after_bytes": stop.get("peak_rss_bytes"),
    }


def _all_or_none(values: list, combine):
    """combine(values), or None if there are none or any is unknown."""
    return None if not values or any(v is None for v in values) else combine(values)


def within(values: list, limit: float) -> bool | None:
    """Whether every value is at most limit: False as soon as a known one is over it,
    None if none is over but one is unknown."""
    if any(v is not None and v > limit for v in values):
        return False
    return None if not values or any(v is None for v in values) else True


def host_b_result(
    loaded: dict, windows: list[dict], final: dict, load_clean: bool, budget_bytes: int
) -> dict:
    """Host B's memory over a run and the two verdicts on it."""
    unchanged = same_worker(loaded, final) and all(w["same_worker"] for w in windows)
    load_peak = loaded.get("peak_rss_bytes") if load_clean else None
    lifetime = final.get("peak_rss_bytes") if unchanged else None
    raised = None if load_peak is None or lifetime is None else lifetime > load_peak
    in_windows = _all_or_none([w["disk_faults"] for w in windows], sum)
    since_load = procmem.delta(loaded, final, "disk_faults") if unchanged else None
    # a peak over the budget in any reading is over it, whatever else is unknown
    seen = [loaded.get("peak_rss_bytes"), final.get("peak_rss_bytes")]
    seen += [w["peak_rss_after_bytes"] for w in windows]
    budget_ok = within(seen, budget_bytes)
    if budget_ok is True and (not unchanged or load_peak is None):
        budget_ok = None
    return {
        "budget_bytes": budget_bytes,
        "load_peak_rss_bytes": load_peak,
        "lifetime_peak_rss_bytes": lifetime,
        "decode_raised_peak": raised,
        # exact only when decoding set the lifetime peak
        "decode_peak_rss_bytes": lifetime if raised else None,
        "decode_sampled_rss_max_bytes": _all_or_none(
            [w["sampled_rss_max_bytes"] for w in windows], max
        ),
        "load_disk_faults": loaded.get("disk_faults") if load_clean else None,
        "load_page_faults": loaded.get("page_faults") if load_clean else None,
        "window_disk_faults": in_windows,
        "window_page_faults": _all_or_none([w["page_faults"] for w in windows], sum),
        "outside_window_disk_faults": None
        if since_load is None or in_windows is None
        else since_load - in_windows,
        "disk_fault_counter": loaded.get("disk_fault_counter"),
        "swap_used_loaded_bytes": loaded.get("swap_used_bytes"),
        "swap_used_final_bytes": final.get("swap_used_bytes"),
        "available_memory_loaded_bytes": loaded.get("available_memory_bytes"),
        "available_memory_final_bytes": final.get("available_memory_bytes"),
        "monitor_rss_bytes": final.get("monitor_rss_bytes"),
        "within_budget": budget_ok,
        "no_disk_faults_in_windows": None
        if in_windows is None or not unchanged
        else in_windows == 0,
    }


# ------------------------------------------------------------------ run


def load_is_clean(run: report.Run) -> bool:
    """Whether the gateway has seen no request and Stage 0's newest beat counts no step,
    so the worker's peak so far is that of its load. False with no beat."""
    beats = [b for b in run.beats if b.get("stage") == 0]
    if not beats or any(r.get("rid") for r in run.requests):
        return False
    return max(beats, key=lambda b: b["t_ns"])["steps"] == 0


def preflight(snapshot: dict, run: report.Run, commit: str | None, model: str, k: int):
    """Why a run starting now could not give a valid record, found before it spends the
    time: an empty list if nothing already rules it out."""
    problems = []
    if not load_is_clean(run):
        problems.append(
            "the pipeline has already served a request (or Stage 0 sent no beat), so the "
            "load peak cannot be told apart: restart both laptops"
        )
    if not worker_matches(snapshot, model, k):
        problems.append(
            f"the monitor is not reading a stage 0 worker of {model} at k = {k}: "
            f"{snapshot.get('worker_args')}"
        )
    for key in ("peak_rss_bytes", "disk_faults"):
        if snapshot.get(key) is None:
            problems.append(f"the monitor cannot read the worker's {key}")
    commits = {commit, run.info.get("commit"), snapshot.get("commit")}
    if len(commits) != 1 or not _clean_commit(next(iter(commits))):
        problems.append(
            "the gateway, this checkout and Host B need one clean commit, got "
            + ", ".join(sorted(map(str, commits)))
        )
    if snapshot.get("host") == socket.gethostname():
        problems.append("the monitor is on this laptop, not on Host B")
    return problems


def one_epoch_each(run: report.Run) -> bool:
    """Whether each stage kept one epoch over the whole metrics file: none restarted."""
    return all(len(report.series(run, stage, "steps")) == 1 for stage in report.STAGES)


def option(args: list[str] | None, name: str) -> str | None:
    """The value after name in a command line, or None."""
    if not args or name not in args[:-1]:
        return None
    return args[args.index(name) + 1]


def worker_matches(snapshot: dict, model: str, k: int) -> bool:
    """Whether the monitored process is a serving stage 0 worker of this model whose
    --split-point, if it has one, is k."""
    args = snapshot.get("worker_args")
    if not args or args[0] != model or option(args, "--stage") != "0" or "--serve" not in args:
        return False
    return option(args, "--split-point") in (None, str(k))


def stage_fields(summary: dict) -> dict:
    """Busy time per stage in a cell's window, in all and per step."""
    out = {}
    for stage in report.STAGES:
        row = summary["stages"].get(stage, {})
        seconds, steps = row.get("busy_s"), row.get("steps")
        out[f"stage{stage}_busy_s"] = seconds
        out[f"stage{stage}_steps"] = steps
        out[f"stage{stage}_busy_ms_per_step"] = (
            None if seconds is None or not steps else seconds * 1e3 / steps
        )
    return out


def aggregate(cells: list[dict]) -> list[dict]:
    """One line per split and concurrency with the spread of each metric over the cells."""
    groups: dict[tuple[int, int], list[dict]] = {}
    for line in cells:
        groups.setdefault((line["split"], line["concurrency"]), []).append(line)
    return [
        {
            "type": "aggregate",
            "split": k,
            "concurrency": c,
            "repeats": len(lines),
            **{name: x2.spread([line.get(name) for line in lines]) for name in CELL_METRICS},
            "failed_requests": sum(line["failed_requests"] for line in lines),
        }
        for (k, c), lines in sorted(groups.items(), key=lambda item: (-item[0][0], item[0][1]))
    ]


def sweep(call, metrics_path: Path, monitor, args, k: int, prompt_ids, emit) -> tuple:
    """Every cell at the deployment's split point, between snapshots of Host B. monitor
    takes a path of pllm.x9monitor. Returns the cells, the load snapshot, whether it was
    clean and the final snapshot."""
    loaded = monitor("/snapshot")
    clean = load_is_clean(report.load(metrics_path))
    cells = []
    for repeat in range(args.repeats):
        for c in args.concurrency:
            x2.serve_pipeline(call, x2.jobs(prompt_ids, 1), 1)  # warm-up, outside the window
            cell = x2.jobs(prompt_ids, args.requests)
            start = monitor("/start")
            x2.serve_pipeline(call, cell, c)
            stop = monitor("/stop")
            time.sleep(args.settle)
            rids = {job.rid for job in cell if job.rid}
            run, missing = x2.cell_run(metrics_path, rids)
            line = x2.cell_line("pipeline", c, repeat, cell, run, missing)
            line.update(split=k, **stage_fields(report.summary(run)))
            line["host_b"] = window_fields(loaded, start, stop)
            cells.append(line)
            emit(line)
    return cells, loaded, clean, monitor("/snapshot")


def run_checks(cells, loaded, clean, final, host_b, run: report.Run, rec: dict, k: int) -> dict:
    """The CHECKS of one run record, each true or false."""
    commits = {rec.get("commit"), run.info.get("commit"), loaded.get("commit")}
    return {
        "requests_ok": bool(cells) and x2.summary_line(cells)["passed"],
        "stage_busy_reported": bool(cells)
        and all(
            line["throughput_tok_s"] is not None
            and all(line[f"stage{s}_busy_ms_per_step"] is not None for s in report.STAGES)
            for line in cells
        ),
        "worker_unchanged": one_epoch_each(run)
        and same_worker(loaded, final)
        and all(line["host_b"]["same_worker"] for line in cells),
        "worker_matches": worker_matches(loaded, rec["model"], k),
        "load_separable": clean,
        "counters_available": all(
            host_b[key] is not None
            for key in ("load_peak_rss_bytes", "lifetime_peak_rss_bytes", "window_disk_faults")
        ),
        "decode_sampled": host_b["decode_sampled_rss_max_bytes"] is not None,
        "same_commit": len(commits) == 1 and _clean_commit(next(iter(commits))),
        "two_hosts": bool(loaded.get("host")) and loaded.get("host") != rec.get("host"),
        "enough_repeats": rec["repeats"] >= REPEATS,
        "rtt_recorded": rec.get("rtt_ms") is not None,
    }


def is_valid(checks: dict, trial: bool) -> bool:
    return all(checks.get(name) is True for name in CHECKS if not (trial and name in TRIAL_EXEMPT))


def run_split(args) -> int:
    info = report.load(args.metrics).info
    if info.get("model") != args.model:
        raise BadInput(f"the gateway serves {info.get('model')}, not {args.model}")
    k = info.get("split")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise BadInput("the gateway's metrics file names no split point")
    hosts = config.load()
    url = args.monitor or f"http://{hosts.stage0.host}:{x9monitor.PORT}"
    prompt_ids = prompts.prompts(args.model, args.length)
    if not x1b.wait_until_ready(args.metrics, args.wait):
        raise BadInput(
            f"both stages did not beat within {args.wait} s: is each laptop started at k = {k}?"
        )
    if not args.trial:
        problems = preflight(
            monitor_get(url, "/snapshot"),
            report.load(args.metrics),
            runrecord.commit_hash(),
            args.model,
            k,
        )
        if problems:
            raise BadInput("; ".join(problems))
    tokenizer = prompts.load_tokenizer(args.model)

    def call(job):
        text = tokenizer.decode(job.ids, skip_special_tokens=False)
        return client.generate(hosts.gateway.address, text, args.new_tokens, args.timeout)

    with runrecord.record(
        "x9", model=args.model, revision=PINNED[args.model].revision, split=k, rtt_ms=args.rtt_ms
    ) as rec:
        rec.update(
            trial=args.trial,
            concurrency=args.concurrency,
            repeats=args.repeats,
            requests=args.requests,
            new_tokens=args.new_tokens,
            length=args.length,
            settle_s=args.settle,
            timeout_s=args.timeout,
            metrics_file=args.metrics.name,
            budget_bytes=round(args.budget_mib * MIB),
        )
        lines_path = runrecord.output_path(rec, ".jsonl")
        rec["results_file"] = lines_path.name
        with lines_path.open("w", encoding="utf-8") as out:

            def emit(line: dict) -> None:
                text = json.dumps(line, allow_nan=False)
                out.write(text + "\n")
                print(text, flush=True)

            cells, loaded, clean, final = sweep(
                call,
                args.metrics,
                lambda path: monitor_get(url, path),
                args,
                k,
                prompt_ids,
                emit,
            )
            time.sleep(args.settle)
            whole = report.load(args.metrics)
            host_b = host_b_result(
                loaded, [c["host_b"] for c in cells], final, clean, rec["budget_bytes"]
            )
            checks = run_checks(cells, loaded, clean, final, host_b, whole, rec, k)
            rows = aggregate(cells)
            for row in rows:
                emit(row)
            threads = option(loaded.get("worker_args"), "--threads")
            rec.update(
                gateway_commit=whole.info.get("commit"),
                host_b_host=loaded.get("host"),
                host_b_commit=loaded.get("commit"),
                host_b_worker_args=loaded.get("worker_args"),
                host_b_threads=None if threads is None else int(threads),
                host_b=host_b,
                host_b_snapshots={"loaded": loaded, "final": final},
                cells=cells,
                aggregates=rows,
                checks=checks,
                valid=is_valid(checks, args.trial),
            )
            emit({"type": "summary", "split": k, "host_b": host_b, "checks": checks,
                  "valid": rec["valid"]})  # fmt: skip
    return 0 if rec["valid"] else 1


# ------------------------------------------------------------------ summary


def check_comparable(records: list[dict]) -> None:
    """Every run record has the same settings, and none is given twice."""
    for rec in records:
        missing = [key for key in (*SHARED, "split", "started") if rec.get(key) is None]
        if missing:
            raise BadInput(f"a run record lacks {', '.join(missing)}")
    for key in SHARED:
        values = {json.dumps(rec[key]) for rec in records}
        if len(values) > 1:
            raise BadInput(f"the run records differ in {key}: {', '.join(sorted(values))}")
    if len({rec["started"] for rec in records}) != len(records):
        raise BadInput("a run record is given twice")


def check_choice(choice: dict, records: list[dict]) -> None:
    first = records[0]
    if (choice.get("model"), choice.get("revision")) != (first["model"], first["revision"]):
        raise BadInput("the choice is for another model")
    if not choice.get("trial"):
        for side, key in (("b", "host_b_host"), ("a", "host")):
            if choice.get(f"host_{side}") != first[key]:
                raise BadInput(f"the choice measured another Host {side.upper()} than the runs'")
        if choice.get("threads_b") != first["host_b_threads"]:
            raise BadInput(
                f"Host B was probed with {choice.get('threads_b')} threads, but ran with "
                f"{first['host_b_threads']}"
            )
    if choice.get("budget_b_bytes") != first["budget_bytes"]:
        raise BadInput("the choice and the runs state different Host B budgets")


def verdict(choice: dict, records: list[dict]) -> dict:
    """The issue's three criteria. Each is true, false or null (unknown)."""
    result = choice.get("result")
    if result is None:
        return {
            "chosen_k": None,
            "criterion_1_fewer_layers_than_even": False,
            "criterion_2_sweep_reported": False,
            "criterion_3_chosen_within_budget_no_disk_faults": None,
            "missing_splits": None,
            "trial": any(rec.get("trial") for rec in records),
            "passed": False,
        }
    k, even = result["k"], result["even_k"]
    valid = {rec["split"] for rec in records}
    valid -= {rec["split"] for rec in records if rec.get("valid") is not True}
    wanted = sorted(set(choice.get("sweep") or []) | {even, k}, reverse=True)
    missing = [point for point in wanted if point not in valid]
    chosen = [rec for rec in records if rec["split"] == k]
    answers = [
        rec["host_b"].get(key)
        for rec in chosen
        for key in ("within_budget", "no_disk_faults_in_windows")
    ]
    if False in answers:
        third = False
    elif chosen and k in valid and all(answer is True for answer in answers):
        third = True
    else:
        third = None
    out = {
        "chosen_k": k,
        "even_k": even,
        "forced_below_even": result["forced_below_even"],
        "extrapolated": choice.get("extrapolated"),
        "criterion_1_fewer_layers_than_even": k < even,
        "criterion_2_sweep_reported": not missing,
        "criterion_3_chosen_within_budget_no_disk_faults": third,
        "missing_splits": missing,
        "trial": bool(choice.get("trial")) or any(rec.get("trial") for rec in records),
    }
    out["passed"] = not out["trial"] and all(
        out[key] is True for key in out if key.startswith("criterion_")
    )
    return out


def _mean_spread(values: list, digits: int = 2) -> str:
    s = x2.spread(values)
    if not s["n"]:
        return "unknown"
    stdev = "" if s["stdev"] is None else f" sd {s['stdev']:.{digits}f}"
    ends = f"min {s['min']:.{digits}f}, max {s['max']:.{digits}f}"
    return f"{s['mean']:.{digits}f}{stdev} ({ends}, n {s['n']})"


def _each(values: list, scale: float = 1, digits: int = 0) -> str:
    """One value per deployment."""
    return ", ".join("unknown" if v is None else f"{v / scale:.{digits}f}" for v in values)


def _yes(value: bool | None) -> str:
    return {True: "yes", False: "no", None: "unknown"}[value]


def tables(records: list[dict], found: dict) -> str:
    """The summary as markdown: speed per split, Host B memory per deployment, verdict."""
    first = records[0]
    splits = sorted({rec["split"] for rec in records}, reverse=True)
    text = [
        "# X9: layer split sweep",
        "",
        f"Model {first['model']}, commit {first['commit']}, network {first['network']}, "
        f"{first['requests']} requests of {first['length']} prompt and {first['new_tokens']} "
        f"new tokens per cell. Host B budget {first['budget_bytes'] / MIB:.1f} MiB.",
        "",
        "| k | concurrency | tokens/s | stage 0 busy s | stage 1 busy s | "
        "stage 0 ms/step | stage 1 ms/step |",
        "|" + "---|" * 7,
    ]
    for k in splits:
        for c in first["concurrency"]:
            cells = [
                line
                for rec in records
                if rec["split"] == k
                for line in rec["cells"]
                if line["concurrency"] == c
            ]
            names = ["throughput_tok_s", "stage0_busy_s", "stage1_busy_s"]
            names += ["stage0_busy_ms_per_step", "stage1_busy_ms_per_step"]
            row = [_mean_spread([line.get(name) for line in cells]) for name in names]
            text.append(f"| {k} | {c} | " + " | ".join(row) + " |")
    text += [
        "",
        "Host B, one value per deployment. Peaks are in MiB over the worker's process tree. "
        "Disk faults are the counter below, not all page faults.",
        "",
        "| k | load peak | lifetime peak | decode raised the peak | decode sampled max | "
        "load disk faults | disk faults in cells | disk faults between cells | "
        "all page faults in cells | within budget | no disk faults in cells | record valid |",
        "|" + "---|" * 12,
    ]
    for k in splits:
        mine = [rec for rec in records if rec["split"] == k]
        host = [rec["host_b"] for rec in mine]
        row = [
            _each([h.get("load_peak_rss_bytes") for h in host], MIB, 1),
            _each([h.get("lifetime_peak_rss_bytes") for h in host], MIB, 1),
            ", ".join(_yes(h.get("decode_raised_peak")) for h in host),
            _each([h.get("decode_sampled_rss_max_bytes") for h in host], MIB, 1),
            _each([h.get("load_disk_faults") for h in host]),
            _each([h.get("window_disk_faults") for h in host]),
            _each([h.get("outside_window_disk_faults") for h in host]),
            _each([h.get("window_page_faults") for h in host]),
            ", ".join(_yes(h.get("within_budget")) for h in host),
            ", ".join(_yes(h.get("no_disk_faults_in_windows")) for h in host),
            ", ".join(_yes(rec.get("valid")) for rec in mine),
        ]
        text.append(f"| {k} | " + " | ".join(row) + " |")
    counters = sorted({str(rec["host_b"].get("disk_fault_counter")) for rec in records})
    text += ["", f"Disk fault counter: {'; '.join(counters)}.", ""]
    text += [
        "0 disk faults in cells means this counter saw the worker read no page from disk "
        "while the cells ran. It does not show pages written out and not read back, or "
        "paging by other processes, and a count above 0 does not say whether the page "
        "file or a mapped file was read.",
        "",
        "## Verdict",
        "",
    ]
    text += [f"- {key}: {json.dumps(value)}" for key, value in found.items()]
    return "\n".join(text) + "\n"


def summarise(args) -> int:
    choice = load_record(args.choice, "x9-choice")
    records = [load_record(path, "x9") for path in args.records]
    check_comparable(records)
    check_choice(choice, records)
    found = verdict(choice, records)
    text = tables(records, found)
    with runrecord.record(
        "x9-summary", model=records[0]["model"], revision=records[0]["revision"]
    ) as rec:
        rec["input_files"] = [path.name for path in (args.choice, *args.records)]
        rec["verdict"] = found
        path = runrecord.output_path(rec, ".md")
        path.write_text(text, encoding="utf-8")
        rec["results_file"] = path.name
    print(text)
    return 0 if found["passed"] else 1


# ------------------------------------------------------------------ command line


def parser_() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="X9: the capacity-aware layer split.")
    commands = parser.add_subparsers(dest="command", required=True)

    one = commands.add_parser("probe", help="measure this laptop's stage per layer")
    one.add_argument("model", choices=sorted(PINNED))
    one.add_argument("--stage", type=int, choices=(0, 1), required=True)
    one.add_argument("--threads", type=int, required=True, help="as the deployment uses")
    one.add_argument("--layer-counts", type=int, nargs=2, help="two layer counts to load")
    one.add_argument("--repeats", type=int, default=REPEATS, help="processes per count")
    one.add_argument("--prompt", type=int, default=PROBE_PROMPT)
    one.add_argument("--steps", type=int, default=PROBE_STEPS)

    two = commands.add_parser("choose", help="choose k from both hosts' measurements")
    two.add_argument("model", choices=sorted(PINNED))
    for name in ("hostmem-a", "hostmem-b", "probe-a", "probe-b"):
        two.add_argument(f"--{name}", type=Path, required=True)
    two.add_argument("--budget-b-mib", type=float, required=True, help="Host B's stated budget")
    two.add_argument("--positions", type=int, default=LENGTH + NEW_TOKENS)
    two.add_argument("--requests", type=int, default=max(CONCURRENCY))
    two.add_argument("--points", type=int, default=4, help="split points in the sweep")
    two.add_argument("--trial", action="store_true", help="allow one host or a dirty commit")

    three = commands.add_parser("run", help="measure the split the deployment is running")
    three.add_argument("model", choices=sorted(PINNED))
    three.add_argument("--metrics", type=Path, required=True, help="the gateway's metrics file")
    three.add_argument("--monitor", help="Host B's pllm.x9monitor (default: stage 0's host)")
    three.add_argument("--budget-mib", type=float, required=True, help="Host B's stated budget")
    three.add_argument("--rtt-ms", type=float, help="round trip from pllm.netcheck")
    three.add_argument("--concurrency", type=int, nargs="+", default=list(CONCURRENCY))
    three.add_argument("--repeats", type=int, default=REPEATS)
    three.add_argument("--requests", type=int, default=REQUESTS, help="per cell")
    three.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    three.add_argument("--length", type=int, choices=prompts.LENGTHS, default=LENGTH)
    three.add_argument("--settle", type=float, default=x2.SETTLE_S, help="seconds")
    three.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    three.add_argument("--wait", type=float, default=x1b.WAIT_S, help="for both stages")
    three.add_argument(
        "--trial",
        action="store_true",
        help="a rehearsal: allows one host, a dirty commit, few repeats and no --rtt-ms",
    )

    four = commands.add_parser("summary", help="compare the splits and judge the chosen one")
    four.add_argument("records", type=Path, nargs="+", help="x9 run records")
    four.add_argument("--choice", type=Path, required=True, help="the x9-choice record")
    return parser


def check_args(parser: argparse.ArgumentParser, args) -> None:
    def positive(*names: str) -> None:
        for name in names:
            value = getattr(args, name.replace("-", "_"))
            if not 0 < value < math.inf:
                parser.error(f"--{name} must be positive")

    if args.command == "probe":
        positive("threads", "repeats", "prompt", "steps")
    elif args.command == "choose":
        positive("budget-b-mib", "positions", "requests")
        if args.points < 2:
            parser.error("--points must be at least 2")
    elif args.command == "run":
        positive("budget-mib", "repeats", "requests", "new-tokens", "timeout")
        if min(args.concurrency) < 1 or len(set(args.concurrency)) != len(args.concurrency):
            parser.error("--concurrency needs different positive values")
        if args.requests < 2 * max(args.concurrency):
            parser.error("--requests must be at least twice the largest concurrency")
        if args.rtt_ms is None and not args.trial:
            parser.error("--rtt-ms is required (pllm.netcheck measures it), or use --trial")
        if args.rtt_ms is not None and not 0 <= args.rtt_ms < math.inf:
            parser.error("--rtt-ms must be a number of at least 0")
        if not args.metrics.is_file():
            parser.error("--metrics must be the running gateway's metrics file")


def main(argv: list[str] | None = None) -> int:
    parser = parser_()
    args = parser.parse_args(argv)
    check_args(parser, args)
    actions = {"probe": probe, "choose": choose, "run": run_split, "summary": summarise}
    try:
        return actions[args.command](args)
    except BadInput as err:
        parser.error(str(err))


if __name__ == "__main__":
    sys.exit(main())
