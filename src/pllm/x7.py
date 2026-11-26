"""X7: overload with the admission bound on and off and the credit limit varied (R7.14).

The runner is a client of a gateway already started with the setting under test (the
gateway's --admission-queue and --credits); the setting is read from the gateway's metrics
file, never from the command line. It offers requests on a schedule that does not wait for
answers (open loop, Poisson arrivals from a seed), so a gateway that falls behind builds a
queue instead of slowing the load. Every setting gets the same schedule: the same seed,
rate, count, prompts and token limit.

- capacity: the completed requests per second of a closed loop with every running slot
  busy. Choose the rate once, above this, and use it for every setting.
- run: offer the load, then read what the gateway wrote during it. Rejections, latencies
  and queue depths come from the metrics file, on the gateway's clock. Percentiles are
  those of pllm.report, over completed requests; refused and failed requests are counted
  beside them, since leaving them out is what keeps a bounded gateway's latency low.
- report: one table over several run results, and the admission queue's depth over time.

Stage 0's memory is read from its laptop's supervisor while the load runs, and compared
with the budget (R1.7). A run passes when every offered request has its line, the load was
above capacity, the queue and the steps in flight stayed within their bounds, and the
worker did not go over the budget. A run whose memory could not be read says so, and does
not count as within the budget or over it.

Start a fresh gateway for each setting: the peaks it reports are kept since it started.

Usage:
uv run python -m pllm.x7 capacity smollm2-135m --metrics runs/<gateway>.metrics.jsonl
uv run python -m pllm.x7 run smollm2-135m --metrics runs/<gateway>.metrics.jsonl --rate 2.0
uv run python -m pllm.x7 report runs/*-x7.jsonl --out runs/x7-report
"""

import argparse
import json
import math
import random
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import grpc

from pllm import client, config, prompts, report, runrecord
from pllm.models import PINNED

REQUESTS = 120
NEW_TOKENS = 32
LENGTH = 128
SEED = 7
TIMEOUT_S = 120.0
BUDGET_MIB = 0.7 * 1024  # the Stage 0 worker's peak RSS budget on Host B (R1.7)
OVERLOAD = 1.1  # offered over completed requests per second, above which the load was over
SETTLE_S = 10.0  # at most this long for the last request lines to be written
MEMORY_POLL_S = 0.5
MIB = 1024 * 1024
# Two runs are compared only if they were offered the same load
SAME_LOAD = ("model", "rate", "requests", "new_tokens", "length", "seed", "timeout_s")


# ------------------------------------------------------------------ the load


def arrivals(rate: float, count: int, seed: int) -> list[float]:
    """When each request is due, in seconds from the first: gaps of a Poisson process with
    this mean rate, the same for the same seed."""
    rng = random.Random(seed)
    out, at = [], 0.0
    for _ in range(count):
        out.append(at)
        at += -math.log(1.0 - rng.random()) / rate
    return out


def offer(
    send: Callable[[int], str],
    schedule: list[float],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[str], float]:
    """Call send(i) at schedule[i], each on its own thread, so that no answer delays a
    later arrival. Returns what each call returned, and the latest any call started after
    it was due, in seconds."""
    outcomes = ["not sent"] * len(schedule)
    late = [0.0] * len(schedule)
    threads = []
    began = clock()

    def one(i: int) -> None:
        late[i] = max(0.0, clock() - began - schedule[i])
        try:
            outcomes[i] = send(i)
        except Exception as err:  # one broken request must not hide the others' results
            outcomes[i] = f"client error: {type(err).__name__}: {err}"

    for i, due in enumerate(schedule):
        wait = due - (clock() - began)
        if wait > 0:
            sleep(wait)
        threads.append(threading.Thread(target=one, args=(i,), name=f"x7-{i}", daemon=True))
        threads[-1].start()
    for thread in threads:
        thread.join()
    return outcomes, max(late, default=0.0)


def sender(call: Callable[[int], object]) -> Callable[[int], str]:
    """send(i) for offer: read every token of call(i) and name how it ended."""

    def send(i: int) -> str:
        try:
            for _ in call(i):
                pass
        except grpc.RpcError as err:
            return err.code().name
        return "OK"

    return send


# ------------------------------------------------------------------ Stage 0's memory


class MemorySampler:
    """Reads the Stage 0 worker's memory from its supervisor's /status until stopped."""

    def __init__(self, address: str, interval: float = MEMORY_POLL_S):
        self.url = f"http://{address}/status"
        self.interval = interval
        self.samples = 0
        self.failures = 0
        self.rss_max: int | None = None  # the largest resident set sampled in the load
        self.peak: int | None = None  # the worker's own lifetime peak, where reported
        self.pids: set[int] = set()  # more than one: the worker was restarted
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="x7-memory", daemon=True)

    def sample(self) -> None:
        try:
            with urllib.request.urlopen(self.url, timeout=2) as reply:
                status = json.load(reply)
            rss = status["worker_rss_bytes"]
        except (OSError, ValueError, KeyError, TypeError):
            self.failures += 1
            return
        if rss is None:  # no worker at this moment, or it could not be read
            self.failures += 1
            return
        self.samples += 1
        self.pids.add(status.get("pid"))
        self.rss_max = max(rss, self.rss_max or 0)
        peak = status.get("worker_peak_rss_bytes")
        if peak is not None:
            self.peak = max(peak, self.peak or 0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def result(self, budget_mib: float) -> dict:
        """The memory fields of a result. The peak is the worker's lifetime peak where the
        system reports it (it includes loading the model), else the largest sample."""
        values = [v for v in (self.peak, self.rss_max) if v is not None]
        worst = max(values) if values else None
        return {
            "samples": self.samples,
            "failed_samples": self.failures,
            "worker_restarts_seen": max(0, len(self.pids) - 1),
            "rss_max_bytes": self.rss_max,
            "lifetime_peak_bytes": self.peak,
            "budget_mib": budget_mib,
            "peak_mib": None if worst is None else worst / MIB,
            "within_budget": None if worst is None else worst <= budget_mib * MIB,
        }


# ------------------------------------------------------------------ the gateway's file


def read_lines(path: Path, offset: int = 0) -> list[dict]:
    """The metrics file's lines after offset bytes. A line cut mid-write is skipped."""
    with Path(path).open("rb") as file:
        file.seek(offset)
        data = file.read()
    out = []
    for text in data.decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(text)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def gateway_settings(path: Path) -> dict:
    """The flow-control setting the gateway wrote when it started."""
    run = report.load(Path(path))
    keys = ("admission_queue", "credits", "stage_inbox", "max_running", "model", "commit")
    missing = [k for k in keys if k not in run.info]
    if missing:
        raise ValueError(
            f"{path} does not say the gateway's {', '.join(missing)}: start the gateway "
            "from this version"
        )
    return {k: run.info[k] for k in keys}


def run_of(path: Path, rows: list[dict]) -> report.Run:
    """The request and beat lines of rows, as pllm.report reads a file."""
    run = report.Run(Path(path))
    for row in rows:
        kind = row.get("type")
        if any(key not in row for key in report.REQUIRED.get(kind, ())):
            continue
        if kind == "request":
            run.requests.append(row)
        elif kind == "beat":
            run.beats.append(row)
    return run


def wait_for_requests(path: Path, offset: int, count: int, timeout: float = SETTLE_S):
    """The lines after offset, once they hold count request lines or timeout has passed."""
    until = time.monotonic() + timeout
    while True:
        rows = read_lines(path, offset)
        if sum(r.get("type") == "request" for r in rows) >= count or time.monotonic() > until:
            return rows
        time.sleep(0.2)


def depth_series(rows: list[dict], origin_ns: int | None) -> list[dict]:
    """The depth lines as seconds from the first arrival."""
    return [
        {
            "type": "depth",
            "t_s": (r["t_ns"] - origin_ns) / 1e9,
            "running": r["running"],
            "queued": r["queued"],
            "in_flight": r["in_flight"],
        }
        for r in rows
        if r.get("type") == "depth" and origin_ns is not None
    ]


def analyse(rows: list[dict], run: report.Run, settings: dict, offered: int, memory: dict) -> dict:
    """The result of one run, from the lines the gateway wrote during it."""
    done = report.completed(run)
    codes: dict[str, int] = {}
    for r in run.requests:
        codes[r.get("code") or "OK"] = codes.get(r.get("code") or "OK", 0) + 1
    arrived = sorted(r["arrived_ns"] for r in run.requests if r.get("arrived_ns") is not None)
    span = report.window(run)
    offered_rps = completed_rps = None
    if len(arrived) > 1 and arrived[-1] > arrived[0]:
        offered_rps = (len(arrived) - 1) / ((arrived[-1] - arrived[0]) / 1e9)
    if span is not None:
        completed_rps = len(done) / ((span[1] - span[0]) / 1e9)
    latency = {}
    for key, values in report.latencies(run).items():
        latency[key] = {"n": len(values)}
        for p in report.PERCENTILES:
            latency[key][f"p{p}_ms"] = report.percentile(values, p) if values else None
    depths = [r for r in rows if r.get("type") == "depth"]
    queued_max = max((r["queued"] for r in depths), default=None)
    queue_peak = max((r["queue_peak"] for r in depths), default=None)
    in_flight_peak = max((r["in_flight_peak"] for r in depths), default=None)
    inbox = {}
    for stage in report.STAGES:
        seen = [b["inbox_depth"] for b in run.beats if b["stage"] == stage and "inbox_depth" in b]
        inbox[str(stage)] = max(seen, default=None)
    bound = settings["admission_queue"]
    queue_ok = None if bound is None or queue_peak is None else queue_peak <= bound
    credits_ok = None if in_flight_peak is None else in_flight_peak <= settings["credits"]
    # The rates alone could mislead: the completion window ends with the last request's whole
    # service time, so a short run of slow requests looks behind even when none ever waited.
    backed_up = codes.get("RESOURCE_EXHAUSTED", 0) > 0 or (queue_peak or 0) > 0
    overloaded = (
        backed_up
        and offered_rps is not None
        and completed_rps is not None
        and offered_rps > OVERLOAD * completed_rps
    )
    result = {
        "type": "result",
        "offered": offered,
        "request_lines": len(run.requests),
        "completed": len(done),
        "rejected": codes.get("RESOURCE_EXHAUSTED", 0),
        "deadline_exceeded": codes.get("DEADLINE_EXCEEDED", 0),
        "other_failed": len(run.requests)
        - len(done)
        - codes.get("RESOURCE_EXHAUSTED", 0)
        - codes.get("DEADLINE_EXCEEDED", 0),
        "codes": codes,
        "offered_rps": offered_rps,
        "completed_rps": completed_rps,
        "overloaded": overloaded,
        "latency": latency,
        "queued_max_sampled": queued_max,
        "queue_peak": queue_peak,
        "in_flight_peak": in_flight_peak,
        "stage_inbox_max": inbox,
        "admission_bound_held": queue_ok,
        "credit_bound_held": credits_ok,
        "memory": memory,
    }
    result["passed"] = (
        len(run.requests) == offered
        and overloaded
        and queue_ok is not False
        and credits_ok is True
        and memory["within_budget"] is not False
    )
    return result


# ------------------------------------------------------------------ the report


def read_result(path: Path) -> tuple[dict, dict, list[dict]]:
    """(setting, result, depth series) of one run's result file."""
    rows = read_lines(path)
    by_type = {r.get("type"): r for r in rows}
    if "setting" not in by_type or "result" not in by_type:
        raise ValueError(f"{path} is not a finished X7 result file")
    return by_type["setting"], by_type["result"], [r for r in rows if r["type"] == "depth"]


def label(setting: dict) -> str:
    bound = setting["admission_queue"]
    return f"queue {'off' if bound is None else bound}, credits {setting['credits']}"


def ordered(loaded: list) -> list:
    """Bounds on before off, then by credits. Refuses runs offered different loads."""
    loads = {tuple(setting[k] for k in SAME_LOAD) for setting, _, _ in loaded}
    if len(loads) > 1:
        raise ValueError(
            f"these runs were offered different loads and cannot be compared: "
            f"{', '.join(SAME_LOAD)} must match"
        )
    return sorted(
        loaded, key=lambda x: (x[0]["admission_queue"] is None, x[0]["credits"], label(x[0]))
    )


COLUMNS = (
    "admission queue",
    "credits",
    "offered req/s",
    "completed req/s",
    "completed",
    "rejected",
    "deadline",
    "other",
    "e2e p50 ms",
    "e2e p99 ms",
    "TTFT p99 ms",
    "queue peak",
    "in flight peak",
    "Stage 0 peak MiB",
    "within budget",
    "passed",
)


def table(loaded: list) -> list[str]:
    def show(value) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for setting, r, _ in ordered(loaded):
        bound = setting["admission_queue"]
        cells = [
            "off" if bound is None else bound,
            setting["credits"],
            r["offered_rps"],
            r["completed_rps"],
            r["completed"],
            r["rejected"],
            r["deadline_exceeded"],
            r["other_failed"],
            r["latency"]["e2e"]["p50_ms"],
            r["latency"]["e2e"]["p99_ms"],
            r["latency"]["ttft"]["p99_ms"],
            r["queue_peak"],
            r["in_flight_peak"],
            r["memory"]["peak_mib"],
            r["memory"]["within_budget"],
            r["passed"],
        ]
        lines.append("| " + " | ".join(show(c) for c in cells) + " |")
    return lines


def plot_depth(loaded: list, path: Path) -> None:
    """The admission queue's depth over time, one panel per setting, on shared axes."""
    plt = report._pyplot()
    runs = ordered(loaded)
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(9, 2.2 * len(runs)), sharex=True, sharey=True, squeeze=False
    )
    for ax, (setting, _, depth) in zip(axes[:, 0], runs, strict=True):
        times = [d["t_s"] for d in depth]
        ax.step(times, [d["queued"] for d in depth], where="post", label="waiting")
        ax.step(times, [d["in_flight"] for d in depth], where="post", label="steps in flight")
        ax.set_title(label(setting))
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("s since the first arrival")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(files: list[Path], folder: Path) -> list[str]:
    loaded = [read_result(path) for path in files]
    lines = table(loaded)
    folder.mkdir(parents=True, exist_ok=True)
    first = loaded[0][0]
    text = [
        "# X7: overload with bounds on and off",
        "",
        "Same offered load in every row: " + ", ".join(f"{k} {first[k]}" for k in SAME_LOAD),
        "",
        "Latencies are of completed requests only, on the gateway's clock.",
        "",
        *lines,
        "",
    ]
    (folder / "table.md").write_text("\n".join(text), encoding="utf-8", newline="\n")
    plot_depth(loaded, folder / "queue_depth.png")
    return lines


# ------------------------------------------------------------------ command line


def prompt_texts(model: str, length: int) -> list[str]:
    tokenizer = prompts.load_tokenizer(model)
    return [
        tokenizer.decode(ids, skip_special_tokens=False) for ids in prompts.prompts(model, length)
    ]


def gateway_call(model: str, length: int, new_tokens: int, timeout: float):
    """call(i): the gateway's token stream for the i-th prompt of the set."""
    texts = prompt_texts(model, length)
    address = config.load().gateway.address
    return lambda i: client.generate(address, texts[i % len(texts)], new_tokens, timeout)


def measure_capacity(call, metrics_path: Path, requests: int, concurrency: int) -> float:
    """Completed requests per second of a closed loop that keeps concurrency outstanding."""
    offset = Path(metrics_path).stat().st_size
    send = sender(call)
    numbers = iter(range(requests))
    lock = threading.Lock()
    outcomes: list[str] = []

    def worker() -> None:
        while True:
            with lock:
                i = next(numbers, None)
            if i is None:
                return
            outcome = send(i)
            with lock:
                outcomes.append(outcome)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [o for o in outcomes if o != "OK"]
    if failed:
        raise ValueError(f"{len(failed)} of {requests} calibration requests failed: {failed[0]}")
    run = run_of(metrics_path, wait_for_requests(metrics_path, offset, requests))
    span = report.window(run)
    if span is None:
        raise ValueError("the gateway's metrics file has no completed calibration request")
    return len(report.completed(run)) / ((span[1] - span[0]) / 1e9)


def run_load(call, metrics_path: Path, args, memory_address: str, out) -> dict:
    """Offer the load, read what the gateway wrote and write the result file's lines."""
    settings = gateway_settings(metrics_path)
    if settings["model"] != args.model:
        raise ValueError(f"the gateway serves {settings['model']}, not {args.model}")

    def emit(line: dict) -> None:
        out.write(json.dumps(line, allow_nan=False) + "\n")
        out.flush()

    emit(
        {
            "type": "setting",
            **settings,
            "rate": args.rate,
            "requests": args.requests,
            "new_tokens": args.new_tokens,
            "length": args.length,
            "seed": args.seed,
            "timeout_s": args.timeout,
        }
    )
    offset = Path(metrics_path).stat().st_size
    schedule = arrivals(args.rate, args.requests, args.seed)
    with MemorySampler(memory_address) as sampler:
        outcomes, latest = offer(sender(call), schedule)
        sampler.sample()  # the lifetime peak after the last step
    rows = wait_for_requests(metrics_path, offset, args.requests)
    run = run_of(metrics_path, rows)
    result = analyse(rows, run, settings, args.requests, sampler.result(args.budget_mib))
    seen: dict[str, int] = {}
    for outcome in outcomes:
        seen[outcome] = seen.get(outcome, 0) + 1
    result["client_outcomes"] = seen
    result["latest_send_s"] = latest  # how far the load generator fell behind its schedule
    arrived = [r["arrived_ns"] for r in run.requests if r.get("arrived_ns") is not None]
    for line in depth_series(rows, min(arrived, default=None)):
        emit(line)
    emit(result)
    return result


def positive(kind):
    def parse(text: str):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text}") from None
        if not 0 < value < float("inf"):
            raise argparse.ArgumentTypeError(f"must be more than 0, got {text}")
        return value

    return parse


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description="X7: overload with bounds on and off.")
    commands = top.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("capacity", "measure completed requests per second with every slot busy"),
        ("run", "offer a load above capacity to the running gateway"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("model", choices=sorted(PINNED))
        sub.add_argument("--metrics", type=Path, required=True, help="the gateway's metrics file")
        sub.add_argument("--new-tokens", type=positive(int), default=NEW_TOKENS)
        sub.add_argument("--length", type=int, choices=prompts.LENGTHS, default=LENGTH)
        sub.add_argument("--timeout", type=positive(float), default=TIMEOUT_S, help="seconds")
        sub.add_argument("--requests", type=positive(int), help="how many to send")
    run = commands.choices["run"]
    run.add_argument("--rate", type=positive(float), required=True, help="requests per second")
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--budget-mib", type=positive(float), default=BUDGET_MIB)
    table_parser = commands.add_parser("report", help="a table and plot over result files")
    table_parser.add_argument("files", type=Path, nargs="+")
    table_parser.add_argument("--out", type=Path, required=True)
    return top


def main(argv: list[str] | None = None) -> int:
    top = parser()
    args = top.parse_args(argv)
    if args.command == "report":
        try:
            print("\n".join(write_report(args.files, args.out)))
        except (OSError, ValueError) as err:
            top.error(str(err))
        return 0
    if not args.metrics.is_file():
        top.error(f"no metrics file at {args.metrics}")
    try:
        settings = gateway_settings(args.metrics)
        if settings["model"] != args.model:
            raise ValueError(f"the gateway serves {settings['model']}, not {args.model}")
        call = gateway_call(args.model, args.length, args.new_tokens, args.timeout)
        memory_address = config.supervisor_endpoint(0).address
    except (OSError, ValueError) as err:
        top.error(str(err))
    if args.command == "capacity":
        count = args.requests or 2 * settings["max_running"]
        try:
            rps = measure_capacity(call, args.metrics, count, settings["max_running"])
        except (OSError, ValueError) as err:
            top.error(str(err))
        print(f"capacity: {rps:.3f} requests/s ({label(settings)}, {count} requests)")
        print(f"a rate well above it, for example --rate {2 * rps:.2f}, overloads the gateway")
        return 0

    args.requests = args.requests or REQUESTS
    spec = PINNED[args.model]
    with runrecord.record(
        "x7", model=args.model, revision=spec.revision, split=config.split_point(args.model)
    ) as rec:
        path = runrecord.output_path(rec, ".jsonl")
        rec.update(
            results_file=path.name,
            metrics_file=args.metrics.name,
            gateway=settings,
            rate=args.rate,
            requests=args.requests,
            new_tokens=args.new_tokens,
            length=args.length,
            seed=args.seed,
            timeout_s=args.timeout,
        )
        with path.open("w", encoding="utf-8", newline="\n") as out:
            result = run_load(call, args.metrics, args, memory_address, out)
        rec["result"] = result
    print(json.dumps(result, indent=2))
    print(f"results: {path}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
