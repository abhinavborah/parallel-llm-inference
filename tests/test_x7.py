"""X7: overload with bounds on and off (R7.14).

The overload tests run a real gateway and two real stage servers on loopback, with a slowed
second stage as the injected cause of the overload, and drive them through the runner.
"""

import argparse
import io
import json
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_gateway import DIGEST, PROMPT, WAIT, System, Words
from test_model import tiny_reference

from pllm import gateway, metrics, report, supervisor, x7
from pllm.config import FlowLimits
from pllm.control import ExpectedModel
from pllm.proto import protocol_pb2

LAYERS = 4
K = 2
MIB = 1024 * 1024
RUNNING = 2  # running slots in the overload tests
QUEUE = 2
CREDITS = 1


@pytest.fixture
def system(tmp_path):
    made = []

    def make(max_queued, credits=CREDITS, delay1=0.05):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        run = System(
            tmp_path,
            K,
            delay1=delay1,
            max_workers=gateway.RESERVED_THREADS + RUNNING,
            max_queued=max_queued,
            credits=credits,
        )
        made.append(run)
        run.path = tmp_path / f"gateway{len(made)}.metrics.jsonl"
        run.gateway.recorder = metrics.Recorder(
            run.path,
            {
                "model": "tiny",
                "commit": None,
                "admission_queue": None if max_queued == gateway.UNBOUNDED_QUEUE else max_queued,
                "credits": credits,
                "stage_inbox": 16,
                "max_running": RUNNING,
            },
        )
        return run

    yield make
    for each in made:
        each.close()


class FakeSupervisor:
    """Answers /status as pllm.supervisor does."""

    def __init__(self, rss=100 * MIB, peak=120 * MIB, pids=(1,)):
        self.rss, self.peak, self.pids = rss, peak, list(pids)

    def status(self):
        pid = self.pids.pop(0) if len(self.pids) > 1 else self.pids[0]
        return {"pid": pid, "worker_rss_bytes": self.rss, "worker_peak_rss_bytes": self.peak}


@pytest.fixture
def serving():
    made = []

    def make(fake):
        server = supervisor.serve_control(fake, ("127.0.0.1", 0))
        made.append(server)
        return f"127.0.0.1:{server.server_address[1]}"

    yield make
    for server in made:
        server.shutdown()
        server.server_close()


def call_of(run, max_tokens):
    def call(i):
        request = protocol_pb2.GenerateReq(
            prompt=PROMPT, max_tokens=max_tokens, timeout_ms=WAIT * 1000
        )
        return run.stub.Generate(request, timeout=WAIT)

    return call


def load_args(**changes):
    values = dict(
        model="tiny",
        rate=60.0,
        requests=20,
        new_tokens=4,
        length=128,
        seed=7,
        timeout=float(WAIT),
        budget_mib=x7.BUDGET_MIB,
    )
    return argparse.Namespace(**{**values, **changes})


def overload(run, address):
    out = io.StringIO()
    result = x7.run_load(call_of(run, 4), run.path, load_args(), address, out)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    return result, lines


# ------------------------------------------------------------------ the arrival schedule


def test_arrivals_are_seeded_and_have_the_mean_rate():
    first = x7.arrivals(4.0, 20_000, 7)
    assert first == x7.arrivals(4.0, 20_000, 7)  # every setting gets the same schedule
    assert first != x7.arrivals(4.0, 20_000, 8)
    assert first[0] == 0.0 and first == sorted(first)
    gaps = [b - a for a, b in zip(first, first[1:], strict=False)]
    mean = sum(gaps) / len(gaps)
    assert mean == pytest.approx(0.25, rel=0.03)
    # Poisson arrivals: the gaps' spread is about their mean, unlike a fixed interval
    spread = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
    assert spread == pytest.approx(mean, rel=0.05)


def test_a_slow_answer_does_not_delay_later_arrivals():
    started = []

    def send(i):
        started.append(time.monotonic())
        time.sleep(0.5)  # every request is still unanswered when the next is due
        return "OK"

    began = time.monotonic()
    outcomes, latest = x7.offer(send, [0.0, 0.05, 0.1, 0.15, 0.2])
    assert outcomes == ["OK"] * 5
    assert max(started) - began < 0.45  # a closed loop would need two seconds
    assert latest < 0.25


def test_a_broken_send_is_named_and_the_others_still_run():
    def send(i):
        if i == 1:
            raise RuntimeError("broken")
        return "OK"

    outcomes, _ = x7.offer(send, [0.0, 0.0, 0.0])
    assert outcomes == ["OK", "client error: RuntimeError: broken", "OK"]


# ------------------------------------------------------------------ the gateway's setting


def test_the_admission_bound_can_be_set_or_turned_off():
    assert gateway.admission_places("0") == 0
    assert gateway.admission_places("8") == 8
    assert gateway.admission_places("off") == gateway.UNBOUNDED_QUEUE
    for bad in ("-1", "many", "1.5", str(gateway.UNBOUNDED_QUEUE)):
        with pytest.raises(argparse.ArgumentTypeError):
            gateway.admission_places(bad)


def test_overrides_replace_the_limits_and_credits_stay_within_the_stage_inbox():
    base = FlowLimits(admission_queue=8, credits=8, stage_inbox=16)
    assert gateway.flow_settings(base, None, None) == base
    assert gateway.flow_settings(base, 3, 2) == FlowLimits(3, 2, 16)
    assert gateway.flow_settings(base, 0, 16) == FlowLimits(0, 16, 16)
    for bad in (0, 17):
        with pytest.raises(ValueError, match="stage_inbox"):
            gateway.flow_settings(base, None, bad)


def test_a_metrics_file_without_the_setting_is_refused(tmp_path):
    path = tmp_path / "old.metrics.jsonl"
    metrics.Recorder(path, {"model": "tiny", "commit": None}).finish()
    with pytest.raises(ValueError, match="admission_queue"):
        x7.gateway_settings(path)


# ------------------------------------------------------------------ overload, injected


def test_with_bounds_on_the_queue_is_capped_and_the_excess_is_refused(system, serving):
    run = system(max_queued=QUEUE)
    result, lines = overload(run, serving(FakeSupervisor()))
    assert result["request_lines"] == 20 and result["offered"] == 20
    assert result["rejected"] > 0 and result["completed"] > 0
    assert result["completed"] + result["rejected"] == 20
    assert result["client_outcomes"] == {
        "OK": result["completed"],
        "RESOURCE_EXHAUSTED": result["rejected"],
    }
    assert result["queue_peak"] <= QUEUE and result["queued_max_sampled"] <= QUEUE
    assert result["in_flight_peak"] <= CREDITS
    assert result["admission_bound_held"] is True and result["credit_bound_held"] is True
    assert result["overloaded"] is True and result["passed"] is True
    assert result["latency"]["e2e"]["n"] == result["completed"]
    assert lines[0]["type"] == "setting" and lines[0]["admission_queue"] == QUEUE
    assert lines[-1]["type"] == "result"
    depth = [line for line in lines if line["type"] == "depth"]
    assert max(d["queued"] for d in depth) == result["queued_max_sampled"]
    assert [d["t_s"] for d in depth] == sorted(d["t_s"] for d in depth)


def test_with_the_bound_off_nothing_is_refused_and_the_queue_and_p99_grow(system, serving):
    address = serving(FakeSupervisor())
    bounded, _ = overload(system(max_queued=QUEUE), address)
    unbounded, _ = overload(system(max_queued=gateway.UNBOUNDED_QUEUE), address)
    assert unbounded["rejected"] == 0 and unbounded["completed"] == 20
    assert unbounded["queue_peak"] > QUEUE * 3  # far deeper than the bound would allow
    assert unbounded["admission_bound_held"] is None  # there is no bound to hold
    assert unbounded["in_flight_peak"] <= CREDITS  # the credit limit still applies
    assert unbounded["latency"]["e2e"]["p99_ms"] > 1.5 * bounded["latency"]["e2e"]["p99_ms"]
    assert unbounded["passed"] is True


def test_capacity_is_the_closed_loop_completion_rate(system):
    run = system(max_queued=QUEUE)
    rate = x7.measure_capacity(call_of(run, 3), run.path, requests=6, concurrency=RUNNING)
    done = report.completed(report.load(run.path))
    span = report.window(report.load(run.path))
    assert len(done) == 6 and rate == pytest.approx(6 / ((span[1] - span[0]) / 1e9))


def test_capacity_refuses_a_run_in_which_requests_failed(system):
    run = system(max_queued=0)
    with pytest.raises(ValueError, match="calibration requests failed"):
        x7.measure_capacity(call_of(run, 3), run.path, requests=8, concurrency=8)


def test_the_gateway_writes_a_depth_line_only_when_the_depth_changes(system):
    run = system(max_queued=QUEUE, delay1=0.0)
    time.sleep(0.3)  # several watcher passes with nothing happening
    assert len(list(call_of(run, 3)(0))) == 3
    time.sleep(0.3)
    rows = [r for r in x7.read_lines(run.path) if r["type"] == "depth"]
    kept = [
        (r["running"], r["queued"], r["in_flight"], r["queue_peak"], r["in_flight_peak"])
        for r in rows
    ]
    assert all(a != b for a, b in zip(kept, kept[1:], strict=False))
    assert len(kept) <= 8  # idle passes write nothing


def test_a_beat_line_keeps_the_stage_inbox_depth(tmp_path):
    path = tmp_path / "m.jsonl"
    recorder = metrics.Recorder(path)
    recorder.beat(1, protocol_pb2.Beat(stage=1, epoch=2, inbox_depth=5))
    recorder.finish()
    [beat] = [r for r in x7.read_lines(path) if r["type"] == "beat"]
    assert beat["inbox_depth"] == 5


# ------------------------------------------------------------------ what passes

SETTINGS = {"admission_queue": 2, "credits": 1, "stage_inbox": 16}
WITHIN = {"within_budget": True}


def request(i, code="OK"):
    done = code == "OK"
    return {
        "type": "request",
        "outcome": "max_tokens" if done else "rejected",
        "code": code,
        "output_tokens": 2 if done else 0,
        "arrived_ns": i * 10_000_000,  # 100 a second
        "first_token_ns": (i + 100) * 10_000_000 if done else None,
        "last_token_ns": (i + 101) * 10_000_000 if done else None,
    }


def depth(queued=2, peak=2, in_flight_peak=1):
    return {
        "type": "depth",
        "t_ns": 1,
        "running": 2,
        "queued": queued,
        "in_flight": 1,
        "queue_peak": peak,
        "in_flight_peak": in_flight_peak,
    }


def analysed(rows, offered=10, memory=WITHIN, settings=SETTINGS):
    return x7.analyse(rows, x7.run_of("m", rows), settings, offered, memory)


def test_a_run_passes_only_with_every_line_overload_held_bounds_and_memory_not_over():
    requests = [request(i) for i in range(10)]
    good = analysed([*requests, depth()])
    assert good["passed"] is True and good["overloaded"] is True
    assert analysed([*requests, depth(peak=3)])["admission_bound_held"] is False
    assert analysed([*requests, depth(peak=3)])["passed"] is False
    assert analysed([*requests, depth(in_flight_peak=2)])["passed"] is False
    assert analysed([*requests, depth()], offered=11)["passed"] is False  # a line is missing
    assert analysed(requests)["passed"] is False  # no depth lines: the bounds are unproven
    assert analysed([*requests, depth()], memory={"within_budget": False})["passed"] is False
    unknown = analysed([*requests, depth()], memory={"within_budget": None})
    assert unknown["passed"] is False  # the budget criterion cannot be met without a reading
    assert any("memory is unknown" in why for why in unknown["failures"])
    assert good["failures"] == []
    # nothing waited in this run: a peak left by an earlier run on the gateway does not count
    stale = analysed([*requests, depth(queued=0, peak=2)])
    assert stale["overloaded"] is False and stale["passed"] is False
    slow = [  # answered faster than they arrive: not an overload
        {
            **request(i),
            "arrived_ns": i * 10**9,
            "first_token_ns": i * 10**9 + 1,
            "last_token_ns": i * 10**9 + 2,
        }
        for i in range(10)
    ]
    assert analysed([*slow, depth()])["overloaded"] is False
    assert analysed([*slow, depth()])["passed"] is False
    tail = [  # arrive quickly and each take long, but none waits or is refused
        {
            **request(i),
            "first_token_ns": i * 10_000_000 + 10**9,
            "last_token_ns": i * 10_000_000 + 2 * 10**9,
        }
        for i in range(10)
    ]
    unqueued = analysed([*tail, depth(queued=0, peak=0)])
    assert unqueued["offered_rps"] > x7.OVERLOAD * unqueued["completed_rps"]
    assert unqueued["overloaded"] is False and unqueued["passed"] is False


def test_refused_and_failed_requests_are_counted_beside_the_latencies():
    rows = [request(i) for i in range(6)]
    rows += [request(6, "RESOURCE_EXHAUSTED"), request(7, "RESOURCE_EXHAUSTED")]
    rows += [request(8, "DEADLINE_EXCEEDED"), request(9, "UNAVAILABLE"), depth()]
    result = analysed(rows)
    assert (result["completed"], result["rejected"]) == (6, 2)
    assert (result["deadline_exceeded"], result["other_failed"]) == (1, 1)
    assert result["latency"]["e2e"]["n"] == 6  # completed requests only
    assert result["latency"]["e2e"]["p99_ms"] == pytest.approx(report.percentile([1010.0] * 6, 99))


def beat(stage, inbox):
    return {
        "type": "beat",
        "t_ns": 5,
        "stage": stage,
        "epoch": 1,
        "busy_ns": 0,
        "steps": 0,
        "inbox_depth": inbox,
    }


def test_a_stage_inbox_over_its_bound_fails_the_run():
    rows = [*(request(i) for i in range(10)), depth()]
    within = analysed([*rows, beat(0, 1), beat(1, 16)])
    assert within["stage_inbox_held"] is True and within["passed"] is True
    over = analysed([*rows, beat(0, 1), beat(1, 17)])
    assert over["stage_inbox_held"] is False and over["passed"] is False
    assert "a stage inbox went over its bound" in over["failures"]
    assert analysed(rows)["stage_inbox_held"] is None  # no beat to judge by


def test_a_cancelled_request_is_not_counted_as_ok():
    cancelled = {**request(1, "RESOURCE_EXHAUSTED"), "outcome": "cancelled by the client"}
    cancelled["code"] = None
    result = analysed([request(0), cancelled, depth()], offered=2)
    assert result["codes"] == {"OK": 1, "cancelled by the client": 1}
    assert (result["completed"], result["other_failed"]) == (1, 1)


# ------------------------------------------------------------------ Stage 0's memory


def sampled(address, budget=700.0):
    sampler = x7.MemorySampler(address, interval=0.02)
    with sampler:
        time.sleep(0.15)
    sampler.sample()  # as run_load does after the load
    return sampler.result(budget)


class Counting(FakeSupervisor):
    """A supervisor whose restart count rises after the first reading."""

    reads = 0

    def status(self):
        self.reads += 1
        return {**super().status(), "restarts": 0 if self.reads == 1 else 1}


@pytest.mark.parametrize("seen_by", ["a new pid", "the supervisor's restart count"])
def test_a_restarted_worker_leaves_the_budget_unknown(serving, seen_by):
    """The dead process's peak is lost, so a small new worker proves nothing."""
    fake = FakeSupervisor(pids=[1, 2]) if seen_by == "a new pid" else Counting()
    result = sampled(serving(fake))
    assert result["worker_restarts_seen"] == 1 and result["within_budget"] is None


def test_a_restart_does_not_hide_a_reading_over_the_budget(serving):
    result = sampled(serving(FakeSupervisor(peak=710 * MIB, pids=[1, 2])))
    assert result["within_budget"] is False


def test_a_failed_last_reading_leaves_the_budget_unknown(serving):
    fake = FakeSupervisor()
    sampler = x7.MemorySampler(serving(fake), interval=0.02)
    sampler.sample()
    assert sampler.result(700.0)["within_budget"] is True
    fake.rss = None  # the worker is gone when the load ends
    sampler.sample()
    result = sampler.result(700.0)
    assert result["samples"] == 1 and result["last_sample_read"] is False
    assert result["within_budget"] is None


def test_a_reply_cut_short_is_a_failed_sample_not_the_end_of_the_sampler():
    class Cut(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b'{"wo')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Cut)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sampler = x7.MemorySampler(f"127.0.0.1:{server.server_address[1]}", interval=0.02)
        sampler.sample()  # raises nothing
        with sampler:
            time.sleep(0.2)
            assert sampler._thread.is_alive()
        assert sampler.failures >= 2 and sampler.samples == 0
    finally:
        server.shutdown()
        server.server_close()


def test_the_lifetime_peak_is_read_against_the_budget(serving):
    result = sampled(serving(FakeSupervisor(rss=500 * MIB, peak=710 * MIB)))
    assert result["samples"] > 0 and result["peak_mib"] == 710
    assert result["peak_source"] == "lifetime"
    assert result["within_budget"] is False
    assert sampled(serving(FakeSupervisor(rss=500 * MIB, peak=690 * MIB)))["within_budget"]


def test_without_a_lifetime_peak_the_largest_sample_is_used(serving):
    result = sampled(serving(FakeSupervisor(rss=650 * MIB, peak=None)))
    assert result["lifetime_peak_bytes"] is None and result["peak_source"] == "samples"
    assert result["peak_mib"] == 650 and result["within_budget"] is True


def test_a_restarted_worker_is_reported(serving):
    result = sampled(serving(FakeSupervisor(pids=[1, 2])))
    assert result["worker_restarts_seen"] == 1


@pytest.mark.parametrize("case", ["no supervisor", "an older supervisor", "no worker"])
def test_memory_that_cannot_be_read_is_unknown_never_within_budget(serving, case):
    if case == "no supervisor":
        address = "127.0.0.1:1"
    else:
        fake = FakeSupervisor()
        if case == "an older supervisor":
            fake.status = lambda: {"pid": 1, "state": "running"}
        else:
            fake.rss = None
        address = serving(fake)
    sampler = x7.MemorySampler(address, interval=0.02)
    sampler.sample()
    result = sampler.result(700.0)
    assert result["samples"] == 0 and result["failed_samples"] == 1
    assert result["within_budget"] is None and result["peak_mib"] is None


# ------------------------------------------------------------------ files and the report


def test_only_lines_after_the_offset_are_read_and_a_cut_line_is_skipped(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_bytes(b'{"type":"old"}\n')
    offset = path.stat().st_size
    with path.open("ab") as file:
        file.write(b'{"type":"new"}\n{"type":"cu')
    assert x7.read_lines(path, offset) == [{"type": "new"}]


def result_file(tmp_path, name, admission_queue, credits, **changes):
    setting = {
        "type": "setting",
        "model": "tiny",
        "admission_queue": admission_queue,
        "credits": credits,
        "rate": 5.0,
        "requests": 10,
        "new_tokens": 4,
        "length": 128,
        "seed": 7,
        "timeout_s": 60.0,
        **changes,
    }
    rows = [request(i) for i in range(10)]
    result = analysed([*rows, depth()], memory={"within_budget": True, "peak_mib": 500.0})
    series = {"type": "depth", "t_s": 0.5, "running": 2, "queued": 2, "in_flight": 1}
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in (setting, series, result)))
    return path


def test_the_report_tables_bounds_on_first_and_plots_the_depth(tmp_path):
    files = [
        result_file(tmp_path, "off", None, 8),
        result_file(tmp_path, "c8", 8, 8),
        result_file(tmp_path, "c1", 8, 1),
    ]
    lines = x7.write_report(files, tmp_path / "out")
    rows = [line for line in lines if line.startswith("|")][2:]
    assert [line.split("|")[1:3] for line in rows] == [
        [" 8 ", " 1 "],
        [" 8 ", " 8 "],
        [" off ", " 8 "],
    ]
    written = (tmp_path / "out" / "table.md").read_text(encoding="utf-8")
    assert "completed requests only" in written and "rate 5.0" in written
    assert (tmp_path / "out" / "queue_depth.png").stat().st_size > 1000


def test_the_report_says_per_credit_limit_whether_the_bound_keeps_p99_down(tmp_path):
    def with_p99(name, queue, credits, p99):
        path = result_file(tmp_path, name, queue, credits)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["latency"]["e2e"]["p99_ms"] = p99
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    files = [
        with_p99("on8", 8, 8, 100.0),
        with_p99("off8", None, 8, 300.0),
        with_p99("on2", 8, 2, 900.0),
        with_p99("off2", None, 2, 400.0),
        with_p99("on1", 8, 1, 50.0),
    ]
    lines = x7.write_report(files, tmp_path / "out")
    found = {line.split(":")[0]: line for line in lines if line.startswith("- ")}
    assert "is below the bound-off 300.00 ms" in found["- queue 8, credits 8"]
    assert "is NOT below the bound-off 400.00 ms" in found["- queue 8, credits 2"]
    assert "not compared" in found["- queue 8, credits 1"]
    written = (tmp_path / "out" / "table.md").read_text(encoding="utf-8")
    assert all(line in written for line in found.values())


@pytest.mark.parametrize(
    ("flags", "queue", "credits"),
    [
        ([], 8, 8),
        (["--admission-queue", "off", "--credits", "2", "--no-kv-admission"], None, 2),
    ],
)
def test_the_gateway_command_applies_the_flags_and_records_them(
    tmp_path, monkeypatch, flags, queue, credits
):
    """gateway.main itself, with only the model files and the serving stood in for."""
    seen = {}
    monkeypatch.setattr(gateway, "expected_model", lambda *a: ExpectedModel(DIGEST, 30, 15))
    shape = types.SimpleNamespace(hidden_size=64)
    monkeypatch.setattr(gateway.ModelConfig, "from_file", lambda *a: shape)
    monkeypatch.setattr(gateway, "Tokenizer", lambda *a: Words())
    monkeypatch.setattr(gateway, "eos_ids", lambda *a: [])
    monkeypatch.setattr(gateway, "max_positions", lambda *a: None)
    monkeypatch.setattr(gateway, "stage_weight_bytes", lambda *a: 0)
    monkeypatch.setattr(gateway, "bytes_per_position", lambda *a: 1000)
    monkeypatch.setattr(gateway, "stop_event", threading.Event)
    monkeypatch.setattr(gateway, "serve", lambda built, *a: (None, 0))

    def stopped(built, *args):
        seen["gateway"] = built
        built.recorder.finish()

    monkeypatch.setattr(gateway, "run_until_stopped", stopped)
    path = tmp_path / "m.jsonl"
    gateway.main(["smollm2-135m", "--metrics", str(path), *flags])
    settings = x7.gateway_settings(path)
    assert (settings["admission_queue"], settings["credits"]) == (queue, credits)
    assert settings["stage_inbox"] == 16 and settings["max_running"] == gateway.MAX_RUNNING
    assert seen["gateway"].credits == credits
    assert (seen["gateway"].kv_budget is None) == ("--no-kv-admission" in flags)
    assert seen["gateway"].max_queued == (gateway.UNBOUNDED_QUEUE if queue is None else queue)


def test_the_gateway_command_refuses_credits_above_the_stage_inbox(monkeypatch, capsys):
    monkeypatch.setattr(gateway, "expected_model", lambda *a: ExpectedModel(DIGEST, 30, 15))
    shape = types.SimpleNamespace(hidden_size=64)
    monkeypatch.setattr(gateway.ModelConfig, "from_file", lambda *a: shape)
    monkeypatch.setattr(gateway, "Tokenizer", lambda *a: Words())
    with pytest.raises(SystemExit):
        gateway.main(["smollm2-135m", "--credits", "17"])
    assert "stage_inbox" in capsys.readouterr().err


def test_runs_offered_different_loads_are_not_compared(tmp_path):
    files = [result_file(tmp_path, "a", 8, 8), result_file(tmp_path, "b", None, 8, rate=9.0)]
    with pytest.raises(ValueError, match="different loads"):
        x7.write_report(files, tmp_path / "out")


def test_an_unfinished_result_file_is_refused(tmp_path):
    path = tmp_path / "cut.jsonl"
    path.write_text('{"type":"setting"}\n')
    with pytest.raises(ValueError, match="not a finished"):
        x7.read_result(path)


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "smollm2-135m", "--metrics", "x"],  # no rate
        ["run", "smollm2-135m", "--metrics", "x", "--rate", "0"],
        ["run", "smollm2-135m", "--metrics", "x", "--rate", "1", "--requests", "0"],
        ["capacity", "smollm2-135m"],  # no metrics file
        ["capacity", "smollm2-135m", "--metrics", "no-such-file"],
        ["report", "a.jsonl"],  # no output folder
    ],
)
def test_bad_arguments_are_refused(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        x7.main(argv)
    assert caught.value.code == 2


def test_offer_returns_when_every_thread_has_ended():
    done = threading.Event()

    def send(i):
        time.sleep(0.1)
        done.set()
        return "OK"

    outcomes, _ = x7.offer(send, [0.0])
    assert done.is_set() and outcomes == ["OK"]
