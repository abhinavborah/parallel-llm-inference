"""X5 runner (R7.12): how it judges a repeat, its argument checks, and the experiment with
each stage's worker a real process under a real supervisor, killed through its control port.

The model is the small random one, so these need no downloaded weights. Equality is on one
host, fp32, batch 1.
"""

import json
import sys

import pytest
from test_gateway import wait_for
from test_heartbeat import Beating, reference
from test_model import tiny_reference
from test_supervisor import free_port

from pllm import config, metrics, runrecord, x5
from pllm.control import ExpectedModel
from pllm.models import sha256_of
from pllm.supervisor import Supervisor, serve_control

LAYERS = 4
K = 2
BEAT = 0.05
TIMEOUT = 1.0  # heartbeat timeout: 20 beats, so a slow runner does not declare a live stage
DELAY = 0.03  # per step on the stage in this process
# The stage under test is the slower one, so it nearly always has a step in hand: a
# suspended stage 1 sends a late result only for a step it held when it was suspended.
OWN_DELAY = 0.1
N = 24  # new tokens
AT = 6  # the fault point
PROMPTS = [[5, 9, 100, 3, 42, 7], [3, 42, 7], [11, 12, 13, 14], [20, 21, 22, 23, 24]]

# ---------------------------------------------------------------- judging a repeat


def good(stage=1, old=2, new=3, path="stream_break", requests=4, tokens=8):
    """The jobs, references and metrics lines of a repeat that passes."""
    jobs, refs, rows = [], [], []
    rows.append({"type": "detected", "declared": True, "stage": stage, "epoch": old, "path": path})
    rows.append({"type": "registered", "stage": stage, "old_epoch": old, "new_epoch": new})
    for i in range(requests):
        rid = f"r{i}"
        out = list(range(10 * i, 10 * i + tokens))
        logits = [0.5 + i + t for t in range(tokens)]
        ref = x5.Job(i, [i + 1], f"ref{i}", out, list(range(tokens)), logits, done=True)
        job = x5.Job(i, [i + 1], rid, list(out), list(range(tokens)), list(logits), done=True)
        job.at_fault = 3
        jobs.append(job)
        refs.append(ref)
        rows.append(
            {
                "type": "recovery",
                "t_ns": 1000 + i,
                "rid": rid,
                "attempt": 2,
                "recovered": True,
                "stage": stage,
                "old_epoch": old,
                "new_epoch": new,
                "path": path,
                "detection_ns": 5_000_000,
                "respawn_ns": 7_000_000,
                "weight_load_ns": 11_000_000,
                "overlap": False,
                "replay_prefill_ns": (i + 1) * 1_000_000,
                "recomputed_tokens": 4,
                "divergences": 0,
                "total_ns": (i + 1) * 100_000_000,
            }
        )
        rows.append(
            {"type": "request", "rid": rid, "outcome": "max_tokens", "output_tokens": tokens}
        )
    return jobs, refs, rows


def judge(jobs, refs, rows, **kwargs):
    kwargs.setdefault("single_host", True)
    return x5.analyse(rows, jobs, refs, stage=1, old_epoch=2, fault_at=3, new_tokens=8, **kwargs)


def of(rows, kind, n=0):
    return [r for r in rows if r["type"] == kind][n]


def test_a_complete_repeat_passes_and_reports_the_four_parts():
    data, failures = judge(*good())
    assert failures == []
    assert (data["detection_ms"], data["respawn_ms"], data["weight_load_ms"]) == (5, 7, 11)
    assert data["replay_prefill_ms"] == pytest.approx(2.5)  # the mean of the requests
    assert data["total_ms"] == pytest.approx(250) and data["total_ms_max"] == 400
    assert (data["path"], data["old_epoch"], data["new_epoch"]) == ("stream_break", 2, 3)
    assert data["duplicate_tokens"] == 0
    assert [r["recomputed_tokens"] for r in data["requests"]] == [4] * 4
    assert [r["tokens_at_fault"] for r in data["requests"]] == [3] * 4


def drop(kind):
    def mutate(jobs, rows):
        rows.remove(of(rows, kind))

    return mutate


def change(kind, **fields):
    def mutate(jobs, rows):
        of(rows, kind).update(fields)

    return mutate


def append(row):
    def mutate(jobs, rows):
        rows.append(row)

    return mutate


def duplicated_token(jobs, rows):
    jobs[0].tokens.insert(2, jobs[0].tokens[2])
    jobs[0].steps.insert(2, jobs[0].steps[2])


def other_output(jobs, rows):
    jobs[1].tokens[-1] += 1


def other_logit(jobs, rows):
    jobs[1].logits[-1] += 1e-6


def failed_request(jobs, rows):
    jobs[2].error = "UNAVAILABLE: gone"


def no_token(jobs, rows):
    jobs[3].rid = None


def repeated(kind):
    def mutate(jobs, rows):
        rows.append(dict(of(rows, kind)))

    return mutate


def incomplete_recovery(jobs, rows):
    del of(rows, "recovery")["recomputed_tokens"]


DROPPED = {"type": "hop", "hop": "stage1_gateway", "rid": "r0", "attempt": 1, "step": 3}
ACCEPTED = {**DROPPED, "attempt": 2, "accepted": True}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (duplicated_token, "duplicate tokens at the client"),
        (other_output, "differs from the fault-free reference"),
        (other_logit, "top logits differ"),
        (failed_request, "the request failed"),
        (no_token, "no token came"),
        (drop("detected"), "the fault did not take effect"),
        (drop("registered"), "no new epoch"),
        (change("registered", new_epoch=None), "no new epoch"),
        (
            append({"type": "detected", "declared": True, "stage": 0, "epoch": 1, "path": "x"}),
            "another death",
        ),
        (drop("request"), "0 request lines"),
        (change("request", outcome="end of sequence", output_tokens=5), "the gateway ended"),
        (change("request", outcome="failed"), "the gateway ended"),
        (
            append({"type": "request", "rid": "zz", "outcome": "max_tokens"}),
            "another request ran",
        ),
        (drop("recovery"), "0 recovery lines"),
        (repeated("recovery"), "2 recovery lines"),
        (incomplete_recovery, "an incomplete recovery line"),
        (change("recovery", rid="zz"), "another request recovered"),
        (change("recovery", recovered=False), "not recovered"),
        (change("recovery", old_epoch=9), "its recovery names"),
        (change("recovery", weight_load_ns=None), "unknown weight_load_ms"),
        (change("recovery", detection_ns=None), "unknown detection_ms"),
        (change("recovery", respawn_ns=None), "unknown respawn_ms"),
        (change("recovery", replay_prefill_ns=None), "unknown replay_prefill_ms"),
        (change("recovery", divergences=1), "divergences 1"),
        (change("recovery", divergences=None), "divergences None"),
        (change("recovery", recomputed_tokens=2), "did not die between"),
        (change("recovery", recomputed_tokens=8), "did not die between"),
        (append({"type": "divergence", "rid": "r0", "step": 1}), "replay diverged"),
        (append({**DROPPED, "accepted": False}), "no discard line"),
        (lambda jobs, rows: rows.extend([ACCEPTED, dict(ACCEPTED)]), "accepted 1 results twice"),
    ],
)
def test_a_repeat_fails_when_a_check_fails_or_cannot_be_made(mutate, message):
    jobs, refs, rows = good()
    mutate(jobs, rows)
    _, failures = judge(jobs, refs, rows)
    assert any(message in f for f in failures), failures


def test_no_metrics_lines_at_all_never_pass():
    jobs, refs, _ = good()
    data, failures = judge(jobs, refs, [])
    assert failures and data["detection_ms"] is None and data["total_ms"] is None


def test_one_unknown_time_leaves_the_mean_unknown():
    jobs, refs, rows = good()
    of(rows, "recovery")["total_ns"] = None
    data, _ = judge(jobs, refs, rows)
    assert data["total_ms"] is None and data["total_ms_max"] is None


def test_the_death_path_can_be_required():
    _, failures = judge(*good(), death_path="heartbeat_timeout")
    assert any("detected by stream_break" in f for f in failures)


def test_top_logits_are_compared_only_on_one_host():
    jobs, refs, rows = good()
    other_logit(jobs, rows)
    data, failures = judge(jobs, refs, rows, single_host=False)
    assert failures == []
    assert data["requests"][1]["top_logits_equal_reference"] is None


def test_dropped_results_are_counted_by_reason_and_per_request():
    jobs, refs, rows = good()
    rows.append({**DROPPED, "accepted": False})
    rows.append({**DROPPED, "type": "discard", "source": "result", "reason": "stale_attempt"})
    rows.append(
        {
            "type": "discard",
            "source": "nack",
            "reason": "stale_nack",
            "nack_reason": "BAD_SENDER",
            "rid": "r0",
        }
    )
    data, failures = judge(jobs, refs, rows)
    assert failures == []
    assert data["discards_by_reason"] == {"stale_attempt": 1, "stale_nack:BAD_SENDER": 1}
    assert [r["stale_discarded"] for r in data["requests"]] == [1, 0, 0, 0]


# ---------------------------------------------------------------- aggregates, summary


def line(stage, repeat, passed=True, detection=5.0, duplicates=0):
    return x5.repeat_line(
        "x5",
        stage,
        repeat,
        {
            "path": "stream_break",
            "detection_ms": detection,
            "duplicate_tokens": duplicates,
            "discards_by_reason": {"stale_attempt": 1},
        },
        [] if passed else ["broke"],
    )


def test_the_spread_is_over_the_passed_repeats_of_each_stage():
    lines = [line(1, 0, detection=4), line(1, 1, detection=6), line(1, 2, False, 99)]
    lines.append(line(0, 0))
    rows = {r["stage"]: r for r in x5.aggregate("x5", lines)}
    assert rows[1]["repeats"] == 3 and rows[1]["passed_repeats"] == 2
    assert rows[1]["detection_ms"]["n"] == 2 and rows[1]["detection_ms"]["mean"] == 5
    assert rows[1]["detection_ms"]["max"] == 6
    assert rows[1]["discards_by_reason"] == {"stale_attempt": 2}
    assert "late_by_reason" not in rows[1]


def test_the_summary_passes_only_when_every_planned_repeat_ran_and_passed():
    lines = [line(1, 0), line(0, 0)]
    assert x5.summary_line("x5", lines, 2, None)["passed"]
    assert not x5.summary_line("x5", lines, 3, None)["passed"]  # one did not run
    assert not x5.summary_line("x5", lines, 2, "unhealthy")["passed"]
    assert not x5.summary_line("x5", [], 0, None)["passed"]
    failed = x5.summary_line("x5", [line(1, 0, False), line(0, 0)], 2, None)
    assert not failed["passed"] and failed["failed"][0]["failures"] == ["broke"]


def test_duplicate_tokens_are_unknown_unless_every_repeat_counted_them():
    lines = [line(1, 0), line(0, 0)]
    assert x5.summary_line("x5", lines, 2, None)["duplicate_tokens"] == 0
    assert x5.summary_line("x5", lines, 3, None)["duplicate_tokens"] is None
    del lines[0]["duplicate_tokens"]
    assert x5.summary_line("x5", lines, 2, None)["duplicate_tokens"] is None


# ---------------------------------------------------------------- reading the file


def test_the_tail_takes_only_whole_lines_and_counts_the_unreadable(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_bytes(b'{"type": "a"}\n{"type": "b"')
    tail = x5.MetricsTail(path)
    assert [r["type"] for r in tail.poll()] == ["a"]
    mark = tail.mark()
    with path.open("ab") as out:
        out.write(b'}\nnot json\n[1]\n{"type": "c"}\n')
    assert [r["type"] for r in tail.since(mark)] == ["b", "c"]
    assert tail.bad == 2


def test_the_live_epochs_need_a_registration_a_beat_and_no_death():
    rows = []
    with pytest.raises(x5.CheckFailed, match="no registration"):
        x5.live_epochs(rows)
    rows += [{"type": "registered", "stage": s, "new_epoch": s + 1} for s in (0, 1)]
    with pytest.raises(x5.CheckFailed, match="not beaten"):
        x5.live_epochs(rows)
    rows += [{"type": "beat", "stage": s, "epoch": s + 1} for s in (0, 1)]
    assert x5.live_epochs(rows) == {0: 1, 1: 2}
    rows.append({"type": "detected", "declared": False, "stage": 1, "epoch": 2})
    assert x5.live_epochs(rows) == {0: 1, 1: 2}  # a step deadline declares nothing
    rows.append({"type": "detected", "declared": True, "stage": 1, "epoch": 2})
    with pytest.raises(x5.CheckFailed, match="dead"):
        x5.live_epochs(rows)


def test_a_request_that_ends_before_the_fault_point_stops_the_repeat():
    class Token:
        rid, step, token, top_logit = "r", 0, 1, 0.5

    load = x5.Load(lambda ids, n: iter([Token()]), [x5.Job(0, [1])], 8).start()
    with pytest.raises(x5.CheckFailed, match="ended before the fault point"):
        load.wait_for_tokens(3, 5)


def test_a_call_that_breaks_is_an_error_not_a_short_answer():
    def call(ids, n):
        raise ValueError("boom")

    job = x5.Job(0, [1])
    x5.serve_one(call, job, 4)
    assert job.done and "boom" in job.error


# ---------------------------------------------------------------- the command line


def hosts_of(host0="127.0.0.1"):
    end = config.Endpoint
    return config.HostConfig("n", end("127.0.0.1", 1), end(host0, 2), end("127.0.0.1", 3))


def test_one_host_means_every_process_and_supervisor_is_on_this_laptop():
    local = {"127.0.0.1"}
    here = {s: config.Endpoint("127.0.0.1", 10 + s) for s in (0, 1)}
    there = {**here, 0: config.Endpoint("192.0.2.7", 10)}
    assert x5.local_to_this_laptop(hosts_of(), here, local)
    assert not x5.local_to_this_laptop(hosts_of("192.0.2.7"), here, local)
    assert not x5.local_to_this_laptop(hosts_of(), there, local)


def gateway_file(tmp_path, **info):
    path = tmp_path / "gw.metrics.jsonl"
    info = {"model": "smollm2-135m", "discard_lines": True, **info}
    metrics.Recorder(path, info)  # left open, as a running gateway's file
    return path


@pytest.mark.parametrize(
    "extra, info, message",
    [
        (["--repeats", "4"], {}, "at least 5 repeats"),
        (["--requests", "2"], {}, "--trial allows another load"),
        (["--fault-at", "10"], {}, "--trial allows another load"),
        (["--trial", "--fault-at", "64"], {}, "--fault-at < --new-tokens"),
        (["--stages", "1", "1"], {}, "names a stage twice"),
        ([], {"discard_lines": False}, "does not write discard lines"),
        ([], {"model": "other"}, "the gateway serves other"),
    ],
)
def test_bad_arguments_are_refused(tmp_path, capsys, extra, info, message):
    path = gateway_file(tmp_path, **info)
    with pytest.raises(SystemExit):
        x5.main(["smollm2-135m", "--metrics", str(path), *extra])
    assert message in capsys.readouterr().err


def test_a_run_across_laptops_needs_its_round_trip_time(tmp_path, capsys, monkeypatch):
    path = gateway_file(tmp_path)
    monkeypatch.setattr(x5, "local_to_this_laptop", lambda *a: False)
    with pytest.raises(SystemExit):
        x5.main(["smollm2-135m", "--metrics", str(path)])
    assert "--rtt-ms" in capsys.readouterr().err


@pytest.mark.parametrize("passed", [True, False])
def test_the_exit_code_and_run_record_follow_the_summary(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path)
    args = x5.parser_for("t").parse_args(["smollm2-135m", "--metrics", "m", "--rtt-ms", "9"])
    summary = {"type": "summary", "passed": passed, "error": None, "failed": []}

    def run(emit):
        emit(summary)
        return [], summary

    code = x5.record_and_run("x5", args, {"commit": "abc"}, False, run)
    assert code == (0 if passed else 1)
    [record] = tmp_path.glob("*-x5.json")
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["summary"] == summary and saved["rtt_ms"] == 9
    assert saved["reference"] == x5.TWO_LAPTOPS and saved["gateway_commit"] == "abc"
    lines = (tmp_path / saved["results_file"]).read_text(encoding="utf-8").splitlines()
    assert [json.loads(row) for row in lines] == [summary]


# ---------------------------------------------------------------- real processes

CHILD = """
import os, sys, threading
from pathlib import Path
from pllm.stage_service import serve
from pllm.worker import StageWorker, WorkerConfig, exit_when_parent_gone, join_gateway
model_dir, stage, k, n, gateway, neighbour, port, delay, beat = sys.argv[1:10]
stage, k, n = int(stage), int(k), int(n)
layers = (0, k) if stage == 0 else (k, n)
where = {"next_address": neighbour} if stage == 0 else {"results_address": gateway}
worker = StageWorker(WorkerConfig(Path(model_dir), *layers, float(delay)))
server, servicer, _ = serve(worker, f"127.0.0.1:{port}", **where)
exit_when_parent_gone(int(os.environ["PLLM_SUPERVISOR_PID"]))
try:
    join_gateway(gateway, stage, layers, "tiny", Path(model_dir), f"127.0.0.1:{port}",
                 servicer, threading.Event(), interval=float(beat))
except RuntimeError:  # declared dead: leave, as a real worker does, to be respawned
    server.stop(0)
    sys.exit(1)
server.wait_for_termination()  # superseded: no more beats, but it serves on
"""


class Live:
    """A gateway, one stage in this process and the stage under test as a real process
    under a real supervisor with its control port. replacement() makes the supervisor a
    standby would start: the same worker on another port."""

    def __init__(self, tmp_path, stage):
        tiny_reference(tmp_path, layers=LAYERS)
        self.tmp_path, self.stage, other = tmp_path, stage, 1 - stage
        self.run = run = Beating(tmp_path, K, interval=BEAT, order=(), delay0=DELAY,
                                 delay1=DELAY)  # fmt: skip
        self.metrics = tmp_path / "gateway.metrics.jsonl"
        run.gateway.recorder = metrics.Recorder(
            self.metrics, {"model": "tiny", "discard_lines": True}
        )
        run.join(other, (run.servicer0, run.servicer1)[other])
        run.gateway.detector.heartbeat_timeout_ns = round(TIMEOUT * 1e9)
        # the children register the real weights' digest, not the stand-in the harness used
        digest = bytes.fromhex(sha256_of(tmp_path / "model.safetensors"))
        run.registry.expected = ExpectedModel(digest, LAYERS, K)
        self.neighbour = "" if stage == 1 else f"127.0.0.1:{run.ports[1]}"
        self.made: list[Supervisor] = []
        self.supervisor = self.replacement("primary")
        self.server = serve_control(self.supervisor, ("127.0.0.1", 0))
        self.supervisor.start()
        port = self.server.server_address[1]
        self.control = x5.Control({stage: config.Endpoint("127.0.0.1", port)})

    def replacement(self, name="replacement") -> Supervisor:
        port = free_port()
        args = [str(self.tmp_path), str(self.stage), str(K), str(LAYERS), self.run.address]
        rest = [self.neighbour, str(port), str(OWN_DELAY), str(BEAT)]
        supervisor = Supervisor(
            [sys.executable, "-c", CHILD, *args, *rest],
            ready_address=("127.0.0.1", port),
            log_path=self.tmp_path / f"{name}.jsonl",
        )
        self.made.append(supervisor)
        return supervisor

    def call(self, ids, max_tokens):
        prompt = " ".join(map(str, ids))
        return self.run.call(prompt=prompt, max_tokens=max_tokens)

    def context(self, kind=x5.Context, **fields):
        fields = {"call": self.call, "control": self.control, **fields}
        ctx = kind(
            tail=x5.MetricsTail(self.metrics),
            single_host=True,
            requests=len(PROMPTS),
            new_tokens=N,
            fault_at=AT,
            timeout=25,
            wait=90,
            settle=0.5,
            **fields,
        )
        ctx.healthy()  # the first worker has loaded and both stages are routed
        ctx.wait = 25
        return ctx

    def close(self):
        for supervisor in self.made:
            supervisor.stop()  # kills its worker, suspended or not
        self.server.shutdown()
        self.run.close()


@pytest.fixture
def live(tmp_path):
    made = []

    def make(stage):
        made.append(Live(tmp_path, stage))
        return made[-1]

    yield make
    for each in made:
        each.close()


def failures_of(lines, summary):
    return [summary["error"], *[f for row in lines for f in row["failures"]]]


@pytest.mark.fault
@pytest.mark.parametrize("stage", [1, 0], ids=["stage1", "stage0"])
def test_x5_kills_a_supervised_stage_with_four_requests_in_flight(live, tmp_path, stage):
    system = live(stage)
    ctx = system.context()
    emitted = []
    lines, summary = x5.run_experiment(
        ctx,
        "x5",
        [stage],
        2,  # twice: the system is whole again after a repeat
        PROMPTS,
        x5.x5_repeat,
        emitted.append,
        lambda done: x5.aggregate("x5", done),
    )
    assert summary["passed"], failures_of(lines, summary)
    assert summary["duplicate_tokens"] == 0 and len(lines) == 2
    assert [row["type"] for row in emitted] == [
        "plan", "reference", "reference", "reference", "reference",
        "repeat", "repeat", "aggregate", "summary",
    ]  # fmt: skip
    # the reference the runner took from the system is the single-process one
    for row, ids in zip(emitted[1:5], PROMPTS, strict=True):
        assert row["tokens"] == reference(tmp_path, K, ids, N)[0]
    for row in lines:
        assert row["detection_ms"] > 0 and row["weight_load_ms"] > 0
        assert row["respawn_ms"] is not None and row["supervisor_respawn_s"] > 0
        assert row["pid_after"] != row["pid_before"] and row["new_epoch"] > row["old_epoch"]
        assert len(row["requests"]) == 4
        for request in row["requests"]:
            assert AT <= request["tokens_at_fault"] <= request["recomputed_tokens"] < N
            assert request["replay_prefill_ms"] > 0
            assert request["output_equals_reference"] is True
            assert request["top_logits_equal_reference"] is True
    assert lines[1]["old_epoch"] == lines[0]["new_epoch"]
    wait_for(lambda: system.run.gateway.table() == [], "every request to leave the table")


@pytest.mark.fault
def test_x5_fails_when_the_kill_does_not_happen(live):
    system = live(1)

    class Pretends(x5.Control):
        def fault(self, stage, action):
            return self.status(stage)["pid"]  # answers as if done, and does nothing

    ctx = system.context(control=Pretends(system.control.endpoints))
    ctx.wait = 3
    jobs = x5.references(ctx, PROMPTS, lambda row: None)
    row = x5.x5_repeat(ctx, 1, 0, jobs, ctx.healthy())
    assert not row["passed"]
    assert any("report a new worker" in f for f in row["failures"]), row["failures"]
    assert any("the fault did not take effect" in f for f in row["failures"])
    assert row["detection_ms"] is None
