"""X6 runner (R7.13): the gateway's discard lines, the standby, the suspend that is always
resumed, how a repeat is judged, and the experiment with each stage a real supervised
process, suspended through its control port and replaced through a standby.

The model is the small random one, so these need no downloaded weights.
"""

import json
import signal
import sys
import tomllib

import psutil
import pytest
from test_gateway import DIGEST, Words
from test_supervisor import free_port
from test_x5 import AT, LAYERS, PROMPTS, K, failures_of, live  # noqa: F401 (fixture)

from pllm import config, metrics, standby, x5, x6
from pllm.control import ExpectedModel, Registry
from pllm.gateway import Gateway, Session
from pllm.proto import protocol_pb2
from pllm.supervisor import Supervisor, serve_control
from pllm.wire import KEEPALIVE_TIMEOUT_SECONDS

# ---------------------------------------------------------------- the discard lines


def test_each_dropped_result_and_nack_is_one_line_with_its_reason(tmp_path):
    gateway = Gateway(Registry("tiny", ExpectedModel(DIGEST, LAYERS, K)), Words(), (), 64, 10**6)
    path = tmp_path / "m.jsonl"
    gateway.recorder = metrics.Recorder(path)
    result = protocol_pb2.StepResult(rid="gone", attempt=1, step=4, epoch=7, token=3)
    gateway.Push(iter([result]), None)
    session = Session(1, 9)
    gateway._sessions[1] = session
    nack = protocol_pb2.Nack(stage=1, epoch=9, rid="gone", attempt=1, step=4, reason="BAD_SENDER")
    gateway._record(session, protocol_pb2.StageMsg(nack=nack))
    gateway.recorder.finish()
    rows = [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines()]
    found = [r for r in rows if r["type"] == "discard"]
    for row in found:
        assert isinstance(row.pop("t_ns"), int)
    common = {"type": "discard", "rid": "gone", "attempt": 1, "step": 4, "stage": 1}
    assert found == [
        {**common, "reason": "unknown_request", "source": "result", "epoch": 7,
         "nack_reason": None},
        {**common, "reason": "stale_nack", "source": "nack", "epoch": 9,
         "nack_reason": "BAD_SENDER"},
    ]  # fmt: skip
    assert gateway.discards == {"unknown_request": 1, "stale_nack": 1}


# ---------------------------------------------------------------- always resumed


class FakeControl:
    """Records the faults sent; fail maps an action to the errors its next calls raise."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}
        self.endpoints = {s: config.Endpoint("192.0.2.1", 50061 + s) for s in (0, 1)}

    def fault(self, stage, action):
        self.calls.append(action)
        errors = self.fail.get(action)
        if errors:
            raise errors.pop(0)
        return 42


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(x6, "RESUME_PAUSE_S", 0)


@pytest.mark.parametrize("error", [RuntimeError("broke"), KeyboardInterrupt(), SystemExit(3)])
def test_every_way_out_of_the_block_resumes(error):
    control = FakeControl()
    with pytest.raises(type(error)):
        with x6.suspended(control, 1):
            raise error
    assert control.calls == ["suspend", "resume"]


def test_a_resume_in_the_block_is_not_sent_again():
    control = FakeControl()
    with x6.suspended(control, 1) as hold:
        assert hold.resume() == 42
    assert control.calls == ["suspend", "resume"] and hold.held_s >= 0


def test_a_suspend_that_fails_is_still_followed_by_a_resume():
    """The answer may be lost although the supervisor did suspend."""
    control = FakeControl({"suspend": [x5.ControlError("timed out")]})
    with pytest.raises(x5.ControlError, match="timed out"):
        with x6.suspended(control, 1):
            raise AssertionError("not reached")
    assert control.calls == ["suspend", "resume"]


def test_an_interrupt_during_the_suspend_resumes():
    control = FakeControl({"suspend": [KeyboardInterrupt()]})
    with pytest.raises(KeyboardInterrupt):
        with x6.suspended(control, 1):
            pass
    assert control.calls == ["suspend", "resume"]


def test_a_resume_is_tried_until_the_supervisor_answers(quick):
    control = FakeControl({"resume": [x5.ControlError("no route")] * 3})
    with x6.suspended(control, 1) as hold:
        pass
    assert control.calls == ["suspend", *["resume"] * 4] and hold.resumed


def test_a_resume_that_never_succeeds_says_how_to_do_it_by_hand(quick, capsys):
    control = FakeControl({"resume": [x5.ControlError("no route")] * x6.RESUME_TRIES})
    with pytest.raises(x6.ResumeFailed, match="MAY STILL BE SUSPENDED"):
        with x6.suspended(control, 0):
            pass
    shown = capsys.readouterr().err
    assert "curl -X POST" in shown and "http://192.0.2.1:50061/resume" in shown


def test_no_worker_means_nothing_is_left_suspended(quick):
    control = FakeControl({"resume": [x5.ControlError("no worker", 409)]})
    with x6.suspended(control, 1) as hold:
        assert hold.resume() is None
    assert control.calls == ["suspend", "resume"] and hold.held_s is None


def test_a_second_ctrl_c_cannot_stop_a_resume():
    with x6.signals_ignored():
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN


def test_a_termination_signal_interrupts_like_ctrl_c():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with x5.interrupts_raise():
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == before


# ---------------------------------------------------------------- the standby


def test_the_replacement_hosts_file_moves_only_its_own_port(tmp_path, monkeypatch):
    hosts = config.load()
    for stage in (0, 1):
        path = tmp_path / f"s{stage}.toml"
        path.write_text(standby.replacement_hosts(hosts, stage), encoding="utf-8")
        made = config.load(path)
        moved, kept = f"stage{stage}", f"stage{1 - stage}"
        assert getattr(made, moved).port == getattr(hosts, moved).port + standby.PORT_OFFSET
        assert getattr(made, moved).host == getattr(hosts, moved).host
        assert getattr(made, kept) == getattr(hosts, kept)
        assert made.gateway == hosts.gateway and made.network == hosts.network
        assert "supervisor0" not in tomllib.loads(path.read_text(encoding="utf-8"))


def test_the_standby_listens_beside_the_supervisor_and_no_two_ports_collide():
    hosts = config.load()
    ports = [getattr(hosts, role).port for role in config.ROLES]
    ports.append(config.dashboard_endpoint().port)
    for stage in (0, 1):
        supervisor = config.supervisor_endpoint(stage)
        end = standby.standby_endpoint(stage)
        assert (end.host, end.port) == (supervisor.host, supervisor.port + standby.PORT_OFFSET)
        ports += [supervisor.port, end.port, getattr(hosts, f"stage{stage}").port + 100]
    assert len(ports) == len(set(ports))


def idle_worker():
    return Supervisor([sys.executable, "-c", "import time; time.sleep(600)"])


@pytest.fixture
def served_standby():
    made = standby.Standby(idle_worker)
    server = standby.serve_standby(made, ("127.0.0.1", 0))
    end = config.Endpoint("127.0.0.1", server.server_address[1])
    yield x6.StandbyControl({1: end}), end
    made.stop()
    server.shutdown()


def test_the_standby_starts_one_replacement_and_stops_it(served_standby):
    control, end = served_standby
    assert control.status(1)["state"] == "idle"
    control.post(1, "start")
    pid = x5.wait_until(lambda: control.status(1).get("pid"), "the replacement", 30)
    with pytest.raises(x5.ControlError) as refused:
        control.post(1, "start")
    assert refused.value.status == 409
    control.post(1, "stop")
    assert control.status(1)["state"] == "idle"
    x5.wait_until(lambda: None if psutil.pid_exists(pid) else True, "the worker to die", 30)
    control.post(1, "stop")  # stopping when idle succeeds


def test_the_standby_refuses_other_paths_and_a_post_with_no_content_type(served_standby):
    _, end = served_standby
    with pytest.raises(x5.ControlError) as unknown:
        x5.http_json(end, "POST", "/kill", 5)
    assert unknown.value.status == 404
    import urllib.error
    import urllib.request

    request = urllib.request.Request(f"http://{end.address}/start", data=b"", method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with pytest.raises(urllib.error.HTTPError) as refused:
        opener.open(request, timeout=5)
    assert refused.value.code == 415


def test_control_reports_a_refusal_with_its_status_and_no_answer_with_none():
    supervisor = idle_worker()  # never started: it has no worker
    server = serve_control(supervisor, ("127.0.0.1", 0))
    try:
        end = config.Endpoint("127.0.0.1", server.server_address[1])
        control = x5.Control({1: end})
        assert control.status(1)["pid"] is None
        with pytest.raises(x5.ControlError) as refused:
            control.fault(1, "kill")
        assert refused.value.status == 409
        with pytest.raises(x5.CheckFailed, match="no running worker"):
            x5.require_running(1, control.status(1))
    finally:
        server.shutdown()
        server.server_close()
    nobody = x5.Control({1: config.Endpoint("127.0.0.1", free_port())}, timeout=2)
    with pytest.raises(x5.ControlError) as silent:
        nobody.status(1)
    assert silent.value.status is None


# ---------------------------------------------------------------- judging a repeat


def hop(name, rid, attempt, accepted=True, step=3):
    return {"type": "hop", "hop": name, "rid": rid, "attempt": attempt, "step": step,
            "accepted": accepted}  # fmt: skip


def late_line(stage, rid="a", attempt=1):
    row = {"type": "discard", "rid": rid, "attempt": attempt, "step": 3}
    if stage == 1:
        return {**row, "source": "result", "epoch": 2, "reason": "stale_attempt"}
    return {**row, "source": "nack", "stage": 1, "epoch": 5, "reason": "stale_nack",
            "nack_reason": "BAD_SENDER"}  # fmt: skip


def window(stage):
    """A window that passes: the rows and the position at which the resume was sent."""
    before = [hop("gateway_stage0", rid, 1) for rid in "ab"]
    before.append(
        {"type": "detected", "declared": True, "stage": stage, "epoch": 2,
         "path": "heartbeat_timeout"}
    )  # fmt: skip
    before.append({"type": "registered", "stage": stage, "old_epoch": 2, "new_epoch": 3})
    before.append(hop("gateway_stage0", "a", 2))
    before.append(hop("stage1_gateway", "a", 2))
    return before + [late_line(stage), late_line(stage, "b")], len(before)


def stale(rows, resume_at, stage, held=3.0):
    return x6.analyse_stale(
        rows, stage=stage, old_epoch=2, rids={"a", "b"}, resume_at=resume_at, held_s=held
    )


@pytest.mark.parametrize("stage", [0, 1])
def test_late_outputs_of_the_resumed_stage_are_counted_by_reason(stage):
    rows, at = window(stage)
    data, failures = stale(rows, at, stage)
    assert failures == []
    name = "stale_attempt" if stage == 1 else "stale_nack:BAD_SENDER"
    assert data == {"held_s": 3.0, "late_outputs": 2, "late_by_reason": {name: 2}}


@pytest.mark.parametrize("stage", [0, 1])
def test_with_no_late_output_the_rejection_was_not_shown(stage):
    rows, at = window(stage)
    data, failures = stale(rows[:at], at, stage)
    assert data["late_outputs"] == 0
    assert any("no late output" in f for f in failures)


@pytest.mark.parametrize("stage", [0, 1])
def test_a_drop_written_before_the_resume_is_not_a_late_output(stage):
    rows, at = window(stage)
    data, failures = stale(rows, len(rows), stage)
    assert data["late_outputs"] == 0 and failures


def test_other_drops_are_not_the_old_stage_s_outputs():
    old = {"a": 1}
    another_epoch = {**late_line(1), "epoch": 3}
    another_request = late_line(1, "zz")
    assert x6.late_outputs([another_epoch, another_request], 1, 2, old) == []
    newer = late_line(0, attempt=2)
    gap = {**late_line(0), "nack_reason": "GAP"}
    from_stage0 = {**late_line(0), "stage": 0}
    assert x6.late_outputs([newer, gap, from_stage0, late_line(1)], 0, 2, old) == []


@pytest.mark.parametrize(
    "stage, extra, held, message",
    [
        (1, [hop("stage1_gateway", "a", 1)], 3.0, "were accepted"),
        (0, [hop("stage1_gateway", "a", 1)], 3.0, "were accepted"),
        (0, [hop("stage1_gateway", "a", 1, False)], 3.0, "ran a late step"),
        (1, [], KEEPALIVE_TIMEOUT_SECONDS, "keepalive timeout"),
        (1, [], None, "how long"),
    ],
)
def test_a_window_that_breaks_a_claim_fails(stage, extra, held, message):
    rows, at = window(stage)
    _, failures = stale(rows + extra, at, stage, held)
    assert any(message in f for f in failures), failures


def test_a_resume_before_the_replacement_served_fails():
    rows, at = window(1)
    for resume_at in (None, at - 1):  # never, or before the accepted replayed result
        data, failures = stale(rows, resume_at, 1)
        assert data["late_outputs"] is None
        assert any("after its replacement had served" in f for f in failures)


def test_unknown_attempts_before_the_fault_never_pass():
    rows, at = window(1)
    _, failures = stale(rows[2:], at - 2, 1)
    assert any("before the fault is unknown" in f for f in failures)


def test_the_replacement_has_served_only_with_a_newer_attempt_accepted():
    old = {"a": 1}
    assert not x6.served([hop("stage1_gateway", "a", 1)], old)
    assert not x6.served([hop("stage1_gateway", "a", 2, False)], old)
    assert not x6.served([hop("gateway_stage0", "a", 2)], old)
    assert x6.served([hop("stage1_gateway", "a", 2)], old)


# ---------------------------------------------------------------- running a repeat again


class Healthy:
    requests, new_tokens, fault_at, single_host = 1, 2, 1, True

    def healthy(self):
        return {0: 1, 1: 2}


def attempts(monkeypatch, outcomes, extra):
    """run_experiment over repeats whose failures are given in turn."""
    monkeypatch.setattr(x5, "references", lambda *a: [])
    left = list(outcomes)

    def one(ctx, stage, repeat, refs, epochs):
        data = {"duplicate_tokens": 0}
        return x5.repeat_line("x6", stage, repeat, data, left.pop(0))

    return x5.run_experiment(
        Healthy(), "x6", [1], 2, [[1]], one, lambda row: None, lambda lines: [],
        x6.only_not_shown, extra,
    )  # fmt: skip


def test_by_default_a_repeat_with_no_late_output_fails_the_run(monkeypatch):
    lines, summary = attempts(monkeypatch, [[x6.NOT_SHOWN], []], 0)
    assert len(lines) == 2 and not summary["passed"] and summary["replaced"] == []


def test_an_extra_attempt_replaces_it_and_stays_listed(monkeypatch):
    lines, summary = attempts(monkeypatch, [[x6.NOT_SHOWN], [], []], 1)
    assert [row.get("replaced", False) for row in lines] == [True, False, False]
    assert [row["repeat"] for row in lines] == [0, 0, 1] and not lines[0]["passed"]
    assert summary["passed"] and summary["passed_repeats"] == 2 and summary["ran_repeats"] == 3
    assert summary["replaced"][0]["failures"] == [x6.NOT_SHOWN]


def test_the_extra_attempts_run_out(monkeypatch):
    lines, summary = attempts(monkeypatch, [[x6.NOT_SHOWN], [x6.NOT_SHOWN], []], 1)
    assert len(lines) == 3 and not summary["passed"]
    assert summary["failed"][0]["repeat"] == 0


def test_a_repeat_with_any_other_failure_is_never_run_again(monkeypatch):
    lines, summary = attempts(monkeypatch, [[x6.NOT_SHOWN, "duplicate tokens"], []], 5)
    assert len(lines) == 2 and not summary["passed"] and summary["replaced"] == []


# ---------------------------------------------------------------- real processes


class Standing:
    """A standby for a Live system's stage, served over HTTP as pllm.standby serves it."""

    def __init__(self, system, make=None):
        self.standby = standby.Standby(make or system.replacement)
        self.server = standby.serve_standby(self.standby, ("127.0.0.1", 0))
        end = config.Endpoint("127.0.0.1", self.server.server_address[1])
        self.control = x6.StandbyControl({system.stage: end})

    def close(self):
        self.standby.stop()
        self.server.shutdown()


@pytest.fixture
def standing():
    made = []

    def make(system, **kwargs):
        made.append(Standing(system, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def not_stopped(pid):
    return psutil.Process(pid).status() != psutil.STATUS_STOPPED


@pytest.mark.fault
@pytest.mark.parametrize("stage", [1, 0], ids=["stage1", "stage0"])
def test_x6_suspends_a_stage_lets_a_replacement_serve_and_resumes_it(live, standing, stage):  # noqa: F811
    system = live(stage)
    spare = standing(system)
    ctx = system.context(x6.Context, standby=spare.control, late_wait=10)
    lines, summary = x5.run_experiment(
        ctx,
        "x6",
        [stage],
        2,  # twice: restoring leaves the stage on its own supervisor again
        PROMPTS,
        x6.x6_repeat,
        lambda row: None,
        lambda done: x5.aggregate("x6", done, x6.SPREAD),
    )
    assert summary["passed"], failures_of(lines, summary)
    assert summary["duplicate_tokens"] == 0 and len(lines) == 2
    for row in lines:
        assert row["path"] == "heartbeat_timeout"
        assert row["held_s"] < KEEPALIVE_TIMEOUT_SECONDS and row["old_alive_after_resume"]
        assert row["late_outputs"] >= 1
        reasons = set(row["late_by_reason"])
        if stage == 1:
            assert reasons <= {"stale_attempt", "unknown_request", "stale_epoch"}
            assert sum(r["stale_discarded"] for r in row["requests"]) >= 1
        else:
            assert reasons == {"stale_nack:BAD_SENDER"}
        for request in row["requests"]:
            assert request["duplicate_tokens"] == 0 and request["tokens_at_fault"] >= AT
            assert request["output_equals_reference"] is True
    assert lines[1]["old_epoch"] > lines[0]["new_epoch"]  # the supervisor's own again
    assert spare.control.status(stage)["state"] == "idle"
    status = system.control.status(stage)
    assert status["state"] == "running" and not_stopped(status["pid"])


@pytest.mark.fault
def test_x6_resumes_the_stage_and_fails_when_no_replacement_serves(live, standing):  # noqa: F811
    system = live(1)
    spare = standing(system, make=idle_worker)  # a replacement that never registers
    ctx = system.context(x6.Context, standby=spare.control, late_wait=2)
    jobs = x5.references(ctx, PROMPTS, lambda row: None)
    epochs = ctx.healthy()
    ctx.wait = 4
    row = x6.x6_repeat(ctx, 1, 0, jobs, epochs)
    assert not row["passed"]
    assert any("the replacement to serve" in f for f in row["failures"]), row["failures"]
    assert row["late_outputs"] is None
    assert spare.control.status(1)["state"] == "idle"
    ctx.wait = 60
    ctx.healthy()  # the resumed worker left, told it was dead, and was respawned
    assert system.control.status(1)["state"] == "running"


@pytest.mark.fault
def test_x6_interrupted_while_the_stage_is_suspended_still_resumes_it(live, standing):  # noqa: F811
    system = live(1)
    spare = standing(system)
    resumed = []

    class Interrupted(x6.StandbyControl):
        def post(self, stage, action):
            if action == "start":
                assert system.control.status(stage)["state"] == "suspended"
                raise KeyboardInterrupt
            super().post(stage, action)

    class Watched(x5.Control):
        def fault(self, stage, action):
            pid = super().fault(stage, action)
            if action == "resume":
                resumed.append(not_stopped(pid))
            return pid

    ctx = system.context(
        x6.Context,
        standby=Interrupted(spare.control.endpoints),
        control=Watched(system.control.endpoints),
        late_wait=2,
    )
    jobs = x5.references(ctx, PROMPTS, lambda row: None)
    epochs = ctx.healthy()
    with pytest.raises(KeyboardInterrupt):
        x6.x6_repeat(ctx, 1, 0, jobs, epochs)
    assert resumed and all(resumed)
    assert system.control.status(1)["state"] != "suspended"
    ctx.wait = 60
    ctx.healthy()
