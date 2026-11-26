"""Unrecoverable and slow requests fail cleanly (R6.9, R6.10).

- Each declared death raises the attempt of every request in the table by exactly one.
- A request that lives through MAX_DEATHS stage deaths fails with UNAVAILABLE at the last.
- A refused replay starts again after a growing, jittered pause, never in a tight loop.
- A stage slower than the step timeout fails each admitted request at its deadline, the
  excess is refused at once, and the admission queue, credits and inboxes keep their bounds.

A request that cannot be recovered because no stage comes back is tested in
test_replay.py (it fails with DEADLINE_EXCEEDED at its deadline).
"""

import threading
import time

import grpc
import pytest
from test_backpressure import run_all
from test_backpressure import system as system  # noqa: F401 (a fixture used below)
from test_deadlines import TOLERANCE
from test_detection import start_replacement, stop_stage
from test_gateway import IDS, code_of, wait_for
from test_heartbeat import Beating, reference
from test_model import tiny_reference

from pllm import gateway as gateway_module
from pllm import metrics
from pllm.faults import StepDelay
from pllm.stage_service import StageServicer

pytestmark = pytest.mark.fault

LAYERS = 4
K = 2
BEAT = 0.05


@pytest.fixture
def beating(tmp_path):
    made = []

    def make(**kwargs):
        tiny_reference(tmp_path, layers=LAYERS)
        made.append(Beating(tmp_path, K, interval=BEAT, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def starts(gateway):
    """(time, rid, attempt) of every step 0 the gateway sends to Stage 0."""
    rows = []
    real = gateway.recorder.hop

    def hop(name, message, accepted=True):
        if name == metrics.GATEWAY_STAGE0 and message.step == 0:
            rows.append((time.monotonic(), message.rid, message.attempt))
        return real(name, message, accepted)

    gateway.recorder.hop = hop
    return rows


def at_declarations(gateway):
    """For each declared death, the attempt of every request just before and just after."""
    seen = []
    real = gateway._on_detection

    def sink(record):
        before = {row["rid"]: row["attempt"] for row in gateway.table()}
        real(record)
        if record["event"] == "detected" and record["declared"]:
            after = {row["rid"]: row["attempt"] for row in gateway.table()}
            seen.append((before, after))

    gateway.detector._sink = sink
    return seen


class Stages:
    """The current server, servicer and heartbeat stop of each stage, so a replacement can
    be killed in its turn (the harness's stop_stage only stops the first two)."""

    def __init__(self, run):
        self.run = run
        self.current = {  # Beating joins Stage 1 first, so its heartbeat is stops[0]
            0: (run.server0, run.servicer0, run.stops[1]),
            1: (run.server1, run.servicer1, run.stops[0]),
        }

    def kill_and_replace(self, stage):
        run = self.run
        server, servicer, stop = self.current[stage]
        old = servicer.own_epoch
        stop.set()
        server.stop(0).wait()
        servicer.close()
        wait_for(lambda: run.gateway.detector.is_dead(old), "the death to be declared")
        new, _ = start_replacement(run, stage)
        new.worker.delay = servicer.worker.delay  # keep requests in flight for the next kill
        self.current[stage] = (run.extra[-1][0], new, run.stops[-1])
        run.wait_routes(servicer0=self.current[0][1], servicer1=self.current[1][1])


# -------------------------------------------------------------- R6.9: one per recovery


def test_each_declared_death_raises_every_attempt_by_exactly_one(beating, tmp_path):
    """Two requests, Stage 1 then Stage 0 killed. At each declaration every request in the
    table goes up by one, each replay starts under a newer attempt than the last, and both
    finish with the fault-free tokens."""
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 12)[0]
    stages = Stages(run)
    seen = at_declarations(run.gateway)
    sent = starts(run.gateway)
    streams = [run.call(max_tokens=12) for _ in range(2)]
    got = [[next(stream) for _ in range(2)] for stream in streams]
    stages.kill_and_replace(1)
    for stream, tokens in zip(streams, got, strict=True):
        tokens.append(next(stream))  # a new token after the first recovery
    stages.kill_and_replace(0)
    for stream, tokens in zip(streams, got, strict=True):
        tokens += list(stream)
        assert [t.token for t in tokens] == expected
        assert [t.step for t in tokens] == list(range(12))
    assert len(seen) == 2
    for before, after in seen:
        assert after == {rid: attempt + 1 for rid, attempt in before.items()}
    for rid in {tokens[0].rid for tokens in got}:
        attempts = [attempt for _, r, attempt in sent if r == rid]
        assert attempts[0] == 1 and attempts == sorted(set(attempts))  # never reused
        assert attempts[-1] >= 3  # two deaths, so at least two above the first


# -------------------------------------------------------------- R6.9: the cap


def test_a_request_that_lives_through_max_deaths_fails_with_unavailable(beating):
    run = beating(delay1=0.05)
    stages = Stages(run)
    stream = run.call(max_tokens=400, timeout_ms=60_000)
    rid = next(stream).rid
    for _ in range(gateway_module.MAX_DEATHS - 1):
        stages.kill_and_replace(1)
        next(stream)  # still served after this many deaths
    assert run.gateway.requests[rid].deaths == gateway_module.MAX_DEATHS - 1
    began = time.monotonic()
    replays = run.gateway.replays
    stages.kill_and_replace(1)
    code, details = code_of(stream)
    assert code == grpc.StatusCode.UNAVAILABLE
    assert f"gave up after {gateway_module.MAX_DEATHS} stage deaths" in details
    assert time.monotonic() - began < 10  # long before its deadline
    assert run.gateway.replays == replays  # the last death started no replay
    wait_for(lambda: not run.gateway.requests, "the failed request to leave the table")


# -------------------------------------------------------------- backoff after a refusal


def test_a_refused_replay_waits_longer_each_time(beating, tmp_path, monkeypatch):
    """The new Stage 1 learns Stage 0's epoch only after 1.2 s, so it refuses each replay
    until then. The pauses between the restarts grow, so the restarts stay few, and the
    request still finishes with the fault-free tokens."""
    hold = 1.2
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 6)[0]
    sent = starts(run.gateway)
    stream = run.call(max_tokens=6, timeout_ms=30_000)
    tokens = [next(stream), next(stream)]
    first = {run.servicer0, run.servicer1}
    real_route = StageServicer.route

    def late(self, update):
        if self not in first and update.next_stage == 0:  # the new stage's upstream epoch
            threading.Timer(hold, real_route, (self, update)).start()
            return True
        return real_route(self, update)

    monkeypatch.setattr(StageServicer, "route", late)
    old = run.registry.stages[1].epoch
    stop_stage(run, 1)
    wait_for(lambda: run.gateway.detector.is_dead(old), "the death to be declared")
    start_replacement(run, 1)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    times = [at for at, rid, attempt in sent if rid == tokens[0].rid and attempt >= 2]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert len(gaps) >= 3  # refused at least three times
    assert min(gaps) >= 0.9 * gateway_module.POLL_SECONDS
    assert gaps[-1] > 2 * gaps[0]  # the pause grew
    # a fixed 0.05 s pause would restart about 24 times in the hold; doubling gives about 5
    assert len(times) <= 9


def test_the_pause_doubles_up_to_its_cap_and_never_passes_the_deadline(monkeypatch):
    slept = []
    monkeypatch.setattr(gateway_module.time, "sleep", slept.append)
    monkeypatch.setattr(gateway_module.random, "uniform", lambda a, b: b)  # the longest
    request = gateway_module.Request("r", 1, time.monotonic() + 60, 3)
    for _ in range(8):
        gateway_module.Gateway._pause(request)
    cap = gateway_module.MAX_BACKOFF_SECONDS * 1.5
    assert slept == pytest.approx([0.075, 0.15, 0.3, 0.6, 1.2, cap, cap, cap])
    request.deadline = time.monotonic() + 0.01
    gateway_module.Gateway._pause(request)
    assert slept[-1] <= 0.01


# -------------------------------------------------------------- R6.10: a slow worker


@pytest.mark.parametrize("stage", [0, 1], ids=["stage0", "stage1"])
def test_a_worker_slower_than_the_step_timeout_fails_requests_at_their_deadline_within_bounds(
    system,  # noqa: F811
    stage,
):
    """One stage takes 0.5 s a step against a step timeout of 0.2 s. Three running slots and a
    queue of two take five requests; three more are refused at once. Every admitted request
    fails with DEADLINE_EXCEEDED at its deadline, none waits past it, and the queue, the
    steps in flight and each inbox stay within their bounds. Afterwards nothing is left."""
    run = system(running=3, max_queued=2, credits=2, inbox_size=2)
    getattr(run, f"servicer{stage}").worker.delay = StepDelay(0.5)
    run.gateway.detector.step_timeout_ns = 200_000_000
    timeout_s = 1.5
    out, _ = run_all(run, 8, start_gap=0.02, max_tokens=20, timeout_ms=round(timeout_s * 1000))
    codes = [got for got, _ in out]
    refused = [took for got, took in out if got == grpc.StatusCode.RESOURCE_EXHAUSTED]
    late = [took for got, took in out if got == grpc.StatusCode.DEADLINE_EXCEEDED]
    assert len(refused) == 3 and len(late) == 5, codes
    assert all(took < 0.5 for took in refused)
    assert all(timeout_s - 0.1 <= took < timeout_s + TOLERANCE for took in late)
    assert run.gateway.queue_peak <= 2 and run.gateway.in_flight_peak <= 2
    assert run.servicer0.inbox_peak <= 2 and run.servicer1.inbox_peak <= 2
    late_steps = [e for e in run.gateway.detector.events() if e.get("path") == "step_deadline"]
    assert late_steps and not any(e["declared"] for e in late_steps)  # slow, not dead
    wait_for(lambda: not run.gateway.requests, "the failed requests to leave the table")
    assert not run.gateway._waiting and run.gateway._running == 0
    wait_for(lambda: run.gateway._in_flight == 0, "every credit to come back")
