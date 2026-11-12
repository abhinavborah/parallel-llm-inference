"""Replay after a stage dies (R6.5 to R6.7).

When a stage's epoch is declared dead, the gateway bumps the attempt of every request in its
table, idle ones included, and each request replays from step 0 once a new stage has
registered: the prompt at start_pos 0, then one single-token message per token already
sent, all with the replay flag. Replayed tokens are compared with the history and never
streamed again. Equality with a fault-free run is on one host, fp32, batch 1.
"""

import logging
import subprocess
import sys
import threading
import time

import grpc
import pytest
from test_deadlines import TOLERANCE
from test_detection import start_replacement, stop_stage
from test_gateway import IDS, WAIT, code_of, wait_for
from test_heartbeat import Beating, reference
from test_model import tiny_reference

from pllm import faults
from pllm.control import ExpectedModel
from pllm.faults import StepDelay
from pllm.models import sha256_of
from pllm.stage_service import StageServicer

LAYERS = 4
K = 2
BEAT = 0.05
P = len(IDS)


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


def seen_by(servicer):
    """Record (rid, attempt, step, start_pos, length, replay) of each message the stage takes."""
    seen = []
    real = servicer._handle

    def handle(request, context, next_epoch, deadline=None):
        seen.append(
            (
                request.rid,
                request.attempt,
                request.step,
                request.start_pos,
                request.length,
                request.replay,
            )
        )
        return real(request, context, next_epoch, deadline)

    servicer._handle = handle
    return seen


def take(stream, n):
    return [next(stream) for _ in range(n)]


def replace(run, stage):
    """Stop a stage in this process, wait for its death to be declared, start a new one."""
    old = run.registry.stages[stage].epoch
    stop_stage(run, stage)
    wait_for(lambda: run.gateway.detector.is_dead(old), "the stage to be declared dead")
    return old


# -------------------------------------------------------------- R6.5: every request is bumped


def test_an_idle_request_holding_kv_is_bumped_at_the_declaration_and_both_finish(beating, tmp_path):
    """With one credit, two requests take turns: at any moment one waits for the credit,
    idle, with KV on both stages. Both attempts are bumped at the declaration, before
    either sends again, and both finish with the fault-free tokens."""
    run = beating(credits=1, delay1=0.05)
    expected = reference(tmp_path, K, IDS, 12)[0]
    streams = [run.call(max_tokens=12), run.call(max_tokens=12)]
    got = [take(stream, 3) for stream in streams]
    rids = {tokens[0].rid for tokens in got}
    stage1 = run.servicer1
    attempts = []
    real = run.gateway._on_detection

    def at_declaration(record):
        real(record)
        if record["event"] == "detected" and record["declared"]:
            attempts.append({row["rid"]: row["attempt"] for row in run.gateway.table()})

    run.gateway.detector._sink = at_declaration
    seen = seen_by(run.servicer0)  # Stage 0 survives
    replace(run, 1)
    assert attempts == [{rid: 2 for rid in rids}]  # both, at the declaration itself
    servicer, _ = start_replacement(run, 1)
    run.wait_routes(servicer1=servicer)
    for stream, tokens in zip(streams, got, strict=True):
        tokens += list(stream)
        assert [t.token for t in tokens] == expected
        assert [t.step for t in tokens] == list(range(12))  # each step streamed once
    assert stage1 is not servicer
    assert run.gateway.divergences == 0 and not run.gateway.requests
    for rid in rids:  # the first message of a new attempt is the prompt, never a decode
        new = [m for m in seen if m[0] == rid and m[1] >= 2]  # 2, or later if it restarted
        assert new[0][2:] == (0, 0, P, True)


# -------------------------------------------------------------- R6.6: the replay's messages


def test_the_replay_resends_the_original_messages_with_the_replay_flag(beating, tmp_path):
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 8)[0]
    stream = run.call(max_tokens=8)
    tokens = take(stream, 4)
    seen = seen_by(run.servicer0)  # Stage 0 survives; it sees the replay first
    replace(run, 1)
    servicer, _ = start_replacement(run, 1)
    run.wait_routes(servicer1=servicer)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    rid = tokens[0].rid
    replay = [m[1:] for m in seen if m[0] == rid and m[1] == max(a for _, a, *_ in seen)]
    sent = len(replay) and next(i for i, m in enumerate(replay) if not m[4])
    attempt = replay[0][0]
    assert attempt >= 2
    # the prompt at 0, then one token each, all flagged: the original chunk boundaries
    assert replay[:sent] == [(attempt, 0, 0, P, True)] + [
        (attempt, step, P + step - 1, 1, True) for step in range(1, sent)
    ]
    assert sent >= 4  # every token the client had when the stage died
    assert replay[sent] == (attempt, sent, P + sent - 1, 1, False)  # then normal decoding


# -------------------------------------------------------------- R6.7: verified, not resent


def test_a_replayed_token_that_differs_from_the_history_is_logged_and_not_sent(
    beating, tmp_path, caplog
):
    """The history is changed to force a mismatch: the replay logs one divergence, writes
    it to the metrics file, sends the client nothing twice, and goes on from the history."""
    run = beating(delay1=0.05)
    stream = run.call(max_tokens=8)
    tokens = take(stream, 4)
    request = run.gateway.requests[tokens[0].rid]
    true = request.tokens[2]
    request.tokens[2] = true + 1  # what the client was "sent" at step 2
    lines = []
    real = run.gateway.recorder.event
    run.gateway.recorder.event = lambda record: (lines.append(record), real(record))
    with caplog.at_level(logging.WARNING, logger="pllm.gateway"):
        replace(run, 1)
        servicer, _ = start_replacement(run, 1)
        run.wait_routes(servicer1=servicer)
        tokens += list(stream)
    assert [t.step for t in tokens] == list(range(8))
    # step 2 differs; later replayed steps take the history's token as input, so they may too
    assert run.gateway.divergences >= 1
    line = [r for r in lines if r["event"] == "divergence"][0]
    assert (line["step"], line["token"], line["history_token"]) == (2, true, true + 1)
    assert any("divergence" in r.getMessage() for r in caplog.records)


def test_a_gap_nack_replays_the_request(beating, tmp_path):
    """Protocol rule 4: a stage that has no cache at the step's position refuses it with GAP,
    and the gateway replays instead of failing the request."""
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 8)[0]
    stream = run.call(max_tokens=8)
    tokens = take(stream, 2)
    request = run.gateway.requests[tokens[0].rid]
    seen = seen_by(run.servicer0)
    run.servicer1.store.free(request.rid)  # Stage 1 loses its cache
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    assert [t.step for t in tokens] == list(range(8))
    assert run.gateway.replays == 1
    # the replay runs under a new attempt, so no late result of the old one can be taken
    replayed = [m for m in seen if m[0] == request.rid and m[2:4] == (0, 0)]
    assert replayed and all(m[1] >= 2 for m in replayed)


def test_a_replay_refused_before_the_route_update_retries_with_a_pause(
    beating, tmp_path, monkeypatch
):
    """A new Stage 1 that has not yet learned Stage 0's epoch refuses the replay with
    BAD_SENDER. The request starts again under a newer attempt each time, 0.05 s apart,
    and finishes unchanged once the route update lands."""
    hold = 0.5
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 8)[0]
    stream = run.call(max_tokens=8)
    tokens = take(stream, 2)
    first = {run.servicer0, run.servicer1}
    real_route, real_nack = StageServicer.route, StageServicer._nack
    nacks = []

    def late(self, update):
        if self not in first and update.next_stage == 0:  # the new stage's upstream epoch
            threading.Timer(hold, real_route, (self, update)).start()
            return True
        return real_route(self, update)

    def counted(self, request, reason):
        nacks.append(reason)
        return real_nack(self, request, reason)

    # set before the new stage registers, so its first route update is held back
    monkeypatch.setattr(StageServicer, "route", late)
    monkeypatch.setattr(StageServicer, "_nack", counted)
    replace(run, 1)
    start_replacement(run, 1)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    assert [t.step for t in tokens] == list(range(8))
    assert "BAD_SENDER" in nacks  # the refused branch was reached
    # without the pause this is one replay per round trip; with it, about hold / 0.05
    assert run.gateway.replays <= hold / 0.05 + 5


@pytest.mark.parametrize("timeout_ms", [20_000, 800], ids=["long", "short"])
def test_an_error_inside_a_live_stage_is_not_replayed(beating, monkeypatch, timeout_ms):
    """A step that fails inside Stage 0 ends its stream, but the stage still beats, so no
    death is declared: the request fails with UNAVAILABLE after the heartbeat timeout, or
    at its own deadline if that comes first, and is never replayed."""
    run = beating(delay1=0.05)
    stream = run.call(max_tokens=50, timeout_ms=timeout_ms)
    take(stream, 2)

    def broken(inputs, start_pos, cache):
        raise RuntimeError("a fault inside the stage")

    began = time.monotonic()
    monkeypatch.setattr(run.servicer0.worker, "step", broken)
    code, _ = code_of(stream)
    took = time.monotonic() - began
    grace = run.gateway.detector.heartbeat_timeout_ns / 1e9
    if timeout_ms > grace * 1000:
        assert code == grpc.StatusCode.UNAVAILABLE
        assert grace - 0.1 <= took < grace + TOLERANCE
    else:
        assert code == grpc.StatusCode.DEADLINE_EXCEEDED
        assert took < timeout_ms / 1000 + TOLERANCE
    assert run.gateway.replays == 0 and not run.gateway.requests
    assert run.gateway.detector.events()[-1]["event"] == "registered"  # nothing declared


def test_with_no_replacement_the_request_fails_at_its_deadline(beating):
    run = beating(delay1=0.05)
    stream = run.call(max_tokens=50, timeout_ms=1500)
    take(stream, 2)
    began = time.monotonic()
    replace(run, 1)
    code, details = code_of(stream)
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED and "waiting to replay" in details
    assert time.monotonic() - began < 1.5 + TOLERANCE
    assert not run.gateway.requests


# -------------------------------------------------------------- a real process killed

CHILD = """
import sys, threading
from pathlib import Path
from pllm.stage_service import serve
from pllm.worker import StageWorker, WorkerConfig, join_gateway
model_dir, stage, k, n, gateway, neighbour = sys.argv[1:7]
stage, k, n = int(stage), int(k), int(n)
layers = (0, k) if stage == 0 else (k, n)
where = {"next_address": neighbour} if stage == 0 else {"results_address": gateway}
server, servicer, port = serve(StageWorker(WorkerConfig(Path(model_dir), *layers)),
                               "127.0.0.1:0", **where)
join_gateway(gateway, stage, layers, "tiny", Path(model_dir), f"127.0.0.1:{port}", servicer,
             threading.Event(), interval=float(sys.argv[7]))
"""


@pytest.mark.parametrize("stage", [0, 1])
def test_a_killed_stage_process_is_replaced_and_the_request_finishes_unchanged(
    beating, tmp_path, stage
):
    """A real stage process is killed mid-decode and a new process takes its place, as its
    supervisor would start one. The client gets the fault-free tokens, each step once."""
    run = beating(order=(1 - stage,), delay1=0.05)  # the other stage stays in this process
    n = LAYERS
    neighbour = "" if stage == 1 else f"127.0.0.1:{run.ports[1]}"
    args = [str(tmp_path), str(stage), str(K), str(n), run.address, neighbour, str(BEAT)]

    # the child registers the real weights' digest, not the stand-in the harness used
    real = bytes.fromhex(sha256_of(tmp_path / "model.safetensors"))
    run.registry.expected = ExpectedModel(real, LAYERS, K)

    def spawn():
        return subprocess.Popen([sys.executable, "-c", CHILD, *args])

    children = [spawn()]
    try:
        first = run.registry.stages
        wait_for(lambda: stage in first and stage in run.gateway.beats(), "the child's beat")
        if stage == 1:  # this process's Stage 0 sends to the child once routed
            wait_for(lambda: run.servicer0.next_epoch == first[1].epoch, "the route update")
        else:
            wait_for(lambda: run.servicer1.upstream_epoch == first[0].epoch, "the route update")
        expected = reference(tmp_path, K, IDS, 16)[0]
        stream = run.call(max_tokens=16)
        tokens = take(stream, 5)
        old = run.registry.stages[stage].epoch
        faults.kill(children[0].pid)
        children[0].wait(WAIT)
        wait_for(lambda: run.gateway.detector.is_dead(old), "the stage to be declared dead")
        children.append(spawn())
        tokens += list(stream)
        assert [t.token for t in tokens] == expected
        assert [t.step for t in tokens] == list(range(16))
        assert run.gateway.replays >= 1 and run.gateway.divergences == 0
        assert run.registry.stages[stage].epoch > old
    finally:
        for child in children:
            if child.poll() is None:
                faults.kill(child.pid)
            child.wait(WAIT)


def test_a_stage_with_a_step_delay_still_matches_after_replay(beating, tmp_path):
    """A slow surviving stage does not change the replayed tokens."""
    run = beating(delay1=0.02)
    run.servicer0.worker.delay = StepDelay(0.03)
    expected = reference(tmp_path, K, IDS, 10)[0]
    stream = run.call(max_tokens=10)
    tokens = take(stream, 3)
    replace(run, 1)
    servicer, _ = start_replacement(run, 1)
    run.wait_routes(servicer1=servicer)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
