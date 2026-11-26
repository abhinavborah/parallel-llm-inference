"""Cross-test of failure detection (R6.3, R6.4), written from the requirements and the
README's Failure detection section by the member who did not build it.

- A broken heartbeat stream declares the epoch dead, once, naming the path.
- A stage whose stream stays open but stops beating is declared dead only after the
  heartbeat timeout, never before.
- After a declaration the gateway waits for a new Register: nothing else changes, requests
  are refused, and the new registration gets a higher epoch and its neighbour a route update.
- A step that outlives the step timeout is recorded, but declares nothing.
- A stage whose epoch was declared dead and has no replacement ends its worker.
"""

import threading
import time

import grpc
import pytest
from test_detection import start_replacement, stop_stage
from test_gateway import IDS, code_of, wait_for
from test_heartbeat import Beating, reference
from test_model import tiny_reference

from pllm.control import ExpectedModel
from pllm.faults import StepDelay
from pllm.models import sha256_of
from pllm.worker import join_gateway

LAYERS = 4
K = 2
BEAT = 0.05
SECOND = 1_000_000_000


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


def detections(run, epoch=None, path=None):
    return [
        e
        for e in run.gateway.detector.events()
        if e["event"] == "detected"
        and (epoch is None or e["epoch"] == epoch)
        and (path is None or e["path"] == path)
    ]


# ------------------------------------------------------------------ the two ways to declare


@pytest.mark.parametrize("stage", [0, 1])
def test_a_broken_heartbeat_stream_declares_that_epoch_dead_once(beating, stage):
    run = beating()
    epoch = run.registry.stages[stage].epoch
    other = run.registry.stages[1 - stage].epoch
    stop_stage(run, stage)  # its heartbeat stream ends
    wait_for(lambda: run.gateway.detector.is_dead(epoch), "the stream break to be declared")
    time.sleep(0.3)  # the silence that follows must not declare it a second time
    found = detections(run, epoch)
    assert len(found) == 1
    record = found[0]
    assert (record["path"], record["stage"], record["declared"]) == ("stream_break", stage, True)
    assert isinstance(record["t_ns"], int) and record["since_last_beat_ns"] is not None
    assert not run.gateway.detector.is_dead(other)  # the other stage is not blamed
    # the first declaration wins and writes the only record; a dead epoch is never revived
    assert run.gateway.detector.declare(stage, epoch, "heartbeat_timeout") is False
    assert run.gateway.detector.beat(stage, epoch) is False
    assert len(detections(run, epoch)) == 1 and run.gateway.detector.is_dead(epoch)


@pytest.mark.parametrize("stage", [0, 1])
def test_a_stage_that_stops_beating_is_declared_dead_only_after_the_timeout(beating, stage):
    """The stream stays open and the process lives, but no Beat comes: the heartbeat
    timeout is the only thing that can notice."""
    run = beating()
    run.gateway.detector.heartbeat_timeout_ns = SECOND
    epoch = run.registry.stages[stage].epoch
    other = run.registry.stages[1 - stage].epoch
    run.hearts[stage].interval = 60  # one more Beat at most, then silence on an open stream
    time.sleep(0.4)
    assert not run.gateway.detector.is_dead(epoch)  # not before the timeout
    wait_for(lambda: run.gateway.detector.is_dead(epoch), "the silence to be declared", timeout=6)
    record = detections(run, epoch)[0]
    assert (record["path"], record["stage"], record["declared"]) == (
        "heartbeat_timeout",
        stage,
        True,
    )
    assert SECOND <= record["since_last_beat_ns"] < 2 * SECOND
    assert len(detections(run, epoch)) == 1
    assert not run.gateway.detector.is_dead(other)  # which still beats


# ------------------------------------------------------------------ after the declaration


@pytest.mark.parametrize("stage", [0, 1])
def test_a_declaration_waits_for_a_new_registration_and_then_routes_the_neighbour(
    beating, tmp_path, stage
):
    run = beating()
    expected = reference(tmp_path, K, IDS, 4)[0]
    old = run.registry.stages[stage].epoch
    stop_stage(run, stage)
    wait_for(lambda: run.gateway.detector.is_dead(old), "the declaration")

    # waiting: a request is refused at once, and nothing registers or changes by itself
    began = time.monotonic()
    code, details = code_of(run.call(max_tokens=4))
    assert code == grpc.StatusCode.UNAVAILABLE and f"epoch {old}" in details
    assert time.monotonic() - began < 2
    time.sleep(0.3)
    assert run.registry.stages[stage].epoch == old
    assert not [
        e
        for e in run.gateway.detector.events()
        if e["event"] == "registered" and e["old_epoch"] == old
    ]

    servicer, new = start_replacement(run, stage)
    assert new > old and run.registry.stages[stage].epoch == new
    record = [e for e in run.gateway.detector.events() if e["event"] == "registered"][-1]
    assert (record["stage"], record["old_epoch"], record["new_epoch"]) == (stage, old, new)
    assert record["waited_ns"] > 0  # measured from the declaration
    if stage == 1:  # the survivor learns the new epoch (the route update)
        wait_for(lambda: run.servicer0.next_epoch == new, "Stage 0 to be routed to the new Stage 1")
    else:
        wait_for(lambda: run.servicer1.upstream_epoch == new, "Stage 1 to be told the new Stage 0")
    assert record["route_queued_ns"] is not None
    other = run.servicer1 if stage == 0 else run.servicer0
    run.wait_routes(**({"servicer1": servicer} if stage == 1 else {"servicer0": servicer}))
    assert other is not servicer
    assert [t.token for t in run.call(max_tokens=4)] == expected  # and it serves again


# ------------------------------------------------------------------ a slow step is not a death


def test_a_step_past_the_step_timeout_is_recorded_but_declares_nothing(beating, tmp_path):
    run = beating()
    run.gateway.detector.step_timeout_ns = round(0.3 * SECOND)
    run.servicer0.worker.delay = StepDelay(0.8)  # every step takes longer than the timeout
    expected = reference(tmp_path, K, IDS, 2)[0]
    stream = run.call(max_tokens=2)
    tokens = list(stream)
    assert [t.token for t in tokens] == expected  # the slow stage still answers
    late = detections(run, path="step_deadline")
    assert late and all(e["declared"] is False for e in late)
    assert {e["rid"] for e in late} == {tokens[0].rid}
    assert not [e for e in detections(run) if e["declared"]]  # nobody is declared dead
    assert not any(run.gateway.detector.is_dead(s.epoch) for s in run.registry.stages.values())


# ------------------------------------------------------------------ the dead stage leaves


def test_a_stage_declared_dead_with_no_replacement_ends_its_worker(beating, tmp_path):
    """Nothing would restart a process that is still alive, so a stage told its epoch is dead
    must leave itself: the worker's join call raises, and its supervisor starts a new one."""
    run = beating(order=(0,))  # Stage 1 joins through the worker's own entry point below
    run.registry.expected = ExpectedModel(
        bytes.fromhex(sha256_of(tmp_path / "model.safetensors")), LAYERS, K
    )
    run.gateway.detector.heartbeat_timeout_ns = round(0.6 * SECOND)
    stop = threading.Event()
    outcome = []

    def worker():
        try:
            join_gateway(
                run.address,
                1,
                run.layers[1],
                "tiny",
                tmp_path,
                f"127.0.0.1:{run.ports[1]}",
                run.servicer1,
                stop,
                interval=1.5,  # slower than the timeout, so the gateway declares it first
            )
            outcome.append("returned")
        except RuntimeError as err:
            outcome.append(str(err))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        wait_for(lambda: 1 in run.registry.stages, "Stage 1 to register")
        epoch = run.registry.stages[1].epoch
        wait_for(lambda: run.gateway.detector.is_dead(epoch), "the declaration", timeout=10)
        assert thread.is_alive()  # told only when it is next heard from
        thread.join(10)
        assert not thread.is_alive()
        assert len(outcome) == 1 and "declared" in outcome[0] and str(epoch) in outcome[0]
    finally:
        stop.set()
        thread.join(5)
