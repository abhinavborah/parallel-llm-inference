"""Stage fencing and the Nacks it raises (R3.11, R3.12).

One test per rule: a wrong target epoch, a sender epoch that is not the live upstream, an
older attempt, and adopting a newer attempt. Each sends steps straight to a real stage
server, registered with the gateway as a worker is, and reads the Nack it raises. The
last tests run the heartbeat and check the gateway learns of a refusal at once.
"""

import queue
import time

import grpc
import pytest
import torch
from test_gateway import HIDDEN, WAIT, System, code_of, wait_for
from test_heartbeat import Beating
from test_model import tiny_reference

from pllm import wire
from pllm.proto import protocol_pb2, protocol_pb2_grpc

IDS = [5, 9, 100]
LAYERS = 4


def _maker(tmp_path, cls):
    made = []

    def make(k=2, **kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        made.append(cls(tmp_path, k, **kwargs))
        return made[-1]

    return make, made


@pytest.fixture
def system(tmp_path):
    """A gateway and two registered stages, epochs set as a worker sets them."""
    make, made = _maker(tmp_path, System)
    yield make
    for each in made:
        each.close()


@pytest.fixture
def beating(tmp_path):
    """The same with each stage's heartbeat stream open."""
    make, made = _maker(tmp_path, Beating)
    yield make
    for each in made:
        each.close()


def send(port, *steps):
    """Send steps on one stream to the stage on port; return once the stream ends."""
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        protocol_pb2_grpc.StageStub(channel).Steps(iter(steps), timeout=WAIT)


def prefill(run, rid, attempt=1, start_pos=0, ids=IDS, **epochs):
    """A Stage 0 step with the right epochs unless epochs overrides them."""
    fields = {"target_epoch": run.servicer0.own_epoch, "sender_epoch": 0, **epochs}
    return protocol_pb2.StepRequest(
        rid=rid,
        attempt=attempt,
        start_pos=start_pos,
        length=len(ids),
        tokens=protocol_pb2.TokenIds(ids=ids),
        **fields,
    )


def activations(run, rid, **epochs):
    """A Stage 1 step with the right epochs unless epochs overrides them."""
    fields = {
        "target_epoch": run.servicer1.own_epoch,
        "sender_epoch": run.servicer1.upstream_epoch,
        **epochs,
    }
    return protocol_pb2.StepRequest(
        rid=rid,
        attempt=1,
        start_pos=0,
        length=2,
        activations=wire.tensor_to_proto(torch.zeros(2, HIDDEN)),
        **fields,
    )


def nack(servicer):
    return servicer.nacks.get(timeout=WAIT)


def no_nack(servicer, seconds=0.3):
    try:
        servicer.nacks.get(timeout=seconds)
    except queue.Empty:
        return True
    return False


# -------------------------------------------------------------- rule 1: WRONG_EPOCH


def test_a_step_for_another_epoch_of_stage0_is_refused(system):
    run = system()
    epoch = run.servicer0.own_epoch
    send(run.ports[0], prefill(run, "old", target_epoch=epoch - 1))
    refused = nack(run.servicer0)
    assert (refused.reason, refused.stage, refused.epoch, refused.rid) == (
        "WRONG_EPOCH",
        0,
        epoch,
        "old",
    )
    assert "old" not in run.servicer0.store and "old" not in run.servicer1.store  # not forwarded
    send(run.ports[0], prefill(run, "new"))
    wait_for(lambda: "new" in run.servicer1.store, "the right epoch to be served")
    assert no_nack(run.servicer0)


def test_a_step_for_another_epoch_of_stage1_is_refused(system):
    run = system()
    send(run.ports[1], activations(run, "old", target_epoch=run.servicer1.own_epoch + 5))
    refused = nack(run.servicer1)
    assert (refused.reason, refused.stage) == ("WRONG_EPOCH", 1)
    assert "old" not in run.servicer1.store


# -------------------------------------------------------------- rule 1b: BAD_SENDER


def test_stage0_accepts_only_the_gateway_as_sender(system):
    run = system()
    send(run.ports[0], prefill(run, "x", sender_epoch=run.servicer1.own_epoch))
    assert nack(run.servicer0).reason == "BAD_SENDER"
    assert "x" not in run.servicer0.store


def test_stage1_refuses_a_resumed_old_stage0(system):
    """Stage 0 registers again (a replacement); the old process, resumed, still sends with
    its old epoch, and Stage 1 refuses it while taking the new one."""
    run = system()
    old = run.servicer0.own_epoch
    new = run.register(0).epoch
    assert run.servicer1.upstream_epoch == new != old
    send(run.ports[1], activations(run, "stale", sender_epoch=old))
    refused = nack(run.servicer1)
    assert (refused.reason, refused.rid) == ("BAD_SENDER", "stale")
    assert "stale" not in run.servicer1.store
    send(run.ports[1], activations(run, "fresh", sender_epoch=new))
    assert "fresh" in run.servicer1.store and no_nack(run.servicer1)


def test_wrong_epoch_is_checked_before_the_sender(system):
    run = system()
    send(run.ports[0], prefill(run, "both", target_epoch=99, sender_epoch=98))
    assert nack(run.servicer0).reason == "WRONG_EPOCH"


def test_an_older_upstream_epoch_in_a_late_route_update_is_ignored(system):
    run = system()
    current = run.servicer1.upstream_epoch
    late = protocol_pb2.RouteUpdate(next_stage=0, addr="127.0.0.1:1", epoch=current - 1)
    assert run.servicer1.route(late) is False and run.servicer1.upstream_epoch == current
    # Stage 0 has no upstream stage: an update naming a stage before it is ignored
    assert (
        run.servicer0.route(protocol_pb2.RouteUpdate(next_stage=-1, addr="x:1", epoch=9)) is False
    )
    assert run.servicer0.upstream_epoch == 0


# -------------------------------------------------------------- rule 2: attempts


def test_an_older_attempt_is_refused(system):
    run = system()
    send(run.ports[0], prefill(run, "r", attempt=2))
    length = run.servicer0.store.length("r")
    send(run.ports[0], prefill(run, "r", attempt=1, start_pos=length, ids=[7]))
    refused = nack(run.servicer0)
    assert (refused.reason, refused.attempt) == ("STALE_ATTEMPT", 1)
    assert run.servicer0.store.length("r") == length  # untouched


def test_a_newer_attempt_is_adopted(system):
    run = system()
    send(
        run.ports[0],
        prefill(run, "r", attempt=1),
        prefill(run, "r", attempt=1, start_pos=3, ids=[7]),
    )
    assert run.servicer0.store.length("r") == 4
    send(run.ports[0], prefill(run, "r", attempt=2))  # a replay from the start
    assert run.servicer0.store.attempt("r") == 2 and run.servicer0.store.length("r") == 3
    assert no_nack(run.servicer0)
    send(run.ports[0], prefill(run, "r", attempt=1, start_pos=3, ids=[7]))  # the old one, late
    assert nack(run.servicer0).reason == "STALE_ATTEMPT"
    assert run.servicer0.store.length("r") == 3


# -------------------------------------------------------------- Nacks reach the gateway


def test_a_refused_step_fails_its_request_at_once(beating):
    """R3.12: the gateway learns of the refusal from the heartbeat stream, not from a
    timeout. The beat interval and the request's deadline are both far longer."""
    run = beating(interval=5.0)
    run.servicer0.next_epoch += 7  # Stage 0 now aims at an epoch of Stage 1 that is gone
    began = time.monotonic()
    code, details = code_of(run.call(timeout_ms=30_000))
    assert time.monotonic() - began < 2.0
    assert code == grpc.StatusCode.UNAVAILABLE
    assert "stage 1 refused step 0" in details and "WRONG_EPOCH" in details
    assert not run.gateway.requests
    assert run.gateway.nacks[-1].reason == "WRONG_EPOCH"


def test_a_nack_for_an_ended_request_changes_nothing(beating):
    run = beating()
    tokens = run.tokens(max_tokens=2)
    late = protocol_pb2.StageMsg(
        nack=protocol_pb2.Nack(stage=1, rid=tokens[0].rid, attempt=1, reason="STALE_ATTEMPT")
    )
    run.gateway._record(run.gateway._sessions[1], late)
    assert run.gateway.nack_count >= 1 and not run.gateway.requests
    assert len(run.tokens(max_tokens=2)) == 2


def test_a_nack_for_another_attempt_does_not_fail_the_request(beating):
    run = beating(delay1=0.3)
    call = run.call(max_tokens=3)
    first = next(call)
    late = protocol_pb2.StageMsg(
        nack=protocol_pb2.Nack(stage=1, rid=first.rid, attempt=7, reason="STALE_ATTEMPT")
    )
    run.gateway._record(run.gateway._sessions[1], late)
    assert len([first, *call]) == 3


def test_stage1_learns_stage0s_new_epoch_from_the_heartbeat(beating):
    run = beating(interval=5.0)
    old = run.servicer0.own_epoch
    new = run.join(0, run.servicer0)  # Stage 0 registers again
    wait_for(lambda: run.servicer1.upstream_epoch == new, "Stage 1's route update")
    wait_for(lambda: run.servicer0.next_epoch == run.servicer1.own_epoch, "Stage 0's route")
    assert len(run.tokens(max_tokens=2)) == 2
    # the old Stage 0, resumed, sends with its old epoch: refused, and the gateway hears of it
    before = run.gateway.nack_count
    send(run.ports[1], activations(run, "stale", sender_epoch=old))
    wait_for(lambda: run.gateway.nack_count == before + 1, "the Nack at the gateway", 2.0)
    assert run.gateway.nacks[-1].reason == "BAD_SENDER" and "stale" not in run.servicer1.store


def test_a_request_refused_before_its_route_arrived_can_be_sent_again(beating):
    """A request that reaches Stage 0 before the route update for a restarted Stage 1 is
    refused; once the update arrives, the stream that carried the stale epoch ends and the
    next request goes through, without restarting anything."""
    run = beating(interval=5.0)
    real = run.servicer0.next_epoch
    run.servicer0.next_epoch = real - 1  # as if the update had not arrived yet
    code, details = code_of(run.call(timeout_ms=30_000))
    assert code == grpc.StatusCode.UNAVAILABLE and "WRONG_EPOCH" in details
    update = protocol_pb2.RouteUpdate(next_stage=1, addr=run.servicer0.next_address, epoch=real)
    assert run.servicer0.route(update) is True
    wait_for(lambda: run.gateway._link.failure() is not None, "the stale stream to end")
    assert len(run.tokens(max_tokens=2)) == 2


def test_a_nack_for_another_step_does_not_fail_the_request(beating):
    """A resumed old stage can send a late step of the current attempt; the Nack it draws
    is not for the step the request waits on, so the request goes on."""
    run = beating(delay1=0.3)
    call = run.call(max_tokens=3)
    first = next(call)
    late = protocol_pb2.StageMsg(
        nack=protocol_pb2.Nack(stage=1, rid=first.rid, attempt=1, step=0, reason="BAD_SENDER")
    )
    run.gateway._record(run.gateway._sessions[1], late)
    assert len([first, *call]) == 3


def test_a_nack_queued_for_a_step_already_done_is_dropped(beating):
    """The gateway decides on a Nack under its lock but queues it after; if the step's
    result is taken first, the Nack arrives for a step already done and must not fail the
    next one."""
    run = beating(delay1=0.3)
    call = run.call(max_tokens=3)
    first = next(call)  # step 0 is done; the request now waits on step 1
    request = run.gateway.requests[first.rid]
    before = run.gateway.dropped
    request.results.put(protocol_pb2.Nack(stage=1, rid=first.rid, attempt=1, step=0, reason="GAP"))
    assert len([first, *call]) == 3
    assert run.gateway.dropped == before + 1
