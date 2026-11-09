"""The result filter and exactly-once delivery at the gateway (R4.7, R4.8).

Each test runs a real gateway and two real stage servers on loopback and injects results
into the gateway's Results.Push the way a stage sends them, while a request is running or
after it ended. Equality with the request run alone is on one host, fp32, batch 1.
"""

import collections

import grpc
import pytest
from test_gateway import WAIT, System, wait_for
from test_model import tiny_reference

from pllm.gateway import Request
from pllm.proto import protocol_pb2, protocol_pb2_grpc

LAYERS = 4
FAKE = 999  # a token the tiny model never produces in these tests


@pytest.fixture
def system(tmp_path):
    made = []

    def make(**kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        made.append(System(tmp_path, 2, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def push(run, *results):
    """Send results to the gateway on a Results.Push stream, as Stage 1 does."""
    with grpc.insecure_channel(run.address) as channel:
        protocol_pb2_grpc.ResultsStub(channel).Push(iter(results), timeout=WAIT)


def live_epoch(run):
    return run.registry.stages[1].epoch


def result(rid, step, token=FAKE, attempt=1, epoch=None, run=None):
    return protocol_pb2.StepResult(
        rid=rid,
        attempt=attempt,
        step=step,
        epoch=live_epoch(run) if epoch is None else epoch,
        token=token,
        top_logit=1.0,
    )


def alone(system, n=4):
    return [(t.token, t.top_logit) for t in system().tokens(max_tokens=n)]


# -------------------------------------------------------------- accepted only when all match


def test_each_kind_of_mismatch_is_dropped_and_counted_by_its_reason(system):
    """While a request waits on step 1, results that differ in exactly one way are each
    dropped for that reason; none of the fake tokens reaches the client."""
    expected = alone(system)
    run = system(delay1=0.3)
    stream = run.call(max_tokens=4)
    first = next(stream)  # step 0 is done; the request now waits on step 1
    rid, live = first.rid, live_epoch(run)
    push(
        run,
        result("nobody", 1, run=run),  # unknown_request
        result(rid, 1, attempt=2, run=run),  # stale_attempt
        result(rid, 1, epoch=live - 1, run=run),  # stale_epoch: an old last stage
        result(rid, 0, run=run),  # duplicate: step 0 is already done
        result(rid, 10, run=run),  # ahead: step 10 has not been asked for
    )
    got = [(first.token, first.top_logit)] + [(t.token, t.top_logit) for t in stream]
    assert got == expected and FAKE not in [token for token, _ in got]
    assert run.gateway.discards == collections.Counter(
        unknown_request=1, stale_attempt=1, stale_epoch=1, duplicate=1, ahead=1
    )
    assert run.gateway.dropped == 5


def test_a_result_from_an_old_last_stage_is_dropped_even_for_the_awaited_step(system):
    """A resumed old Stage 1 can send the very step the request waits on; its epoch is not
    the live one, so it is dropped and the real result is the one the client sees."""
    expected = alone(system)
    run = system(delay1=0.3)
    stream = run.call(max_tokens=4)
    first = next(stream)
    push(run, result(first.rid, 1, epoch=live_epoch(run) + 7, run=run))
    got = [(first.token, first.top_logit)] + [(t.token, t.top_logit) for t in stream]
    assert got == expected
    assert run.gateway.discards["stale_epoch"] == 1


# -------------------------------------------------------------- duplicates: once to the client


def test_every_result_sent_twice_still_reaches_the_client_once(system):
    """Stage 1 is made to push each result twice; the client sees each step once, in
    order, and the second copies are counted as duplicates."""
    expected = alone(system, 6)
    run = system()
    real = run.servicer1._handle

    def twice(request, context, *rest):
        out = real(request, context, *rest)
        if out is not None:
            run.gateway.Push(iter([out]), None)  # the copy, as a stage resending
        return out

    run.servicer1._handle = twice
    tokens = run.tokens(max_tokens=6)
    assert [(t.token, t.top_logit) for t in tokens] == expected
    assert [t.step for t in tokens] == list(range(6))
    # One of each pair is dropped: as a duplicate, or as a late result if the request has
    # ended. The second of the last pair can arrive after the client's stream ends.
    wait_for(lambda: run.gateway.dropped == 6, "the second copy of every step to be dropped")
    assert run.gateway.discards["duplicate"] + run.gateway.discards["unknown_request"] == 6


def test_an_injected_copy_of_a_delivered_step_is_dropped(system):
    expected = alone(system)
    run = system(delay1=0.3)
    stream = run.call(max_tokens=4)
    first = next(stream)
    copy = result(first.rid, 0, first.token, run=run)
    push(run, copy, copy)
    got = [(first.token, first.top_logit)] + [(t.token, t.top_logit) for t in stream]
    assert got == expected
    assert run.gateway.discards["duplicate"] == 2


# -------------------------------------------------------------- late results


def test_a_late_result_after_the_request_ended_changes_nothing(system):
    run = system()
    done = run.tokens(max_tokens=3)
    push(run, result(done[0].rid, 3, run=run), result(done[0].rid, 1, run=run))
    assert run.gateway.discards == collections.Counter(unknown_request=2)
    assert not run.gateway.requests
    again = run.tokens(max_tokens=3)  # the next request is not affected
    assert [t.token for t in again] == [t.token for t in done]


def test_a_late_result_queued_before_its_step_finished_is_dropped_when_taken(system):
    """The check runs again when the request takes a result: one queued just before its
    step completed is a duplicate by then."""
    expected = alone(system)
    run = system(delay1=0.3)
    stream = run.call(max_tokens=4)
    first = next(stream)
    request = run.gateway.requests[first.rid]
    request.results.put(result(first.rid, 0, run=run))
    got = [(first.token, first.top_logit)] + [(t.token, t.top_logit) for t in stream]
    assert got == expected
    assert run.gateway.discards["duplicate"] == 1


def test_results_are_filtered_on_arrival_before_anything_is_queued(system):
    """The filter runs in Results.Push: a mismatched result is never queued for its
    request, and only the matching one is. No decode loop is waiting here, so the queue
    shows exactly what arrived."""
    run = system()
    request = Request(rid="r", attempt=2, deadline=float("inf"), prompt_len=3)
    request.expected_step = 4
    run.gateway.requests["r"] = request
    try:
        live = live_epoch(run)
        push(
            run,
            result("r", 4, attempt=1, run=run),
            result("r", 4, attempt=2, epoch=live + 1, run=run),
            result("r", 3, attempt=2, run=run),
            result("r", 5, attempt=2, run=run),
        )
        assert request.results.empty()
        assert run.gateway.discards == collections.Counter(
            stale_attempt=1, stale_epoch=1, duplicate=1, ahead=1
        )
        push(run, result("r", 4, token=7, attempt=2, run=run))
        taken = request.results.get_nowait()
        assert (taken.step, taken.token) == (4, 7)
    finally:
        run.gateway.requests.pop("r", None)


def test_a_queued_result_from_an_old_last_stage_is_dropped_when_taken(system):
    """The take-time check covers the epoch too: a result accepted just before a new
    Stage 1 registered is no longer from the live last stage when the request takes it."""
    expected = alone(system)
    run = system(delay1=0.3)
    stream = run.call(max_tokens=4)
    first = next(stream)
    request = run.gateway.requests[first.rid]
    request.results.put(result(first.rid, 1, epoch=live_epoch(run) + 1, run=run))
    got = [(first.token, first.top_logit)] + [(t.token, t.top_logit) for t in stream]
    assert got == expected
    assert run.gateway.discards["stale_epoch"] == 1


def test_a_nack_that_fails_no_request_is_counted(system):
    run = system()
    done = run.tokens(max_tokens=2)

    class Session:
        stage = 1

    session = Session()
    run.gateway._sessions[1] = session
    late = protocol_pb2.Nack(stage=1, rid=done[0].rid, attempt=1, step=1, reason="STALE_ATTEMPT")
    run.gateway._record(session, protocol_pb2.StageMsg(nack=late))
    assert run.gateway.discards["stale_nack"] == 1 and run.gateway.dropped == 1
