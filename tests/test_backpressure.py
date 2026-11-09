"""Bounded admission queue, credits and stage inboxes (R5.1, R5.2).

Each test runs a real gateway and two real stage servers on loopback. Equality with the
request run alone is on one host, fp32, batch 1.
"""

import threading
import time

import grpc
import pytest
from test_gateway import WAIT, System, wait_for
from test_model import tiny_reference

from pllm import config, gateway
from pllm.proto import protocol_pb2, protocol_pb2_grpc

LAYERS = 4
K = 2
FAST = 0.5  # seconds within which a refusal must come back


@pytest.fixture
def system(tmp_path):
    made = []

    def make(running=None, **kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        if running is not None:
            kwargs["max_workers"] = gateway.RESERVED_THREADS + running
        made.append(System(tmp_path, K, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def alone(system, n):
    return [(t.token, t.top_logit) for t in system().tokens(max_tokens=n)]


def run_all(run, count, start_gap=0.0, **kwargs):
    """Start count requests (start_gap seconds apart) and return, per request, its tokens
    or its status code with the seconds it took and the order it ended in."""
    out = [None] * count
    order = []
    lock = threading.Lock()

    def one(i):
        began = time.monotonic()
        try:
            got = [(t.token, t.top_logit) for t in run.call(**kwargs)]
        except grpc.RpcError as err:
            got = err.code()
        with lock:
            out[i] = (got, time.monotonic() - began)
            order.append(i)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
        time.sleep(start_gap)
    for thread in threads:
        thread.join(WAIT)
    return out, order


# -------------------------------------------------------------- overload


def test_under_overload_the_excess_is_refused_at_once_and_every_bound_holds(system):
    """3 running slots, a queue of 2, 2 credits and stage inboxes of 2, with Stage 1 slowed.
    Five requests fill the slots and the queue, then five more arrive together: those five
    are refused at once with RESOURCE_EXHAUSTED; the 5 admitted give
    the run-alone tokens; the queue, the steps in flight and each stage's inbox never go
    past their bounds."""
    expected = alone(system, 3)
    run = system(running=3, max_queued=2, credits=2, inbox_size=2, delay1=0.3)
    results = []
    first = threading.Thread(
        target=lambda: results.append(run_all(run, 5, max_tokens=3, timeout_ms=20_000)[0])
    )
    first.start()  # five fill the three slots and the queue of two
    wait_for(lambda: len(run.gateway._waiting) == 2, "three running and two queued")
    excess, _ = run_all(run, 5, max_tokens=3, timeout_ms=20_000)  # then five more
    first.join(WAIT)
    out = results[0] + excess
    refused = [took for got, took in out if got == grpc.StatusCode.RESOURCE_EXHAUSTED]
    done = [got for got, _ in out if isinstance(got, list)]
    assert len(refused) == 5 and len(done) == 5
    assert all(took < FAST for took in refused)  # rejected immediately, not after a wait
    assert all(got == expected for got in done)
    assert run.gateway.queue_peak == 2  # the bound was reached and never passed
    assert run.gateway.in_flight_peak == 2
    assert run.servicer0.inbox_peak <= 2 and run.servicer1.inbox_peak <= 2
    assert not run.gateway.requests and not run.gateway._waiting
    assert run.gateway._running == 0 and run.gateway._in_flight == 0


def test_without_credits_the_same_load_puts_more_steps_in_flight(system):
    """The control for the test above: with no credit limit, the three running requests
    put three steps in flight at once."""
    run = system(running=3, max_queued=2, inbox_size=2, delay1=0.15)
    run_all(run, 5, max_tokens=3, timeout_ms=20_000)
    assert run.gateway.in_flight_peak == 3


# -------------------------------------------------------------- the admission queue


def test_queued_requests_start_in_arrival_order(system):
    run = system(running=1, max_queued=3, delay1=0.2)  # each request takes about 0.6 s
    out, order = run_all(run, 4, start_gap=0.05, max_tokens=3, timeout_ms=20_000)
    assert all(isinstance(got, list) for got, _ in out)
    assert order == [0, 1, 2, 3]
    assert run.gateway.queue_peak == 3


def test_a_request_whose_deadline_passes_in_the_queue_fails_and_leaves_it(system):
    run = system(running=1, max_queued=2, delay1=0.1)
    first = run.call(max_tokens=10, timeout_ms=20_000)
    next(first)  # holds the only running slot for about a second
    began = time.monotonic()
    with pytest.raises(grpc.RpcError) as err:
        run.tokens(max_tokens=2, timeout_ms=300)
    took = time.monotonic() - began
    assert err.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "running slot" in err.value.details()
    assert 0.25 <= took < 0.3 + FAST
    assert not run.gateway._waiting
    assert len(list(first)) == 9  # the running request is unaffected
    assert len(run.tokens(max_tokens=2)) == 2  # and the slot is free again


def test_a_request_cancelled_in_the_queue_leaves_it_at_once(system):
    """The place frees while the running request still runs, so a later request queues
    instead of being refused."""
    run = system(running=1, max_queued=1, delay1=0.1)
    first = run.call(max_tokens=20, timeout_ms=20_000)  # about 2 s
    next(first)
    queued = run.call(max_tokens=2, timeout_ms=20_000)
    wait_for(lambda: len(run.gateway._waiting) == 1, "the second request to queue")
    queued.cancel()
    left = time.monotonic() + 0.5
    while run.gateway._waiting and time.monotonic() < left:
        time.sleep(0.01)
    assert not run.gateway._waiting and len(run.gateway.requests) == 1
    later = run.call(max_tokens=2, timeout_ms=20_000)  # takes the freed place
    assert len(list(first)) == 19
    assert len(list(later)) == 2
    assert run.gateway._running == 0


def test_time_in_the_queue_counts_against_the_deadline(system):
    """Queued about 0.6 s, then 0.6 s of steps: a 0.9 s deadline counted from arrival
    passes while the request runs, so it fails at 0.9 s instead of finishing."""
    run = system(running=1, max_queued=1, delay1=0.1)
    first = run.call(max_tokens=6, timeout_ms=20_000)
    next(first)
    began = time.monotonic()
    with pytest.raises(grpc.RpcError) as err:
        run.tokens(max_tokens=6, timeout_ms=900)
    assert err.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    assert time.monotonic() - began < 0.9 + FAST
    list(first)


# -------------------------------------------------------------- credits


def test_a_request_whose_deadline_passes_waiting_for_a_credit_fails(system):
    run = system(running=2, credits=1, delay1=0.4)
    first = run.call(max_tokens=2, timeout_ms=20_000)
    holder = threading.Thread(target=lambda: list(first))
    holder.start()
    wait_for(lambda: run.gateway._in_flight == 1, "the first request to take the credit")
    with pytest.raises(grpc.RpcError) as err:
        run.tokens(max_tokens=2, timeout_ms=150)
    holder.join(WAIT)
    assert err.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "credit" in err.value.details()
    assert run.gateway.in_flight_peak == 1 and run.gateway._in_flight == 0


# -------------------------------------------------------------- stage inboxes


def burst(run, count, length=2):
    """count prefill steps of length positions, for different requests, sent to Stage 0
    on one stream."""
    epoch0 = run.registry.stages[0].epoch
    steps = [
        protocol_pb2.StepRequest(
            rid=f"r{i}",
            attempt=1,
            step=0,
            start_pos=0,
            length=length,
            target_epoch=epoch0,
            sender_epoch=0,
            tokens=protocol_pb2.TokenIds(ids=[5, 9] * (length // 2)),
        )
        for i in range(count)
    ]
    with grpc.insecure_channel(f"127.0.0.1:{run.ports[0]}") as channel:
        protocol_pb2_grpc.StageStub(channel).Steps(iter(steps), timeout=WAIT)


def handled(servicer):
    seen = []
    real = servicer._handle

    def spy(request, context, *rest):
        seen.append(request.rid)
        return real(request, context, *rest)

    servicer._handle = spy
    return seen


def test_a_bounded_inbox_holds_back_a_burst_and_loses_nothing(system):
    """Eight messages sent at once to a slow Stage 0 with an inbox of 2: the inbox never
    holds more than 2, the sender is slowed down instead, and all eight are handled in
    order at both stages."""
    run = system(inbox_size=2, delay0=0.05, delay1=0.05)
    at1 = handled(run.servicer1)
    burst(run, 8)
    wait_for(lambda: len(at1) == 8, "Stage 1 to handle all eight")
    assert at1 == [f"r{i}" for i in range(8)]
    assert 1 <= run.servicer0.inbox_peak <= 2
    assert run.servicer1.inbox_peak <= 2


def test_an_unbounded_inbox_takes_the_whole_burst(system):
    """The control: with no bound the same burst piles up in Stage 0's inbox."""
    run = system(delay0=0.05)
    at1 = handled(run.servicer1)
    burst(run, 8)
    wait_for(lambda: len(at1) == 8, "Stage 1 to handle all eight")
    assert run.servicer0.inbox_peak > 2


# -------------------------------------------------------------- configuration


def test_the_committed_flow_limits_are_valid():
    limits = config.flow_limits()
    assert limits.stage_inbox >= limits.credits >= 1 and limits.admission_queue >= 0


@pytest.mark.parametrize(
    "lines, message",
    [
        ("admission_queue = -1\ncredits = 4\nstage_inbox = 4\n", "admission_queue"),
        ("admission_queue = 2\ncredits = 0\nstage_inbox = 4\n", "credits"),
        ("admission_queue = 2\ncredits = true\nstage_inbox = 4\n", "credits"),
        ("admission_queue = 2\ncredits = 4\n", "stage_inbox"),
        ("admission_queue = 2\ncredits = 4\nstage_inbox = 3\n", "at least credits"),
    ],
)
def test_bad_flow_limits_are_rejected(tmp_path, lines, message):
    path = tmp_path / "hosts.toml"
    path.write_text("[limits]\n" + lines)
    with pytest.raises(ValueError, match=message):
        config.flow_limits(path)


def test_an_admission_queue_of_zero_is_allowed(tmp_path):
    path = tmp_path / "hosts.toml"
    path.write_text("[limits]\nadmission_queue = 0\ncredits = 1\nstage_inbox = 1\n")
    assert config.flow_limits(path) == config.FlowLimits(0, 1, 1)


def test_a_route_change_with_a_full_outbox_still_ends_the_outbound_stream(system):
    """Fault injection: Stage 1 is slow, so Stage 0's queue to it fills; then a route update
    moves Stage 0 to a new channel at the same address. Stage 0 cannot queue the end of
    its stream, and the stream to Stage 1 must still end rather than hold one of Stage 1's
    server threads. The messages are
    500-position prefills (128 KiB each), enough to fill gRPC's flow-control window, so
    Stage 0 cannot hand them on and its queue to Stage 1 stays full."""
    run = system(inbox_size=1, delay1=0.5)
    sender = threading.Thread(target=lambda: _quietly(burst, run, 60, 500))
    sender.start()
    wait_for(lambda: run.servicer1._inboxes, "a stream from Stage 0 to Stage 1")
    wait_for(lambda: run.servicer1.inbox_depth == 1, "Stage 1's inbox to fill")
    wait_for(lambda: run.servicer0.inbox_depth == 1, "Stage 0 to be held back")
    run.servicer0.route(
        protocol_pb2.RouteUpdate(
            next_stage=1,
            addr=run.servicer0.next_address,
            epoch=run.servicer0.next_epoch + 1,
        )
    )
    sender.join(WAIT)
    assert not sender.is_alive()
    wait_for(lambda: not run.servicer1._inboxes, "the stream into Stage 1 to end")


def _quietly(fn, *args):
    try:
        fn(*args)
    except grpc.RpcError:
        pass  # the stream into Stage 0 ends with UNAVAILABLE when its route changes
