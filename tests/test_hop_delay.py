"""Hop delay for X4 (R7.7, decision #48): the gateway to Stage 0 hop is delayed at the
gateway's send, the Stage 0 to Stage 1 hop at Stage 1's receive. Pipelined and counted as
transit. Real gateway and stages on loopback; timing bounds are loose, since CI runs on
Windows and the tests share a laptop.
"""

import statistics
import threading
import time

import grpc
import pytest
from test_gateway import WAIT, System, wait_for
from test_model import tiny_reference

from pllm import faults
from pllm.faults import HopDelay

LAYERS = 4
K = 2
D = 0.03  # seconds per delayed hop in the timing tests


@pytest.fixture
def system(tmp_path):
    made = []

    def make(**kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        made.append(System(tmp_path, K, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def controllable(run):
    """Turn the hop delay on at 0 on both hops before the first request, as main does."""
    run.gateway.hop_delay = HopDelay()
    run.servicer1.hop_delay = HopDelay()
    return run


def per_token(run, tokens=12) -> float:
    """Median seconds between tokens arriving at the client."""
    times = [time.monotonic() for _ in run.call(max_tokens=tokens)]
    return statistics.median(b - a for a, b in zip(times, times[1:], strict=False))


def spy(servicer, seen):
    """Record (budget_ms, local deadline, monotonic time) of each step the stage handles."""
    real = servicer._handle

    def handle(request, context, next_epoch, deadline=None):
        seen.append((request.budget_ms, deadline, time.monotonic()))
        return real(request, context, next_epoch, deadline)

    servicer._handle = handle


# -------------------------------------------------------------- the wrapper


def test_a_hop_delay_releases_each_message_its_delay_after_it_entered():
    assert HopDelay(0.02).release(object(), 10.0) == pytest.approx(10.02)
    assert HopDelay().release(object(), 10.0) == 10.0
    for bad in (-0.001, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            HopDelay(bad)


def test_sleep_until_gives_up_when_asked():
    start = time.monotonic()
    assert not faults.sleep_until(start + 5, give_up=lambda: True, poll=0.01)
    assert time.monotonic() - start < 1
    assert faults.sleep_until(time.monotonic() + 0.02)


# -------------------------------------------------------------- on the chain


@pytest.mark.parametrize(
    "gateway_hop, stage1_hop, hops", [(True, False, 1), (False, True, 1), (True, True, 2)]
)
def test_each_delayed_hop_adds_its_delay_to_every_token(system, gateway_hop, stage1_hop, hops):
    run = controllable(system())
    base = per_token(run)
    if gateway_hop:
        run.gateway.hop_delay = HopDelay(D)
    if stage1_hop:
        run.servicer1.hop_delay = HopDelay(D)
    added = per_token(run) - base
    assert 0.8 * hops * D <= added <= hops * D + 0.04


def test_a_delay_is_pipelined_not_one_message_at_a_time(system):
    """Eight requests at once, 4 tokens each, 50 ms on both hops: one message at a time
    would take 8 x 4 x 2 x 50 ms = 3.2 s; pipelined it is about 4 x 100 ms."""
    run = controllable(system())
    run.gateway.hop_delay = run.servicer1.hop_delay = HopDelay(0.05)
    errors = []

    def one():
        try:
            assert len(run.tokens(max_tokens=4)) == 4
        except Exception as err:  # surfaced below, not lost in the thread
            errors.append(err)

    threads = [threading.Thread(target=one) for _ in range(8)]
    start = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    elapsed = time.monotonic() - start
    assert not errors
    assert 0.4 <= elapsed < 1.6


def test_the_delay_counts_as_transit_and_leaves_budgets_alone(system):
    """Stage 1 sets a step's deadline from its release, not from when it was read, and
    the delay at the gateway's send is not taken off the budget."""
    run = controllable(system())
    at0, at1 = [], []
    spy(run.servicer0, at0)
    spy(run.servicer1, at1)
    run.gateway.hop_delay = HopDelay(0.2)
    run.servicer1.hop_delay = HopDelay(0.05)
    timeout_ms = 20_000
    assert len(run.tokens(max_tokens=3, timeout_ms=timeout_ms)) == 3
    assert timeout_ms - at0[0][0] < 100  # stamped before the 200 ms at the gateway
    for budget, deadline, handled in at1:
        released = deadline - budget / 1000
        assert 0 <= handled - released < 0.03  # from the release, not 50 ms before it
    for (b0, *_), (b1, *_) in zip(at0, at1, strict=True):
        assert b0 - 50 < b1 <= b0  # Stage 1's 50 ms is not taken off either


def test_tokens_and_order_are_unchanged_by_a_delay(system):
    run = controllable(system())
    plain = [(t.token, t.top_logit) for t in run.tokens(max_tokens=6)]
    run.gateway.hop_delay = run.servicer1.hop_delay = HopDelay(0.02)
    delayed = [(t.token, t.top_logit) for t in run.tokens(max_tokens=6)]
    assert delayed == plain
    assert run.servicer0.order_violations == 0 and run.servicer1.order_violations == 0


def test_a_cancelled_stream_stops_holding_its_delayed_messages(system):
    """A message held 5 s at Stage 1 is let go once Stage 0's stream to it is cancelled
    (its channel closed): that stream ends within a second and the step is never run. A
    stream that is only half-closed still delivers what it holds, as TCP would."""
    run = controllable(system())
    run.servicer1.hop_delay = HopDelay(5.0)
    at1 = []
    spy(run.servicer1, at1)
    caller = threading.Thread(target=lambda: _quietly(run.tokens, max_tokens=2))
    caller.start()
    wait_for(lambda: run.servicer0.busy[1] == 1, "Stage 0 to forward the first step")
    time.sleep(0.3)  # the step is now held in Stage 1's delay
    run.servicer0.close()  # cancels Stage 0's stream to Stage 1
    start = time.monotonic()
    wait_for(lambda: not run.servicer1._inboxes, "the stream into Stage 1 to end", timeout=4)
    assert time.monotonic() - start < 1.0
    assert at1 == []
    caller.join(WAIT)


def _quietly(fn, **kwargs):
    try:
        fn(**kwargs)
    except grpc.RpcError:
        pass


def test_no_hop_delay_keeps_the_plain_paths(system):
    run = system()
    run.tokens(max_tokens=2)
    assert run.gateway.hop_delay is None and run.servicer1.hop_delay is None
    assert not run.gateway._link.delayed
