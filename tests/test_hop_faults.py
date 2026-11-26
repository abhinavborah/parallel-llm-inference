"""Delayed or lost hop messages (R6.11), on the hop wrapper of X4 (pllm.faults.HopDelay).

One message on one hop, the gateway to Stage 0 hop or the Stage 0 to Stage 1 hop, is lost or
held past the step timeout. The step timeout records the late step and declares nothing
(pllm.detection), so the request either completes once the message arrives inside its
deadline, or fails with DEADLINE_EXCEEDED at its deadline. A lost message does not hold up
the ones behind it, and the admission queue, credits and inboxes keep their bounds.
Real gateway and stages on loopback.
"""

import threading
from dataclasses import dataclass, field

import grpc
import pytest
from test_backpressure import FAST, run_all
from test_backpressure import system as system  # noqa: F401 (a fixture used below)
from test_deadlines import TOLERANCE
from test_gateway import WAIT, wait_for
from test_hop_delay import spy

from pllm.faults import DROP, HopDelay

pytestmark = pytest.mark.fault

HOPS = ("gateway", "stage1")  # where the wrapper sits: the gateway's send, Stage 1's receive
STEP_TIMEOUT_NS = 200_000_000
TIMEOUT_S = 1.5
TOKENS = 6
OVERLOAD_S = 6.0  # the lost request keeps a slot and a credit until then


@dataclass(frozen=True)
class Lose(HopDelay):
    """Holds the first message of the hop at step for extra seconds; DROP loses it. hit
    gets its (rid, attempt). The gateway sends on each request's thread, hence the lock."""

    step: int = 2
    extra: float = DROP
    hit: list = field(default_factory=list, compare=False, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    def release(self, message, now: float) -> float:
        with self.lock:
            if message.step == self.step and not self.hit:
                self.hit.append((message.rid, message.attempt))
                return now + self.extra
        return super().release(message, now)


def faulty(system, hop, fault, **slow):  # noqa: F811
    """A run with the hop wrapper on before the first request (the gateway's link fixes
    whether it delays when it opens), the fault-free tokens, then the fault applied."""
    run = system(running=3, max_queued=2, credits=2, inbox_size=2, **slow)
    run.gateway.detector.step_timeout_ns = STEP_TIMEOUT_NS
    owner = run.gateway if hop == "gateway" else run.servicer1
    owner.hop_delay = HopDelay()
    plain = [(t.token, t.top_logit) for t in run.tokens(max_tokens=TOKENS)]
    owner.hop_delay = fault
    return run, plain


def late_steps(run, rid):
    return [
        e
        for e in run.gateway.detector.events()
        if e.get("path") == "step_deadline" and e["rid"] == rid
    ]


def assert_bounds(run):
    """The bounds hold, and once every request has ended nothing is left in them."""
    gateway = run.gateway
    assert gateway.queue_peak <= 2 and gateway.in_flight_peak <= 2
    assert run.servicer0.inbox_peak <= 2 and run.servicer1.inbox_peak <= 2
    wait_for(lambda: not gateway.requests, "every request to leave the table")
    assert not gateway._waiting and gateway._running == 0
    wait_for(lambda: gateway._in_flight == 0, "every credit to come back")


@pytest.mark.parametrize("hop", HOPS)
def test_a_lost_hop_message_fails_its_request_at_the_deadline_and_holds_up_no_other(
    system,  # noqa: F811
    hop,
):
    """Step 2 of one of two requests is lost. That request fails with DEADLINE_EXCEEDED at
    its deadline after a step deadline record that declares nothing; the step never reaches
    the stage behind the hop; the other request finishes with the fault-free tokens."""
    lose = Lose()
    run, plain = faulty(system, hop, lose)
    handled = []
    spy(run.servicer0 if hop == "gateway" else run.servicer1, handled)
    out, _ = run_all(run, 2, max_tokens=TOKENS, timeout_ms=round(TIMEOUT_S * 1000))
    late = [took for got, took in out if got == grpc.StatusCode.DEADLINE_EXCEEDED]
    assert len(late) == 1 and [
        got for got, _ in out if got != grpc.StatusCode.DEADLINE_EXCEEDED
    ] == [plain], out
    assert TIMEOUT_S - 0.1 <= late[0] < TIMEOUT_S + TOLERANCE
    ((rid, attempt),) = lose.hit
    assert (rid, attempt, 2) not in [key for *_, key in handled]
    records = late_steps(run, rid)
    assert records and not any(e["declared"] for e in records)
    assert_bounds(run)


@pytest.mark.parametrize("hop", HOPS)
def test_a_hop_message_late_past_the_step_timeout_still_completes_its_request(
    system,  # noqa: F811
    hop,
):
    """Step 2 is held 0.4 s against a 0.2 s step timeout and a 5 s deadline: the step deadline
    is recorded, nothing is declared, and the request finishes with the fault-free tokens."""
    lose = Lose(extra=0.4)
    run, plain = faulty(system, hop, lose)
    got = [(t.token, t.top_logit) for t in run.tokens(max_tokens=TOKENS, timeout_ms=5000)]
    assert got == plain
    ((rid, _),) = lose.hit
    records = late_steps(run, rid)
    assert records and not any(e["declared"] for e in records)
    assert run.servicer0.order_violations == 0 and run.servicer1.order_violations == 0
    assert_bounds(run)


@pytest.mark.parametrize("hop", HOPS)
def test_a_hop_message_late_past_the_deadline_fails_its_request_and_its_result_is_dropped(
    system,  # noqa: F811
    hop,
):
    """Step 2 is held 2 s against a 1.5 s deadline: the request fails with DEADLINE_EXCEEDED
    at its deadline. The delay counts as transit, so the stage still runs the step when it
    arrives, and the gateway drops its late result as for a request it no longer has."""
    run, _ = faulty(system, hop, Lose(extra=2.0))
    dropped = run.gateway.discards["unknown_request"]
    out, _ = run_all(run, 1, max_tokens=TOKENS, timeout_ms=round(TIMEOUT_S * 1000))
    ((got, took),) = out
    assert got == grpc.StatusCode.DEADLINE_EXCEEDED
    assert TIMEOUT_S - 0.1 <= took < TIMEOUT_S + TOLERANCE
    assert_bounds(run)
    wait_for(
        lambda: run.gateway.discards["unknown_request"] > dropped,
        "the late result to be dropped at the gateway",
        timeout=WAIT,
    )


@pytest.mark.parametrize("extra", (DROP, 0.4), ids=("lost", "late"))
@pytest.mark.parametrize("hop", HOPS)
def test_under_overload_with_a_lost_or_late_hop_message_every_bound_is_reached_and_holds(
    system,  # noqa: F811
    hop,
    extra,
):
    """Stage 1 slowed, five requests fill the 3 running slots and the queue of 2 while step 2
    of one of them is lost or held past the step timeout, then a sixth is refused at once.
    The queue and the steps in flight reach their bounds and never pass them; the requests
    the fault did not hit give the fault-free tokens, and a lost one fails at its deadline."""
    lose = Lose(extra=extra)
    run, plain = faulty(system, hop, lose, delay1=0.05)
    results = []
    first = threading.Thread(
        target=lambda: results.append(
            run_all(run, 5, max_tokens=TOKENS, timeout_ms=round(OVERLOAD_S * 1000))[0]
        )
    )
    first.start()
    wait_for(lambda: len(run.gateway._waiting) == 2, "three running and two queued")
    ((refused, took),) = run_all(run, 1, max_tokens=TOKENS, timeout_ms=5000)[0]
    first.join(WAIT)
    (out,) = results
    assert refused == grpc.StatusCode.RESOURCE_EXHAUSTED and took < FAST
    late = [took for got, took in out if got == grpc.StatusCode.DEADLINE_EXCEEDED]
    done = [got for got, _ in out if isinstance(got, list)]
    assert len(late) == (extra == DROP) and len(late) + len(done) == 5, out
    assert all(got == plain for got in done)
    assert all(OVERLOAD_S - 0.1 <= took < OVERLOAD_S + TOLERANCE for took in late)
    assert len(lose.hit) == 1
    assert run.gateway.queue_peak == 2 and run.gateway.in_flight_peak == 2
    assert_bounds(run)
