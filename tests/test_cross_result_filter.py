"""Cross-tests of the gateway result filter (R4.7, R4.8, R3.14).

Written by the member who did not implement the filter, against its documented rule: a
result is accepted only when its request is running and its attempt, epoch and step all
match; otherwise it is dropped for the first reason in the order unknown_request,
stale_attempt, stale_epoch, duplicate, ahead.

- Every combination of a match or a mismatch on each field gets the reason that order
  gives, through Results.Push, and is counted once.
- A real request keeps its single-process result while another stream floods the gateway
  with wrong results; each one is dropped and counted, and none reaches the client.

Equality with the request run alone is on one host, fp32, batch 1.
"""

import collections
import itertools
import random
import threading
import time

import grpc
from test_gateway import WAIT
from test_results import FAKE, alone, live_epoch, push, result
from test_results import system as system  # noqa: F401 (a fixture used below)

from pllm.gateway import Request
from pllm.proto import protocol_pb2_grpc

SEED = 31
REASONS = ("unknown_request", "stale_attempt", "stale_epoch", "duplicate", "ahead")


def expected_reason(known, attempt_ok, epoch_ok, step):
    """The documented order; step is "behind", "awaited" or "ahead"."""
    if not known:
        return "unknown_request"
    if not attempt_ok:
        return "stale_attempt"
    if not epoch_ok:
        return "stale_epoch"
    return {"behind": "duplicate", "awaited": None, "ahead": "ahead"}[step]


def test_every_combination_of_mismatches_gets_the_first_reason_in_order(system):  # noqa: F811
    run = system()
    request = Request("r", 2, time.monotonic() + 60, 3)
    request.expected_step = 5
    with run.gateway._lock:
        run.gateway.requests["r"] = request
    live = live_epoch(run)
    steps = {"behind": 4, "awaited": 5, "ahead": 6}
    combos = list(itertools.product([True, False], [True, False], [True, False], steps))
    want = collections.Counter()
    sent = []
    for known, attempt_ok, epoch_ok, step in combos:
        sent.append(
            result(
                "r" if known else "other",
                steps[step],
                attempt=2 if attempt_ok else 1,
                epoch=live if epoch_ok else live + 1,
                run=run,
            )
        )
        reason = expected_reason(known, attempt_ok, epoch_ok, step)
        if reason is not None:
            want[reason] += 1
        with run.gateway._lock:  # the filter itself, case by case
            got = run.gateway._why_not(sent[-1], run.gateway.requests.get(sent[-1].rid))
        assert got == reason, (known, attempt_ok, epoch_ok, step)
    push(run, *sent)
    with run.gateway._lock:
        del run.gateway.requests["r"]
    assert run.gateway.discards == want
    assert set(want) == set(REASONS)
    assert run.gateway.dropped == len(combos) - 1
    accepted = request.results.get_nowait()
    assert (accepted.attempt, accepted.step) == (2, 5) and request.results.empty()


def test_a_flood_of_wrong_results_never_reaches_the_client(system):  # noqa: F811
    """While a real request runs, a second Push stream sends wrong results until it ends:
    unknown ids, other attempts, old epochs, steps far ahead, and step 0 again once it is
    done. Each is dropped and counted; the client gets the result of the request alone."""
    expected = alone(system, n=8)
    run = system(delay1=0.1)
    stream = run.call(max_tokens=8)
    first = next(stream)
    rid, live = first.rid, live_epoch(run)
    rng = random.Random(SEED)
    stop = threading.Event()
    sent = collections.Counter()

    def flood():
        while not stop.is_set():
            kind = rng.choice(REASONS)
            sent[kind] += 1
            yield {
                "unknown_request": lambda: result("nobody", rng.randint(0, 6), run=run),
                "stale_attempt": lambda: result(rid, rng.randint(0, 6), attempt=2, run=run),
                "stale_epoch": lambda: result(rid, rng.randint(0, 6), epoch=live + 1, run=run),
                "duplicate": lambda: result(rid, 0, run=run),
                "ahead": lambda: result(rid, rng.randint(100, 200), run=run),
            }[kind]()
            time.sleep(0.001)

    pusher = threading.Thread(target=stream_push, args=(run, flood()), daemon=True)
    pusher.start()
    got = [(first.token, first.top_logit)] + [(t.token, t.top_logit) for t in stream]
    stop.set()
    pusher.join(timeout=10)
    assert not pusher.is_alive()
    assert got == expected and FAKE not in [token for token, _ in got]
    assert run.gateway.dropped == sum(run.gateway.discards.values()) == sum(sent.values())
    # a wrong result that lands after the request ended is unknown_request instead
    for kind in REASONS[1:]:
        assert 1 <= run.gateway.discards[kind] <= sent[kind]  # the flood met the live request
    assert run.gateway.discards["unknown_request"] >= sent["unknown_request"]


def stream_push(run, results):
    """Push results on one stream as they are made, as Stage 1 does."""
    with grpc.insecure_channel(run.address) as channel:
        protocol_pb2_grpc.ResultsStub(channel).Push(results, timeout=WAIT)
