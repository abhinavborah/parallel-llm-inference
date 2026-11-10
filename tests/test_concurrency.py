"""Many requests in flight through one gateway and two stages (R4.5).

Every request goes to Stage 0 on the gateway's one stream, so the stages see the
requests' steps interleaved; each request still sends its next step only after its last
result, so its own steps stay strictly in order. Equality with the single-process
reference is on one host, fp32, batch 1, so tokens and top logits are compared exactly.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
from test_gateway import WAIT, System, code_of
from test_heartbeat import Beating
from test_model import tiny_reference

from pllm.gateway import max_positions
from pllm.proto import protocol_pb2
from pllm.reference import generate
from pllm.split import InProcessSplit

LAYERS = 4
K = 2
# Eight prompts of different lengths, and how many tokens each asks for.
JOBS = [
    ([5, 9, 100, 3, 42, 7], 6),
    ([11], 4),
    ([60, 61, 62, 63, 64, 65, 66, 67], 3),
    ([2, 4], 8),
    ([90, 17, 33], 5),
    ([7, 7, 7, 7], 7),
    ([45, 3, 81, 12, 6], 2),
    ([99, 1, 50], 6),
]


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


@pytest.fixture
def beating(tmp_path):
    """The same with each stage's heartbeat stream open, so a stage's Nack reaches the
    gateway as in a real run."""
    made = []

    def make(**kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        made.append(Beating(tmp_path, K, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def interleaved(order):
    """True if some request's messages have another request's between them."""
    firsts = {rid: order.index(rid) for rid in set(order)}
    lasts = {rid: len(order) - 1 - order[::-1].index(rid) for rid in set(order)}
    return any(firsts[a] < firsts[b] < lasts[a] for a in firsts for b in firsts if a != b)


def joined(threads):
    for thread in threads:
        thread.join(WAIT)
    assert not any(thread.is_alive() for thread in threads), "a request did not finish"


def reference(model_dir, ids, n):
    tokens, tops = generate(InProcessSplit.load(model_dir, K), ids, n)
    return list(zip(tokens, tops, strict=True))


def text(ids):
    return " ".join(str(i) for i in ids)


def run_together(run, jobs, **call):
    """Release every request at once; return each one's streamed Tokens, in job order."""
    start = threading.Barrier(len(jobs))

    def one(job):
        ids, n = job
        start.wait(WAIT)
        return list(run.call(prompt=text(ids), max_tokens=n, **call))

    with ThreadPoolExecutor(len(jobs)) as pool:
        return list(pool.map(one, jobs))


def spy_stage0(run):
    """Record (rid, step, start_pos, length) of each message Stage 0 handles, in order."""
    seen = []
    real = run.servicer0._handle

    def spy(request, context, *rest):
        seen.append((request.rid, request.step, request.start_pos, request.length))
        return real(request, context, *rest)

    run.servicer0._handle = spy
    return seen


# -------------------------------------------------------------- each as if alone


def test_eight_requests_together_each_produce_what_they_produce_alone(system, tmp_path):
    run = system(delay1=0.05)  # long enough for all eight to be in flight together
    seen = spy_stage0(run)
    outs = run_together(run, JOBS)
    assert interleaved([rid for rid, *_ in seen])
    assert len({rid for rid, *_ in seen}) == len(JOBS)
    for (ids, n), out in zip(JOBS, outs, strict=True):
        assert [(t.token, t.top_logit) for t in out] == reference(tmp_path, ids, n)
        assert [t.step for t in out] == list(range(n))
        assert len({t.rid for t in out}) == 1
    assert len({out[0].rid for out in outs}) == len(JOBS)  # no two requests share an id
    assert not run.gateway.requests  # all left the table


# -------------------------------------------------------------- interleaved, each in order


def test_requests_interleave_and_each_stays_strictly_sequential(system):
    run = system(delay1=0.05)  # long enough for the requests to overlap
    seen = spy_stage0(run)
    jobs = JOBS[:4]
    outs = run_together(run, jobs)
    order = [rid for rid, *_ in seen]
    for (ids, n), out in zip(jobs, outs, strict=True):
        rid = out[0].rid
        mine = [(step, pos, length) for r, step, pos, length in seen if r == rid]
        # one prefill of the whole prompt, then one token per step, one position further on
        expected = [(0, 0, len(ids))] + [(s, len(ids) + s - 1, 1) for s in range(1, n)]
        assert mine == expected
    assert interleaved(order), order


def test_a_later_shorter_request_can_finish_first(system, tmp_path):
    """No order between requests is promised or relied on: completion follows each
    request's own length, not the order the requests arrived in."""
    run = system(delay1=0.05)
    done = {}

    def go(name, ids, n, wait):
        time.sleep(wait)
        out = list(run.call(prompt=text(ids), max_tokens=n))
        done[name] = (time.monotonic(), out)

    long_job, short_job = (JOBS[0][0], 12), (JOBS[1][0], 2)
    threads = [
        threading.Thread(target=go, args=("long", *long_job, 0.0)),
        threading.Thread(target=go, args=("short", *short_job, 0.2)),
    ]
    for thread in threads:
        thread.start()
    joined(threads)
    assert done["short"][0] < done["long"][0]
    for name, (ids, n) in (("long", long_job), ("short", short_job)):
        assert [(t.token, t.top_logit) for t in done[name][1]] == reference(tmp_path, ids, n)


# -------------------------------------------------------------- throughput


def test_requests_in_flight_overlap_the_two_stages(system):
    """One request leaves one stage idle while the other works; several keep both busy.
    With each stage taking 0.1 s a step, 4 requests of 5 tokens take about 4 s one after
    another and about 2.1 s together."""
    run = system(delay0=0.1, delay1=0.1)
    jobs = [(ids, 5) for ids, _ in JOBS[:4]]
    began = time.monotonic()
    for ids, n in jobs:
        assert len(list(run.call(prompt=text(ids), max_tokens=n))) == n
    one_by_one = time.monotonic() - began
    began = time.monotonic()
    outs = run_together(run, jobs)
    together = time.monotonic() - began
    assert all(len(out) == 5 for out in outs)
    assert one_by_one > 3.5
    assert together < 0.75 * one_by_one, (together, one_by_one)


# -------------------------------------------------------------- one request's end is its own


def test_one_request_ending_badly_leaves_the_others_running(beating, tmp_path):
    """A cancel, a deadline and a refusal by Stage 1 each end their own request; the others
    finish with the reference's tokens, on the same stream to Stage 0."""
    run = beating(delay1=0.1)
    run.tokens(max_tokens=1)  # open the stream to Stage 0
    link = run.gateway._link
    healthy = [JOBS[0], JOBS[3], JOBS[7]]
    start = threading.Barrier(len(healthy) + 3)
    results = {}

    def healthy_one(i, ids, n):
        start.wait(WAIT)
        results[i] = list(run.call(prompt=text(ids), max_tokens=n))

    def cancelled():
        start.wait(WAIT)
        call = run.call(prompt=text(JOBS[1][0]), max_tokens=20)
        next(call)
        call.cancel()
        results["cancelled"] = True

    def late():
        start.wait(WAIT)
        results["late"] = code_of(run.call(prompt=text(JOBS[2][0]), max_tokens=20, timeout_ms=350))

    def refused():
        start.wait(WAIT)
        call = run.call(prompt=text(JOBS[4][0]), max_tokens=20)
        first = next(call)
        # Stage 1 forgets the request, so it refuses the next step with STALE_ATTEMPT; the
        # Nack reaches the gateway on Stage 1's heartbeat stream
        run.servicer1.free(protocol_pb2.Free(rid=first.rid, attempt=1))
        results["refused"] = code_of(call)

    threads = [
        threading.Thread(target=healthy_one, args=(i, *job)) for i, job in enumerate(healthy)
    ]
    threads += [threading.Thread(target=f) for f in (cancelled, late, refused)]
    for thread in threads:
        thread.start()
    joined(threads)
    assert results["late"][0] == grpc.StatusCode.DEADLINE_EXCEEDED
    assert results["refused"][0] == grpc.StatusCode.UNAVAILABLE
    assert "stage 1 refused" in results["refused"][1]
    assert results.get("cancelled")
    for i, (ids, n) in enumerate(healthy):
        assert [(t.token, t.top_logit) for t in results[i]] == reference(tmp_path, ids, n)
    assert run.gateway._link is link and link.failure() is None  # the stream carried on
    assert not run.gateway.requests


# -------------------------------------------------------------- admission


def test_a_request_longer_than_the_context_is_refused_before_stage0(system):
    """An unbounded max_tokens would grow the request's cache on both stages until a stage
    ran out of memory and ended the shared stream for every request."""
    run = system(max_positions=10)
    code, details = code_of(run.call(prompt=text(JOBS[0][0]), max_tokens=5))  # 6 + 5 > 10
    assert code == grpc.StatusCode.INVALID_ARGUMENT
    assert "together 11" in details and "context length of 10" in details
    assert len(run.servicer0.store) == 0 and run.gateway._link is None  # nothing was sent
    assert len(list(run.call(prompt=text(JOBS[0][0]), max_tokens=4))) == 4  # 6 + 4 fits


def test_the_context_length_comes_from_the_model_config(tmp_path):
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 8192}')
    assert max_positions(tmp_path) == 8192
    (tmp_path / "config.json").write_text("{}")
    assert max_positions(tmp_path) is None
