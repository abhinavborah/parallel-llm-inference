"""Cross-test of the gateway's replay (R6.5 to R6.7), written from the requirements and the
README's Replay section by the member who did not build it.

What is claimed, and where it is tested here:

- R6.5: a declared death bumps the attempt of every request in the table, idle ones too.
- R6.6: the replay resends the prompt at start_pos 0, then one single-token message per
  token already sent, all with the replay flag, under the new attempt, then decodes on.
- R6.7: replayed tokens are compared with the history and never streamed again; a mismatch
  is counted, logged and written as a divergence line; the output equals a fault-free run.
- A GAP Nack replays; an error inside a live stage does not.
- A refused replay is tried again after a pause, under a newer attempt, and finishes.
- With no replacement the request fails with DEADLINE_EXCEEDED at its deadline.

Equality with a fault-free run is on one host, fp32, batch 1. The messages are observed
where the gateway sends them (the recorder's hop), so nothing here depends on how a stage
handles them.
"""

import json
import logging
import threading
import time
from collections import defaultdict

import grpc
import pytest
from test_deadlines import TOLERANCE
from test_detection import start_replacement, stop_stage
from test_gateway import IDS, code_of, wait_for
from test_heartbeat import Beating, reference
from test_model import tiny_reference

from pllm import metrics
from pllm.proto import protocol_pb2
from pllm.stage_service import StageServicer

LAYERS = 4
K = 2
BEAT = 0.05
P = len(IDS)
PAUSE = 0.05  # the README's pause before a refused replay starts again


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


class Sent:
    """Every step the gateway sends to Stage 0, in order, with the time it was sent."""

    def __init__(self, gateway):
        self.rows = []
        real = gateway.recorder.hop

        def hop(name, message, accepted=True):
            if name == metrics.GATEWAY_STAGE0:
                self.rows.append(
                    {
                        "at": time.monotonic(),
                        "rid": message.rid,
                        "attempt": message.attempt,
                        "shape": (
                            message.step,
                            message.start_pos,
                            message.length,
                            message.replay,
                            list(message.tokens.ids),
                        ),
                    }
                )
            return real(name, message, accepted)

        gateway.recorder.hop = hop

    def attempts(self, rid, first=2):
        """The messages of each attempt from `first` on, as {attempt: [shape, ...]}."""
        out = defaultdict(list)
        for row in self.rows:
            if row["rid"] == rid and row["attempt"] >= first:
                out[row["attempt"]].append(row["shape"])
        return dict(out)


def first_steps(sent, rid):
    """When each replay attempt began: the time of its step 0, by attempt."""
    return {
        row["attempt"]: row["at"]
        for row in sent.rows
        if row["rid"] == rid and row["attempt"] >= 2 and row["shape"][0] == 0
    }


def take(stream, n):
    return [next(stream) for _ in range(n)]


def full_replay(history):
    """What R6.6 says one replay sends for a history of m tokens, then the next decode."""
    m = len(history)
    steps = [(0, 0, P, True, list(IDS))]
    steps += [(k, P + k - 1, 1, True, [history[k - 1]]) for k in range(1, m)]
    steps += [(m, P + m - 1, 1, False, [history[m - 1]])]
    return steps


def kill(run, stage):
    """Stop a stage, and wait until the gateway has declared its epoch dead."""
    old = run.registry.stages[stage].epoch
    stop_stage(run, stage)
    wait_for(lambda: run.gateway.detector.is_dead(old), "the dead epoch to be declared")
    return old


def revive(run, stage):
    return start_replacement(run, stage)[0]


# ------------------------------------------------------------------ R6.5: every request


@pytest.mark.parametrize("stage", [0, 1])
def test_a_declared_death_bumps_the_attempt_of_every_request_idle_ones_included(
    beating, tmp_path, stage
):
    """One credit lets one step run at a time, so most of the three requests are idle at any
    moment, holding KV on both stages and waiting for the credit. All of them are bumped,
    once, by the declaration itself, and all finish with the fault-free tokens."""
    run = beating(credits=1, delay1=0.05)
    expected = reference(tmp_path, K, IDS, 10)[0]
    streams = [run.call(max_tokens=10) for _ in range(3)]
    got = [take(stream, 1) for stream in streams]
    rids = {tokens[0].rid for tokens in got}
    assert len(rids) == 3 and {row["rid"] for row in run.gateway.table()} == rids
    before = {row["rid"]: row["attempt"] for row in run.gateway.table()}
    assert set(before.values()) == {1}

    kill(run, stage)  # nothing replaces it yet, so nobody can have moved on

    after = {row["rid"]: row["attempt"] for row in run.gateway.table()}
    assert after == {rid: 2 for rid in rids}  # every one, idle ones too, bumped exactly once
    revive(run, stage)
    for stream, tokens in zip(streams, got, strict=True):
        tokens += list(stream)
        assert [t.token for t in tokens] == expected
        assert [t.step for t in tokens] == list(range(10))
    assert run.gateway.replays >= 3 and run.gateway.divergences == 0
    wait_for(lambda: run.gateway.table() == [], "the finished requests to leave the table")


def test_a_request_that_has_finished_is_not_in_the_table_to_be_bumped(beating):
    """Only requests with KV state are replayed: a finished one is gone from the table."""
    run = beating()
    assert len(list(run.call(max_tokens=2))) == 2
    wait_for(lambda: run.gateway.table() == [], "the finished request to leave the table")
    kill(run, 1)
    assert run.gateway.table() == [] and run.gateway.replays == 0


# ------------------------------------------------------------------ R6.6: the messages


@pytest.mark.parametrize("stage", [0, 1])
def test_the_replay_sends_the_prompt_then_one_token_per_message_all_flagged(
    beating, tmp_path, stage
):
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 9)[0]
    sent = Sent(run.gateway)
    stream = run.call(max_tokens=9)
    tokens = take(stream, 4)
    rid = tokens[0].rid
    kill(run, stage)
    history = list(run.gateway.requests[rid].tokens)  # fixed now: the pipeline is down
    m = len(history)
    assert m >= 4
    revive(run, stage)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    assert history == expected[:m]

    attempts = sent.attempts(rid)
    assert min(attempts) == 2  # the first replay is the attempt the declaration made
    plan = full_replay(history)
    for attempt, shapes in attempts.items():
        # an attempt that a stage still refused stops early, but never sends anything else
        assert shapes[: len(plan)] == plan[: len(shapes)], attempt
        assert shapes[0] == (0, 0, P, True, list(IDS))  # always the whole prompt first
    last = attempts[max(attempts)]
    assert last[: m + 1] == plan  # the last attempt replayed all m tokens, then decoded on
    assert all(shape[3] is True for shape in last[:m])  # every replayed message is flagged
    assert [shape[3] for shape in last[m:]] == [False] * (len(last) - m)  # then none is
    assert [shape[0] for shape in last] == list(range(len(last)))  # steps in order, no gaps


def test_every_attempt_is_new_and_never_reuses_an_older_one(beating):
    """A replay never reuses an attempt, so a late result of an older one can be told apart."""
    run = beating(delay1=0.05)
    sent = Sent(run.gateway)
    stream = run.call(max_tokens=8)
    rid = take(stream, 3)[0].rid
    kill(run, 1)
    revive(run, 1)
    list(stream)
    starts = [r for r in sent.rows if r["rid"] == rid and r["shape"][0] == 0]
    attempts = [r["attempt"] for r in starts]
    assert attempts[0] == 1 and attempts == sorted(set(attempts))  # one start per attempt
    assert len(attempts) >= 2


# ------------------------------------------------------------------ R6.7: verified, not resent


def test_replayed_tokens_are_not_streamed_again_and_nothing_diverges(beating, tmp_path):
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 9)[0]
    stream = run.call(max_tokens=9)
    tokens = take(stream, 4)
    kill(run, 1)
    revive(run, 1)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    assert [t.step for t in tokens] == list(range(9))  # each step reached the client once
    assert run.gateway.tokens_out == 9  # and was counted once
    assert run.gateway.divergences == 0
    assert run.gateway.replays >= 1


def test_a_replayed_token_that_differs_from_the_history_is_counted_logged_and_written(
    beating, tmp_path, caplog
):
    """The one token the replay cannot reproduce is forced by changing the history of the
    last token sent. The replay of that step then returns the true token, which differs from
    what the history says: exactly one divergence, and still no token sent twice."""
    run = beating(delay1=0.05)
    path = tmp_path / "metrics.jsonl"
    run.gateway.recorder = metrics.Recorder(path)
    stream = run.call(max_tokens=9)
    tokens = take(stream, 4)
    rid = tokens[0].rid
    already = [t.token for t in tokens]
    kill(run, 1)
    request = run.gateway.requests[rid]
    last = len(request.tokens) - 1
    true = request.tokens[last]
    request.tokens[last] = true + 1
    with caplog.at_level(logging.WARNING, logger="pllm.gateway"):
        revive(run, 1)
        tokens += list(stream)
    assert [t.step for t in tokens] == list(range(9))  # the replay streamed nothing again
    assert [t.token for t in tokens[:4]] == already  # what the client had is untouched
    assert tokens[last].token == true  # and so is the token the history was changed for
    assert run.gateway.divergences == 1
    assert any("divergence" in record.getMessage() for record in caplog.records)
    run.gateway.recorder.finish()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    found = [line for line in lines if line["type"] == "divergence"]
    assert len(found) == 1
    assert (found[0]["rid"], found[0]["step"]) == (rid, last)
    assert (found[0]["token"], found[0]["history_token"]) == (true, true + 1)


# ------------------------------------------------------------------ what starts a replay


def test_a_gap_nack_replays_the_request_without_any_death_declared(beating, tmp_path):
    """A stage that has lost the request's cache refuses the next step with GAP. The gateway
    replays instead of failing the request, and no epoch is declared dead."""
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 8)[0]
    sent = Sent(run.gateway)
    stream = run.call(max_tokens=8)
    tokens = take(stream, 3)
    rid = tokens[0].rid
    run.servicer1.free(protocol_pb2.Free(rid=rid))  # Stage 1 forgets the request
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    assert [t.step for t in tokens] == list(range(8))
    assert run.gateway.replays == 1 and run.gateway.divergences == 0
    assert not [e for e in run.gateway.detector.events() if e.get("declared")]
    attempts = sent.attempts(rid)
    assert min(attempts) == 2 and attempts[2][0] == (0, 0, P, True, list(IDS))


def test_an_error_inside_a_live_stage_is_not_replayed(beating, monkeypatch):
    """Stage 0 fails a step but keeps beating, so nobody is dead: the request fails with
    UNAVAILABLE. Replaying would only hit the same fault again."""
    run = beating(delay1=0.05)
    stream = run.call(max_tokens=50, timeout_ms=30_000)
    take(stream, 2)

    def broken(inputs, start_pos, cache):
        raise RuntimeError("a fault injected into stage 0")

    monkeypatch.setattr(run.servicer0.worker, "step", broken)
    code, _ = code_of(stream)
    assert code == grpc.StatusCode.UNAVAILABLE
    assert run.gateway.replays == 0
    assert not [e for e in run.gateway.detector.events() if e.get("declared")]
    wait_for(lambda: run.gateway.table() == [], "the failed request to leave the table")


def test_a_replay_refused_before_the_route_update_is_tried_again_after_a_pause(
    beating, tmp_path, monkeypatch
):
    """The new Stage 1 does not yet know Stage 0's epoch, so it refuses the replay. The
    request starts over under a newer attempt, not at once but after a pause, and finishes
    with the fault-free tokens when the route update arrives."""
    hold = 0.6
    run = beating(delay1=0.05)
    expected = reference(tmp_path, K, IDS, 8)[0]
    sent = Sent(run.gateway)
    stream = run.call(max_tokens=8)
    tokens = take(stream, 3)
    rid = tokens[0].rid
    old = {run.servicer0, run.servicer1}
    real_route = StageServicer.route

    def late(self, update):
        if self not in old:  # a stage started after the failure learns its route late
            threading.Timer(hold, real_route, (self, update)).start()
            return True
        return real_route(self, update)

    monkeypatch.setattr(StageServicer, "route", late)
    kill(run, 1)
    revive(run, 1)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    assert [t.step for t in tokens] == list(range(8))

    began = first_steps(sent, rid)
    assert len(began) >= 2  # the first replay was refused and started again
    assert list(began) == sorted(began)  # under newer attempts, never the same one
    times = [began[a] for a in sorted(began)]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert min(gaps) >= PAUSE * 0.6  # each retry waited
    assert len(began) <= hold / (PAUSE * 0.6) + 3  # and did not spin through the hold


@pytest.mark.parametrize("stage", [0, 1])
def test_with_no_replacement_the_request_fails_with_deadline_exceeded_at_its_deadline(
    beating, stage
):
    run = beating(delay1=0.05)
    stream = run.call(max_tokens=50, timeout_ms=1500)
    take(stream, 2)
    began = time.monotonic()
    kill(run, stage)
    code, details = code_of(stream)
    took = time.monotonic() - began
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED and "replay" in details
    assert took < 1.5 + TOLERANCE
    assert run.gateway.replays >= 1
    wait_for(lambda: run.gateway.table() == [], "the failed request to leave the table")


# ------------------------------------------------------------------ the end result


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("before", [1, 5, 9])
def test_the_tokens_after_a_recovery_equal_a_fault_free_run(beating, tmp_path, stage, before):
    """Whatever the stage was in the middle of, the client sees the fault-free tokens, each
    step once, wherever in the request the stage died."""
    run = beating(delay1=0.03)
    expected = reference(tmp_path, K, IDS, 12)[0]
    stream = run.call(max_tokens=12)
    tokens = take(stream, before)
    kill(run, stage)
    revive(run, stage)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    assert [t.step for t in tokens] == list(range(12))
    assert run.gateway.divergences == 0
