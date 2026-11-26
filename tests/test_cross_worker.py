"""Cross-tests of the stage worker: KV truncate and fencing (R3.6, R3.11, R3.14).

Written by the member who did not implement the KV store or the stage fencing, against
their documented rules rather than their tests:

- A seeded random run of acquire and free agrees with a small model of the rules at every
  operation (result, length, attempt), and the cache holds what a fresh run of the kept
  messages holds, bitwise.
- A replay that truncates both stages at a message boundary gives the results of a fresh
  run, bitwise, through the real stage servers.
- The epoch checks come before the cache rules, so a message that breaks both is refused
  by epoch and changes nothing.
- A refused message of each kind is neither cached nor forwarded, and the stream carries on.
- A failed step frees only its own request's cache, and that request can start again.

Equality is bitwise, on one host, fp32, batch 1.
"""

import random

import grpc
import pytest
import torch
from test_fencing import IDS, nack, no_nack, prefill, send
from test_fencing import system as system  # noqa: F401 (a fixture used below)
from test_gateway import wait_for
from test_kvstore import FIRST_LAYERS, fresh
from test_kvstore import stage as stage  # noqa: F401 (a fixture used below)
from test_kvstore import store as store  # noqa: F401 (a fixture used below)

from pllm import metrics
from pllm.kvstore import Gap, Stale

SEED = 31
OPERATIONS = 600
RIDS = ("a", "b", "c")


# -------------------------------------------------------------- the store against a model


class Model:
    """The store's rules as its docstring states them, kept per request id."""

    def __init__(self):
        self.messages = {}  # rid to the (start_pos, tokens) messages its cache holds
        self.attempts = {}  # rid to the newest attempt that used its cache
        self.freed = {}  # rid to the newest attempt freed

    def length(self, rid):
        return sum(len(tokens) for _, tokens in self.messages.get(rid, []))

    def acquire(self, rid, start_pos, attempt):
        """The exception acquire must raise, as (type, freed), or None."""
        freed, known = self.freed.get(rid), self.attempts.get(rid)
        if attempt is not None and freed is not None and attempt <= freed:
            return Stale, True
        if attempt is not None and known is not None and attempt < known:
            return Stale, False
        if rid not in self.messages:
            if start_pos != 0:
                return Gap, None
        elif start_pos > self.length(rid):
            return Gap, None
        return None

    def append(self, rid, start_pos, tokens, attempt):
        kept = [m for m in self.messages.get(rid, []) if m[0] < start_pos]
        self.messages[rid] = [*kept, (start_pos, tokens)]
        if attempt is not None:
            self.attempts[rid] = max(attempt, self.attempts.get(rid, attempt))

    def free(self, rid, attempt):
        if attempt is not None:
            self.freed[rid] = max(attempt, self.freed.get(rid, attempt))
            if self.attempts.get(rid, attempt) > attempt:
                return False
        self.attempts.pop(rid, None)
        return self.messages.pop(rid, None) is not None


def boundaries(model, rid):
    """Positions a message may start at: each message boundary, and one past the end."""
    starts = [start for start, _ in model.messages.get(rid, [])]
    return [*starts, model.length(rid), model.length(rid) + 1]


def maybe_attempt(rng):
    return None if rng.random() < 0.1 else rng.randint(1, 4)


def test_a_random_run_of_acquire_and_free_follows_the_rules(stage, store):  # noqa: F811
    rng = random.Random(SEED)
    model = Model()
    seen = {"append": 0, "truncate": 0, "gap": 0, "stale_freed": 0, "stale_older": 0}
    for _ in range(OPERATIONS):
        rid = rng.choice(RIDS)
        if rng.random() < 0.15:
            attempt = maybe_attempt(rng)
            assert store.free(rid, attempt) == model.free(rid, attempt)
        else:
            start_pos = rng.choice(boundaries(model, rid))
            attempt = maybe_attempt(rng)
            tokens = [rng.randint(0, 127) for _ in range(rng.randint(1, 3))]
            refused = model.acquire(rid, start_pos, attempt)
            if refused is None:
                seen["truncate" if start_pos < model.length(rid) else "append"] += 1
                cache = store.acquire(rid, start_pos, attempt)
                assert len(cache) == start_pos
                stage.forward(torch.tensor(tokens), start_pos, cache)
                model.append(rid, start_pos, tokens, attempt)
            else:
                kind, freed = refused
                with pytest.raises(kind) as caught:
                    store.acquire(rid, start_pos, attempt)
                if kind is Stale:
                    assert caught.value.freed is freed
                    seen["stale_freed" if freed else "stale_older"] += 1
                else:
                    seen["gap"] += 1
        for each in RIDS:  # every request, so a raise is seen to change nothing anywhere
            assert store.length(each) == model.length(each)
            assert store.attempt(each) == model.attempts.get(each)
            assert (each in store) == (each in model.messages)
        assert len(store) == len(model.messages)
    assert all(seen.values()), seen  # the run reached every rule
    for rid in store.rids():
        messages = model.messages[rid]
        tokens = torch.tensor([t for _, ts in messages for t in ts])
        _, expected = fresh(stage, tokens, [len(ts) for _, ts in messages])
        got = store._caches[rid]
        assert len(got.keys) == FIRST_LAYERS
        for mine, theirs in zip(
            got.keys + got.values, expected.keys + expected.values, strict=True
        ):
            assert torch.equal(mine, theirs)


# -------------------------------------------------------------- truncate through the chain


def step_message(run, rid, step, start_pos, ids, attempt=1):
    message = prefill(run, rid, attempt=attempt, start_pos=start_pos, ids=ids)
    message.step = step
    return message


def results_of(run):
    """Every result Stage 1 pushes to the gateway, accepted or not, in arrival order."""
    seen = []
    real = run.gateway.recorder.hop

    def hop(name, message, accepted=True):
        if name == metrics.STAGE1_GATEWAY:
            seen.append(message)
        return real(name, message, accepted)

    run.gateway.recorder.hop = hop
    return seen


def test_a_replay_that_truncates_both_stages_matches_a_fresh_run(system):  # noqa: F811
    """ "fresh" runs the prompt then tokens 7 and 11. "replayed" runs the prompt then 50 and
    60, and attempt 2 replays from the end of the prompt with 7 and 11, so both stages
    truncate to 3 positions and append. Its results equal fresh's, bitwise."""
    run = system()
    seen = results_of(run)
    n = len(IDS)
    send(
        run.ports[0],
        step_message(run, "fresh", 0, 0, IDS),
        step_message(run, "fresh", 1, n, [7]),
        step_message(run, "fresh", 2, n + 1, [11]),
        step_message(run, "replayed", 0, 0, IDS),
        step_message(run, "replayed", 1, n, [50]),
        step_message(run, "replayed", 2, n + 1, [60]),
    )
    wait_for(lambda: len(seen) == 6, "the first six results")
    assert run.servicer1.store.length("replayed") == n + 2
    send(
        run.ports[0],
        step_message(run, "replayed", 1, n, [7], attempt=2),
        step_message(run, "replayed", 2, n + 1, [11], attempt=2),
    )
    wait_for(lambda: len(seen) == 8, "the replayed results")
    by = {(r.rid, r.attempt, r.step): (r.token, r.top_logit) for r in seen}
    for step in (1, 2):
        assert by[("replayed", 2, step)] == by[("fresh", 1, step)]  # float equality: bitwise
    assert (
        by[("replayed", 1, 1)] != by[("fresh", 1, 1)]
        or by[("replayed", 1, 2)] != by[("fresh", 1, 2)]
    )  # the first attempt really differed, so the truncation did the work
    for servicer in (run.servicer0, run.servicer1):
        assert servicer.store.length("replayed") == n + 2
        assert servicer.store.attempt("replayed") == 2
    assert no_nack(run.servicer0) and no_nack(run.servicer1)


# -------------------------------------------------------------- epochs before the cache


@pytest.mark.parametrize(
    ("epochs", "reason"),
    [({"target_epoch": 99}, "WRONG_EPOCH"), ({"sender_epoch": 98}, "BAD_SENDER")],
)
def test_an_epoch_refusal_comes_before_the_attempt_rule(system, epochs, reason):  # noqa: F811
    run = system()
    send(run.ports[0], prefill(run, "r", attempt=3))
    length = run.servicer0.store.length("r")
    send(run.ports[0], prefill(run, "r", attempt=1, start_pos=length, ids=[7], **epochs))
    refused = nack(run.servicer0)
    assert (refused.reason, refused.attempt) == (reason, 1)
    assert run.servicer0.store.length("r") == length
    assert run.servicer0.store.attempt("r") == 3


# -------------------------------------------------------------- a refusal leaves no trace


REFUSALS = {
    "WRONG_EPOCH": lambda run: prefill(run, "bad", target_epoch=run.servicer0.own_epoch + 1),
    "BAD_SENDER": lambda run: prefill(run, "bad", sender_epoch=7),
    "STALE_ATTEMPT": lambda run: prefill(run, "bad", attempt=1, start_pos=len(IDS), ids=[7]),
    "GAP": lambda run: prefill(run, "bad", start_pos=4, ids=[7]),
}


@pytest.mark.parametrize("reason", sorted(REFUSALS))
def test_a_refused_message_is_not_cached_or_forwarded_and_the_stream_goes_on(system, reason):  # noqa: F811
    """Each refusal sits between two good messages on one stream. The refused request's
    cache on both stages is as it was, and the message after it is served."""
    run = system()
    if reason == "STALE_ATTEMPT":  # "bad" exists, used by attempt 2
        send(run.ports[0], prefill(run, "bad", attempt=2))
        wait_for(lambda: "bad" in run.servicer1.store, "the attempt 2 prefill")
    before = [s.store.length("bad") for s in (run.servicer0, run.servicer1)]
    send(run.ports[0], prefill(run, "first"), REFUSALS[reason](run), prefill(run, "after"))
    refused = nack(run.servicer0)
    assert (refused.reason, refused.rid) == (reason, "bad")
    wait_for(lambda: "after" in run.servicer1.store, "the message after the refusal")
    assert "first" in run.servicer1.store
    assert [s.store.length("bad") for s in (run.servicer0, run.servicer1)] == before
    assert no_nack(run.servicer0) and no_nack(run.servicer1)


# -------------------------------------------------------------- a failed step


def test_a_failed_step_frees_only_its_cache_and_the_request_can_start_again(
    system,  # noqa: F811
    monkeypatch,
):
    run = system()
    send(run.ports[0], prefill(run, "keep"), prefill(run, "boom"))
    worker = run.servicer0.worker
    real = worker.step

    def fail_once(inputs, start_pos, cache):
        monkeypatch.setattr(worker, "step", real)
        raise RuntimeError("injected")

    monkeypatch.setattr(worker, "step", fail_once)
    with pytest.raises(grpc.RpcError) as caught:
        send(run.ports[0], prefill(run, "boom", start_pos=len(IDS), ids=[7]))
    assert caught.value.code() == grpc.StatusCode.INTERNAL
    assert "boom" not in run.servicer0.store
    assert run.servicer0.store.length("keep") == len(IDS)
    send(run.ports[0], prefill(run, "boom"))  # the same attempt, from position 0
    assert run.servicer0.store.length("boom") == len(IDS)
    assert no_nack(run.servicer0)
