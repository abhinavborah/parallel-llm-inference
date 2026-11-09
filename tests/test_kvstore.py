"""The per-request KV store: position rules, release and memory cost (R3.5, R3.6, R3.9).

The rules are tested on the store with real stage caches, then through the chain in
test_chain.py. Equality is bitwise, on one host, fp32, batch 1.
"""

import pytest
import torch
from test_model import tiny_reference

from pllm import kvstore
from pllm.kvstore import Gap, KVStore, Stale
from pllm.split import Stage

LAYERS = 5
FIRST_LAYERS = 2


@pytest.fixture
def stage(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    return Stage.load(tmp_path, 0, FIRST_LAYERS)


@pytest.fixture
def store(stage):
    return KVStore(stage.config, FIRST_LAYERS, stage.new_cache)


def run(stage, store, rid, tokens, start_pos):
    """One message through the stage, with the cache the store hands out."""
    return stage.forward(tokens, start_pos, store.acquire(rid, start_pos))


def fresh(stage, tokens, sizes):
    """Hidden states from sending tokens as consecutive messages of the given sizes."""
    cache, out, pos = stage.new_cache(), [], 0
    for size in sizes:
        out.append(stage.forward(tokens[pos : pos + size], pos, cache))
        pos += size
    return torch.cat(out), cache


TOKENS = torch.tensor([5, 9, 100, 3, 42, 7, 11, 60, 2, 77])


def test_a_new_request_starts_at_position_zero(store):
    cache = store.acquire("a", 0)
    assert len(cache) == 0 and store.rids() == {"a"} and len(store) == 1


def test_a_start_equal_to_the_cached_length_appends(stage, store):
    first = run(stage, store, "a", TOKENS[:6], 0)
    second = run(stage, store, "a", TOKENS[6:8], 6)
    third = run(stage, store, "a", TOKENS[8:], 8)
    expected, cache = fresh(stage, TOKENS, [6, 2, 2])
    assert torch.equal(torch.cat([first, second, third]), expected)
    assert store.length("a") == len(cache) == 10


def test_a_start_below_the_cached_length_truncates_then_appends(stage, store):
    run(stage, store, "a", TOKENS[:6], 0)
    run(stage, store, "a", TOKENS[6:8], 6)
    run(stage, store, "a", TOKENS[8:], 8)
    # replay from the end of the first message: the cache must forget positions 6 to 9
    cache = store.acquire("a", 6)
    assert len(cache) == 6
    again = stage.forward(TOKENS[6:8], 6, cache)
    expected, _ = fresh(stage, TOKENS[:8], [6, 2])
    assert torch.equal(again, expected[6:])
    assert store.length("a") == 8


def test_truncating_to_zero_restarts_the_request(stage, store):
    first = run(stage, store, "a", TOKENS[:6], 0)
    again = run(stage, store, "a", TOKENS[:6], 0)
    assert torch.equal(first, again) and store.length("a") == 6


def test_a_start_above_the_cached_length_is_a_gap_and_changes_nothing(stage, store):
    run(stage, store, "a", TOKENS[:6], 0)
    with pytest.raises(Gap, match="starts at position 7 but this stage caches 6 positions") as err:
        store.acquire("a", 7)
    assert (err.value.rid, err.value.start_pos, err.value.length) == ("a", 7, 6)
    assert store.length("a") == 6  # untouched


def test_an_unknown_request_that_does_not_start_at_zero_is_a_gap(store):
    with pytest.raises(Gap, match="has no cache for it") as err:
        store.acquire("ghost", 3)
    assert err.value.length is None
    assert "ghost" not in store and len(store) == 0  # a refused message creates nothing


def test_requests_do_not_share_state(stage, store):
    a = run(stage, store, "a", TOKENS[:6], 0)
    b = run(stage, store, "b", TOKENS[2:5], 0)
    run(stage, store, "a", TOKENS[6:8], 6)
    assert torch.equal(a, fresh(stage, TOKENS[:6], [6])[0])
    assert torch.equal(b, fresh(stage, TOKENS[2:5], [3])[0])
    assert store.length("a") == 8 and store.length("b") == 3


def test_free_releases_a_request_and_the_count_returns_to_zero(stage, store):
    run(stage, store, "a", TOKENS[:6], 0)
    run(stage, store, "b", TOKENS[:3], 0)
    assert len(store) == 2
    assert store.free("a") is True and store.rids() == {"b"}
    assert store.free("b") is True and len(store) == 0


def test_freeing_an_unknown_or_already_freed_request_does_nothing(stage, store):
    assert store.free("ghost") is False
    run(stage, store, "a", TOKENS[:6], 0)
    assert store.free("a") is True and store.free("a") is False and len(store) == 0


def test_a_freed_request_cannot_continue_but_can_start_again(stage, store):
    run(stage, store, "a", TOKENS[:6], 0)
    store.free("a")
    with pytest.raises(Gap):
        store.acquire("a", 6)
    assert len(store.acquire("a", 0)) == 0


@pytest.mark.parametrize("positions", [1, 6, 10])
def test_the_memory_cost_per_position_matches_the_cache(stage, store, positions):
    run(stage, store, "a", TOKENS[:positions], 0)
    cfg = stage.config
    assert store.bytes_per_position == FIRST_LAYERS * 2 * cfg.num_kv_heads * cfg.head_dim * 4
    assert store.nbytes("a") == positions * store.bytes_per_position
    assert store.nbytes("ghost") == 0


def test_truncation_returns_the_memory(stage, store):
    run(stage, store, "a", TOKENS, 0)
    full = store.nbytes("a")
    assert full == len(TOKENS) * store.bytes_per_position
    store.acquire("a", 4)
    # the storage itself shrinks, not only the view of it
    assert store.nbytes("a") == 4 * store.bytes_per_position < full


def test_free_cache_frees_only_the_cache_it_is_given(stage, store):
    old = store.acquire("a", 0)
    store.free("a")
    new = store.acquire("a", 0)  # the request started again while an old step still ran
    assert store.free_cache("a", old) is False and "a" in store
    assert store.free_cache("a", new) is True and "a" not in store
    assert store.free_cache("a", new) is False


def test_a_freed_attempt_is_refused_and_a_newer_one_starts_again(stage, store):
    run(stage, store, "a", TOKENS[:6], 0)
    assert store.free("a", attempt=2) is True
    for attempt in (1, 2):  # a step of the freed attempt, or an older one, still in flight
        with pytest.raises(Stale) as refused:
            store.acquire("a", 0, attempt)
        assert (refused.value.attempt, refused.value.newer, refused.value.freed) == (
            attempt,
            2,
            True,
        )
    assert "a" not in store
    assert len(store.acquire("a", 0, attempt=3)) == 0 and "a" in store  # a replay


def test_a_free_for_an_older_attempt_keeps_the_newer_attempts_cache(stage, store):
    store.acquire("a", 0, attempt=1)
    store.acquire("a", 0, attempt=2)  # a replay has started from the beginning
    assert store.free("a", attempt=1) is False and "a" in store
    with pytest.raises(Stale):
        store.acquire("a", 0, attempt=1)
    assert store.free("a", attempt=2) is True and "a" not in store


def test_an_older_attempt_than_the_cache_has_used_is_refused(stage, store):
    run(stage, store, "a", TOKENS[:6], 0)
    store.acquire("a", 6, attempt=3)
    with pytest.raises(Stale) as refused:
        store.acquire("a", 6, attempt=2)
    assert (refused.value.newer, refused.value.freed) == (3, False)
    assert store.length("a") == 6 and store.attempt("a") == 3
    assert store.acquire("a", 0, attempt=4) is not None and store.attempt("a") == 4


def test_a_free_never_lowers_the_attempt_remembered(stage, store):
    store.free("a", attempt=3)
    store.free("a", attempt=1)  # a late Free for an older attempt
    with pytest.raises(Stale):
        store.acquire("a", 0, attempt=3)


def test_without_an_attempt_nothing_is_remembered(stage, store):
    store.free("a")
    store.free("b", attempt=1)
    assert len(store.acquire("a", 0, attempt=1)) == 0
    assert len(store.acquire("b", 0)) == 0  # a message with no attempt is not checked


def test_only_the_latest_frees_are_remembered(stage, store, monkeypatch):
    monkeypatch.setattr(kvstore, "MAX_FREED", 3)
    for rid in "abcd":
        store.free(rid, attempt=1)
    assert len(store.acquire("a", 0, attempt=1)) == 0  # forgotten: the oldest of four
    for rid in "bcd":
        with pytest.raises(Stale):
            store.acquire(rid, 0, attempt=1)
