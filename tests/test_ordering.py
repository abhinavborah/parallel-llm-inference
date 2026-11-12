"""Step ordering on every hop (R3.7): a step that does not go up is counted and logged."""

import logging
import random

import grpc
import pytest
from test_chain import LAYERS, Chain
from test_gateway import WAIT, System
from test_model import tiny_reference

from pllm.ordering import StepOrder
from pllm.proto import protocol_pb2, protocol_pb2_grpc


def oracle(messages) -> list[bool]:
    """For each (rid, attempt, step), whether it is in order: its step is above every
    earlier step of the same rid and attempt. Kept apart from StepOrder on purpose: the
    whole history, no eviction, quadratic."""
    flags = []
    for i, (rid, attempt, step) in enumerate(messages):
        earlier = [s for r, a, s in messages[:i] if (r, a) == (rid, attempt)]
        flags.append(all(step > s for s in earlier))
    return flags


# -------------------------------------------------------------- the check on its own


def test_rising_steps_skipped_steps_and_a_new_attempt_are_in_order():
    order = StepOrder("test")
    for rid, attempt, step in [("a", 1, 0), ("a", 1, 1), ("b", 1, 0), ("a", 1, 4), ("a", 2, 0)]:
        assert order.check(rid, attempt, step)
    assert order.violations == 0


def test_a_repeated_or_lower_step_is_a_violation_and_is_logged(caplog):
    order = StepOrder("stage 0")
    order.check("a", 1, 3)
    with caplog.at_level(logging.WARNING, logger="pllm.ordering"):
        assert not order.check("a", 1, 3)
        assert not order.check("a", 1, 1)
    assert order.violations == 2
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [
        "stage 0: step 3 of 'a' attempt 1 arrived after step 3: out of order",
        "stage 0: step 1 of 'a' attempt 1 arrived after step 3: out of order",
    ]
    assert order.check("a", 1, 4)  # a violation does not lower the highest step


def test_an_older_attempt_is_judged_by_its_own_steps():
    order = StepOrder("test")
    order.check("a", 1, 5)
    order.check("a", 2, 0)
    assert order.check("a", 1, 6)  # late but in order for attempt 1
    assert not order.check("a", 1, 2)
    assert order.violations == 1


def test_the_pairs_kept_are_bounded_and_the_oldest_go_first():
    order = StepOrder("test", max_tracked=2)
    order.check("a", 1, 5)
    order.check("b", 1, 0)
    order.check("a", 1, 6)  # a is now the most recent
    order.check("c", 1, 0)  # pushes out b, not a
    assert len(order) == 2
    assert not order.check("a", 1, 6)
    assert order.check("b", 1, 0)  # forgotten, so it starts afresh


@pytest.mark.parametrize("seed", range(300))
def test_random_interleavings_count_exactly_the_out_of_order_steps(seed):
    rng = random.Random(seed)
    streams = {(rid, a): 0 for rid in "abcd" for a in (1, 2)}
    messages = []
    for _ in range(rng.randint(1, 60)):
        key = rng.choice(list(streams))
        if rng.random() < 0.7:
            streams[key] += rng.randint(1, 2)  # mostly up, sometimes skipping one
            step = streams[key]
        else:
            step = rng.randint(0, streams[key] + 1)  # anything, out of order or not
        messages.append((*key, step))
    order = StepOrder("test")
    got = [order.check(*m) for m in messages]
    assert got == oracle(messages)
    assert order.violations == got.count(False)


# -------------------------------------------------------------- through real stages


def interleaving(rng: random.Random) -> list[protocol_pb2.StepRequest]:
    """Requests of a prefill and single-token steps, interleaved at random with the order
    of each request kept, then disturbed: steps swapped, steps sent twice, and a request
    started again under attempt 2."""
    per_rid = {}
    for rid in "abcd":
        prompt = rng.randint(2, 5)
        msgs = [(rid, 1, 0, 0, [rng.randrange(1, 100) for _ in range(prompt)])]
        for step in range(1, rng.randint(2, 5)):
            msgs.append((rid, 1, step, prompt + step - 1, [rng.randrange(1, 100)]))
        per_rid[rid] = msgs
    if rng.random() < 0.5:  # a restart: the same prompt again under attempt 2, appended
        rid = rng.choice("abcd")
        per_rid[rid] += [(rid, 2, *m[2:]) for m in per_rid[rid][:2]]
    sequence = []
    queues = {rid: list(msgs) for rid, msgs in per_rid.items()}
    while queues:
        rid = rng.choice(list(queues))
        sequence.append(queues[rid].pop(0))
        if not queues[rid]:
            del queues[rid]
    for _ in range(rng.randint(1, 4)):
        i, j = sorted(rng.sample(range(len(sequence)), 2))
        if rng.random() < 0.5:
            sequence[i], sequence[j] = sequence[j], sequence[i]
        else:
            sequence.insert(j, sequence[i])  # sent again later
    return [
        protocol_pb2.StepRequest(
            rid=rid,
            attempt=attempt,
            step=step,
            start_pos=start,
            length=len(ids),
            tokens=protocol_pb2.TokenIds(ids=ids),
        )
        for rid, attempt, step, start, ids in sequence
    ]


@pytest.fixture
def chain(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    made = Chain(tmp_path, 2)
    yield made
    made.close()


def test_random_interleavings_through_both_stages_leave_no_violation_uncounted(chain):
    """Each stage's count equals the count over what that stage received: everything sent
    for Stage 0, and for Stage 1 everything it returned a result for (Stage 1 refuses
    nothing that Stage 0 forwarded, since both keep the same cache positions)."""
    total0 = total1 = 0
    for seed in range(12):
        rng = random.Random(seed)
        tag = f"s{seed}-"  # request ids of one seed never meet those of another
        sent = interleaving(rng)
        for message in sent:
            message.rid = tag + message.rid
            chain.outbox.put(message)
        chain.send(tag + "end", 0, 0, [1])  # handled last at both stages
        returned = []
        while (result := chain.result()).rid != tag + "end":
            returned.append((result.rid, result.attempt, result.step))
        want0 = oracle([(m.rid, m.attempt, m.step) for m in sent]).count(False)
        want1 = oracle(returned).count(False)
        total0, total1 = total0 + want0, total1 + want1
        assert chain.servicer0.order_violations == total0, seed
        assert chain.servicer1.order_violations == total1, seed
    assert total0 > 0 and total1 > 0  # the disturbances did reach both stages


def test_a_repeated_step_that_expires_while_it_waits_is_dropped_and_still_counted(tmp_path):
    """The check comes before the deadline rule. Stage 0 is slow with the first step, so
    the repeat waits behind it and its short deadline passes: it is dropped, never
    forwarded, and still counted as out of order."""
    tiny_reference(tmp_path, layers=LAYERS)
    run = System(tmp_path, 2, delay0=0.4)
    try:
        epoch0 = run.registry.stages[0].epoch

        def step(rid, budget_ms):
            return protocol_pb2.StepRequest(
                rid=rid,
                attempt=1,
                step=0,
                start_pos=0,
                length=2,
                target_epoch=epoch0,
                sender_epoch=0,
                tokens=protocol_pb2.TokenIds(ids=[5, 9]),
                budget_ms=budget_ms,
            )

        with grpc.insecure_channel(f"127.0.0.1:{run.ports[0]}") as channel:
            protocol_pb2_grpc.StageStub(channel).Steps(
                iter([step("r", 0), step("r", 100), step("other", 0)]), timeout=WAIT
            )
        assert run.servicer0.expired == 1  # the repeat was dropped by its deadline
        assert run.servicer0.order_violations == 1  # and still counted
        assert run.servicer1.order_violations == 0  # Stage 1 never saw the repeat
        assert run.servicer0.store.attempt("other") is not None  # the stream went on
    finally:
        run.close()
