"""Admission by Host B's KV budget (R5.3).

Each gateway test runs a real gateway and two real stage servers on loopback, with the
budget set in bytes the way main sets it from [limits].
"""

import threading
import time

import grpc
import pytest
import torch
from test_gateway import IDS, WAIT, System, code_of, wait_for
from test_model import tiny_reference

from pllm import config, gateway
from pllm.kvstore import bytes_per_position
from pllm.model import ModelConfig
from pllm.split import Stage, stage_weight_bytes

LAYERS = 4
K = 2
FAST = 0.5  # seconds within which a refusal must come back
TOKENS = 3


@pytest.fixture
def system(tmp_path):
    made = []

    tiny_reference(tmp_path, layers=LAYERS)
    stage0 = ModelConfig.from_file(tmp_path / "config.json")

    def make(running=None, **kwargs):
        if running is not None:
            kwargs["max_workers"] = gateway.RESERVED_THREADS + running
        made.append(System(tmp_path, K, **kwargs))
        made[-1].gateway.kv_per_position = bytes_per_position(stage0, K)  # as main sets it
        return made[-1]

    yield make
    for each in made:
        each.close()


def need(run, max_tokens=TOKENS):
    """The bytes one request of the test prompt and max_tokens reserves."""
    return (len(IDS) + max_tokens) * run.gateway.kv_per_position


def tokens(run, **kwargs):
    return [(t.token, t.top_logit) for t in run.call(**kwargs)]


def in_thread(fn):
    out = []
    thread = threading.Thread(target=lambda: out.append(_outcome(fn)))
    thread.start()
    return thread, out


def _outcome(fn):
    try:
        return fn()
    except grpc.RpcError as err:
        return err.code()


def idle(run):
    """Nothing left reserved, running or queued once every request has ended."""
    g = run.gateway
    wait_for(lambda: g._kv_reserved == 0 and g._running == 0, "the reservation given back")
    assert not g.requests and not g._waiting


# -------------------------------------------------------------- boundary at the budget


def test_a_request_exactly_at_the_budget_is_admitted(system):
    expected = tokens(system(), max_tokens=TOKENS)
    run = system()
    run.gateway.kv_budget = need(run)
    assert tokens(run, max_tokens=TOKENS) == expected
    assert run.gateway.kv_rejected == 0
    idle(run)


def test_a_request_one_byte_over_the_budget_is_refused_at_once_and_counted(system):
    run = system()
    run.gateway.kv_budget = need(run) - 1
    began = time.monotonic()
    code, details = code_of(run.call(max_tokens=TOKENS))
    assert time.monotonic() - began < FAST
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED and "KV cache" in details
    assert run.gateway.kv_rejected == 1 and run.gateway.outcomes["rejected"] == 1
    assert len(run.servicer0.store) == 0 and len(run.servicer1.store) == 0  # never sent
    idle(run)


def test_two_requests_fill_a_budget_of_two_and_a_third_is_refused(system):
    run = system(delay1=0.3)
    one = need(run)
    run.gateway.kv_budget = 2 * one
    first = [in_thread(lambda: tokens(run, max_tokens=TOKENS)) for _ in range(2)]
    wait_for(lambda: run.gateway._kv_reserved == 2 * one, "two requests reserved")
    code, _ = code_of(run.call(max_tokens=TOKENS))
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED
    for thread, out in first:
        thread.join(WAIT)
        assert isinstance(out[0], list) and len(out[0]) == TOKENS
    assert run.gateway.kv_rejected == 1
    idle(run)
    assert len(tokens(run, max_tokens=TOKENS)) == TOKENS  # the budget is free again


def test_a_short_request_fits_beside_a_long_one_where_a_second_long_one_does_not(system):
    run = system(delay1=0.3)
    long, short = need(run, 8), need(run, 2)
    run.gateway.kv_budget = long + short
    thread, out = in_thread(lambda: tokens(run, max_tokens=8))
    wait_for(lambda: run.gateway._kv_reserved == long, "the long request reserved")
    code, _ = code_of(run.call(max_tokens=8))
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert len(tokens(run, max_tokens=2)) == 2
    thread.join(WAIT)
    assert len(out[0]) == 8
    idle(run)


def test_a_queued_request_holds_its_reservation(system):
    """One running slot and a queue of two: the second request waits in the queue with its
    bytes reserved, so the third is refused for KV, not for a full queue."""
    run = system(running=1, max_queued=2, delay1=0.3)
    one = need(run)
    run.gateway.kv_budget = 2 * one
    started = [in_thread(lambda: tokens(run, max_tokens=TOKENS)) for _ in range(2)]
    wait_for(lambda: len(run.gateway._waiting) == 1, "one running and one queued")
    assert run.gateway._kv_reserved == 2 * one
    code, details = code_of(run.call(max_tokens=TOKENS))
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED and "KV cache" in details
    for thread, out in started:
        thread.join(WAIT)
        assert len(out[0]) == TOKENS
    idle(run)


def test_without_a_budget_nothing_is_reserved(system):
    run = system()
    assert run.gateway.kv_budget is None
    assert len(tokens(run, max_tokens=TOKENS)) == TOKENS
    assert run.gateway._kv_reserved == 0 and run.gateway.kv_rejected == 0


def test_the_dashboard_state_shows_the_budget(system):
    run = system()
    run.gateway.kv_budget = need(run)
    state = run.gateway.state()
    assert state["kv_budget"] == need(run) and state["kv_reserved"] == 0
    assert state["kv_rejected"] == 0


# -------------------------------------------------------------- every end gives it back


def test_a_request_past_its_deadline_gives_its_reservation_back(system):
    run = system(delay1=2.0)
    run.gateway.kv_budget = need(run)
    code, _ = code_of(run.call(max_tokens=TOKENS, timeout_ms=300))
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED
    idle(run)


def test_a_cancelled_request_gives_its_reservation_back(system):
    run = system(delay1=0.3)
    run.gateway.kv_budget = need(run, 8)
    call = run.call(max_tokens=8)
    next(call)  # running, with its bytes reserved
    assert run.gateway._kv_reserved == need(run, 8)
    call.cancel()
    idle(run)


def test_a_request_refused_by_a_full_queue_gives_its_reservation_back(system):
    run = system(running=1, max_queued=0, delay1=0.3)
    one = need(run)
    run.gateway.kv_budget = 2 * one
    thread, out = in_thread(lambda: tokens(run, max_tokens=TOKENS))
    wait_for(lambda: run.gateway._running == 1, "the slot taken")
    code, details = code_of(run.call(max_tokens=TOKENS))
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED and "waiting" in details
    assert run.gateway._kv_reserved == one and run.gateway.kv_rejected == 0
    thread.join(WAIT)
    assert len(out[0]) == TOKENS
    idle(run)


def test_a_request_whose_stage_is_down_gives_its_reservation_back(system):
    run = system()
    run.gateway.kv_budget = need(run)
    run.server1.stop(0).wait()
    code, _ = code_of(run.call(max_tokens=TOKENS))
    assert code == grpc.StatusCode.UNAVAILABLE
    idle(run)


def test_a_request_refused_before_reserving_gives_nothing_back(system):
    run = system()
    run.gateway.kv_budget = need(run)
    code, _ = code_of(run.call(max_tokens=0))
    assert code == grpc.StatusCode.INVALID_ARGUMENT
    assert run.gateway._kv_reserved == 0 and run.gateway.kv_rejected == 0


# -------------------------------------------------------------- where the numbers come from


def test_the_committed_host_b_budget_is_the_agreed_one():
    budget = config.host_b_budget()
    assert budget.worker_bytes == int(0.7 * 2**30)
    assert budget.runtime_bytes == 196 * 2**20


@pytest.mark.parametrize(
    "lines, message",
    [
        ("host_b_runtime_mib = 196\n", "host_b_budget_gib"),
        ("host_b_budget_gib = 0\nhost_b_runtime_mib = 196\n", "host_b_budget_gib"),
        ("host_b_budget_gib = 0.7\nhost_b_runtime_mib = -1\n", "host_b_runtime_mib"),
        ("host_b_budget_gib = true\nhost_b_runtime_mib = 196\n", "host_b_budget_gib"),
        ("host_b_budget_gib = inf\nhost_b_runtime_mib = 196\n", "host_b_budget_gib"),
        ("host_b_budget_gib = 0.1\nhost_b_runtime_mib = 196\n", "must be below"),
    ],
)
def test_bad_host_b_budgets_are_rejected(tmp_path, lines, message):
    path = tmp_path / "hosts.toml"
    path.write_text("[limits]\n" + lines)
    with pytest.raises(ValueError, match=message):
        config.host_b_budget(path)


def test_stage_weight_bytes_is_what_the_stage_holds(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    held = Stage.load(tmp_path, 0, K, mapped_embedding=True)
    tensors = [t for layer in held.layers for t in vars(layer).values() if t is not None]
    assert all(isinstance(t, torch.Tensor) for t in tensors)
    assert stage_weight_bytes(tmp_path, 0, K) == sum(t.nbytes for t in tensors)
    assert stage_weight_bytes(tmp_path, K, LAYERS) > 0
