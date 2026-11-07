"""Stage registration over a real gRPC server on localhost (R3.3, R4.4)."""

from concurrent import futures

import grpc
import pytest

from pllm import config
from pllm.control import ControlService, ExpectedModel, Registry, expected_models, register
from pllm.models import MODELS_DIR, PINNED
from pllm.proto import protocol_pb2_grpc

MODEL = "smollm2-135m"
SHA = PINNED[MODEL].files["model.safetensors"]
EXPECTED = {MODEL: ExpectedModel(bytes.fromhex(SHA), num_layers=30, split_point=15)}
ADDR = "10.0.0.2:50051"


@pytest.fixture
def gateway():
    """A Control server on a free localhost port, and a channel to it."""
    registry = Registry(EXPECTED)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    protocol_pb2_grpc.add_ControlServicer_to_server(ControlService(registry), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        yield registry, channel
    server.stop(None)


def rejected(channel, **changes):
    args = {
        "stage": 0,
        "layers": (0, 15),
        "model_id": MODEL,
        "weights_sha256": SHA,
        "addr": ADDR,
    } | changes
    with pytest.raises(grpc.RpcError) as err:
        register(channel, **args)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    return err.value.details()


def test_an_accepted_stage_gets_an_epoch_and_is_recorded(gateway):
    registry, channel = gateway
    epoch = register(channel, 0, (0, 15), MODEL, SHA, ADDR)
    assert epoch >= 1
    entry = registry.stages[0]
    assert (entry.epoch, entry.layer_start, entry.layer_end, entry.addr) == (epoch, 0, 15, ADDR)
    assert register(channel, 1, (15, 30), MODEL, SHA, "10.0.0.1:50052") > epoch


def test_a_mismatched_model_or_weights_hash_is_rejected(gateway):
    _, channel = gateway
    assert "model_id 'qwen3-0.6b'" in rejected(channel, model_id="qwen3-0.6b")
    assert "weights_sha256" in rejected(channel, weights_sha256="00" * 32)


def test_a_wrong_stage_or_layer_range_is_rejected(gateway):
    registry, channel = gateway
    assert "stage 2" in rejected(channel, stage=2)
    # A stage started with another split point leaves a gap or an overlap.
    assert "needs 0:15" in rejected(channel, layers=(0, 14))
    assert "needs 15:30" in rejected(channel, stage=1, layers=(14, 30))
    assert "addr" in rejected(channel, addr="")
    assert registry.stages == {}


def test_a_stage_that_registers_again_gets_a_higher_epoch(gateway):
    registry, channel = gateway
    first = register(channel, 0, (0, 15), MODEL, SHA, ADDR)
    second = register(channel, 0, (0, 15), MODEL, SHA, ADDR)
    assert second > first
    assert registry.stages[0].epoch == second


def test_a_rejection_does_not_use_up_an_epoch(gateway):
    _, channel = gateway
    first = register(channel, 0, (0, 15), MODEL, SHA, ADDR)
    rejected(channel, weights_sha256="00" * 32)
    assert register(channel, 0, (0, 15), MODEL, SHA, ADDR) == first + 1


def test_concurrent_registrations_get_distinct_epochs(gateway):
    _, channel = gateway
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        epochs = list(
            pool.map(
                lambda i: register(channel, i % 2, ((0, 15), (15, 30))[i % 2], MODEL, SHA, ADDR),
                range(40),
            )
        )
    assert sorted(epochs) == list(range(1, 41))


@pytest.mark.skipif(
    not (MODELS_DIR / MODEL / "config.json").exists(), reason="model weights not downloaded"
)
def test_the_gateway_expects_the_pinned_hash_and_the_committed_split():
    expected = expected_models()[MODEL]
    assert expected.weights_sha256 == bytes.fromhex(SHA)
    assert expected.num_layers == 30
    assert expected.layer_range(0) == (0, config.split_point(MODEL))
    assert expected.layer_range(1) == (config.split_point(MODEL), 30)
