import hashlib
import sys
from pathlib import Path

import pytest
import torch

from pllm.proto import SCHEMA_VERSION, protocol_pb2, protocol_pb2_grpc
from pllm.proto.generate import generate

PROTO_DIR = Path(protocol_pb2.__file__).parent

# Changing protocol.proto must bump SCHEMA_VERSION and this pin together.
SCHEMA_PINS = {
    1: "5c0c7386d0c6fdc28a7967e30c683138d2f6124aa905f9c6e467e5ac0641dd2f",
    2: "dac7535c5753a470a65725e0d1dbdb14993622da7ee8ab8496ac28efd0e83a5c",
}


def lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def sample_tensor() -> torch.Tensor:
    t = torch.randn(1, 3, 576, generator=torch.Generator().manual_seed(0))
    t[0, 0, :3] = torch.tensor([-0.0, float("inf"), float("-inf")])
    return t


def tensor_msg() -> protocol_pb2.Tensor:
    t = sample_tensor().contiguous()
    data = bytes(t.untyped_storage())  # native order; little-endian on every supported host
    return protocol_pb2.Tensor(dtype="float32", shape=list(t.shape), data=data)


def step(**payload) -> protocol_pb2.StepRequest:
    return protocol_pb2.StepRequest(
        rid="r1",
        attempt=2,
        step=3,
        start_pos=7,
        length=1,
        target_epoch=5,
        sender_epoch=4,
        replay=True,
        budget_ms=30_000,
        **payload,
    )


# Every field set to a non-default value: proto3 drops defaults on the wire.
MESSAGES = [
    protocol_pb2.Empty(),
    protocol_pb2.RegisterReq(
        stage=1,
        layer_start=15,
        layer_end=30,
        model_id="HuggingFaceTB/SmolLM2-135M",
        weights_sha256=bytes(range(32)),
        addr="192.168.1.20:50051",
    ),
    protocol_pb2.RegisterResp(epoch=2**40),
    protocol_pb2.Beat(stage=1, epoch=3, inbox_depth=4, cached_requests=5),
    protocol_pb2.Nack(stage=1, epoch=3, rid="r1", attempt=2, step=9, reason="GAP"),
    protocol_pb2.StageMsg(beat=protocol_pb2.Beat(stage=1, epoch=3)),
    protocol_pb2.StageMsg(nack=protocol_pb2.Nack(rid="r1", reason="WRONG_EPOCH")),
    protocol_pb2.RouteUpdate(next_stage=1, addr="10.0.0.2:50052", epoch=6),
    protocol_pb2.Free(rid="r1", attempt=2),
    protocol_pb2.ControlMsg(route_update=protocol_pb2.RouteUpdate(next_stage=1, epoch=6)),
    protocol_pb2.ControlMsg(free=protocol_pb2.Free(rid="r1", attempt=2)),
    tensor_msg(),
    protocol_pb2.TokenIds(ids=[1, 2, 49151]),
    step(tokens=protocol_pb2.TokenIds(ids=[1, 2, 3])),
    step(activations=tensor_msg()),
    protocol_pb2.StepResult(rid="r1", attempt=2, step=3, epoch=4, token=42, top_logit=-1.5),
    protocol_pb2.GenerateReq(prompt="Hello", max_tokens=32, timeout_ms=30_000),
    protocol_pb2.Token(rid="r1", step=3, token=42, text=" world", top_logit=-1.5),
]


@pytest.mark.parametrize("msg", MESSAGES, ids=lambda m: type(m).__name__)
def test_round_trip_is_lossless(msg):
    decoded = type(msg).FromString(msg.SerializeToString())
    assert decoded == msg
    for oneof in msg.DESCRIPTOR.oneofs_by_name:
        assert decoded.WhichOneof(oneof) == msg.WhichOneof(oneof)


def test_every_message_type_is_covered():
    covered = {type(m).DESCRIPTOR.name for m in MESSAGES}
    assert covered == set(protocol_pb2.DESCRIPTOR.message_types_by_name)


def test_tensor_keeps_dtype_shape_and_bits():
    assert sys.byteorder == "little"
    sent = sample_tensor()
    wire = step(activations=tensor_msg()).SerializeToString()

    t = protocol_pb2.StepRequest.FromString(wire).activations
    assert t.dtype == "float32"
    got = torch.frombuffer(bytearray(t.data), dtype=torch.float32).reshape(tuple(t.shape))
    assert got.shape == sent.shape
    assert torch.equal(got.view(torch.int32), sent.view(torch.int32))


def test_services_match_the_agreed_rpcs():
    services = {
        name: {m.name: (m.client_streaming, m.server_streaming) for m in s.methods}
        for name, s in protocol_pb2.DESCRIPTOR.services_by_name.items()
    }
    assert services == {
        "Control": {"Register": (False, False), "Heartbeat": (True, True)},
        "Stage": {"Steps": (True, False)},
        "Results": {"Push": (True, False)},
        "Frontend": {"Generate": (False, True)},
    }
    for name in services:
        assert hasattr(protocol_pb2_grpc, f"{name}Stub")
        assert hasattr(protocol_pb2_grpc, f"{name}Servicer")


def test_generated_code_matches_proto(tmp_path):
    generate(tmp_path)
    for name in ("protocol_pb2.py", "protocol_pb2.pyi", "protocol_pb2_grpc.py"):
        fresh = tmp_path / "pllm" / "proto" / name
        assert lf_bytes(fresh) == lf_bytes(PROTO_DIR / name), f"{name} is stale, regenerate"


def test_schema_change_bumps_version():
    digest = hashlib.sha256(lf_bytes(PROTO_DIR / "protocol.proto")).hexdigest()
    assert SCHEMA_PINS.get(SCHEMA_VERSION) == digest, "bump SCHEMA_VERSION and its pin"
