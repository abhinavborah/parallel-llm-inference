"""Recovery time in four parts (R6.12): detection, respawn, weight load and replay prefill,
with the tokens recomputed. A new stage reports its weight load time with Register."""

import pytest
from test_detection import DIGEST, LAYERS, K, Plane

from pllm import control
from pllm.proto import protocol_pb2, protocol_pb2_grpc


@pytest.fixture
def plane(tmp_path):
    made = Plane(tmp_path)
    yield made
    made.close()


def registered(plane, epoch):
    return [e for e in plane.detector.events() if e.get("new_epoch") == epoch][0]


# ------------------------------------------------------------------ the weight load time


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, None),
        ((), None),
        (((control.LOAD_KEY, "123456789"),), 123456789),
        (((control.LOAD_KEY, b"42"),), 42),
        (((control.LOAD_KEY, "-5"),), None),
        (((control.LOAD_KEY, "1.5"),), None),
        (((control.LOAD_KEY, ""),), None),
        (((control.LOAD_KEY, "٣"),), None),  # a digit, but not an ASCII one
        ((("other", "7"),), None),
    ],
)
def test_the_load_time_is_a_plain_integer_or_none(metadata, expected):
    assert control.load_ns_of(metadata) == expected


def test_a_stage_reports_its_load_time_with_register(plane):
    epoch = control.register(
        plane.channel, 1, (K, LAYERS), "tiny", DIGEST.hex(), "127.0.0.1:1", load_ns=987654321
    )
    assert registered(plane, epoch)["load_ns"] == 987654321


@pytest.mark.parametrize("value", [None, "soon", "-1"])
def test_a_register_with_no_or_a_bad_load_time_still_registers(plane, value):
    """An older worker sends no load time; a bad value is dropped, never logged."""
    req = protocol_pb2.RegisterReq(
        stage=1, layer_start=K, layer_end=LAYERS, model_id="tiny",
        weights_sha256=DIGEST, addr="127.0.0.1:1",
    )  # fmt: skip
    metadata = None if value is None else ((control.LOAD_KEY, value),)
    stub = protocol_pb2_grpc.ControlStub(plane.channel)
    epoch = stub.Register(req, timeout=10, metadata=metadata).epoch
    assert registered(plane, epoch)["load_ns"] is None
