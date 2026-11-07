"""Stage registration and epochs: the gateway side of Control.Register (R3.3, R4.4).

A stage registers with its index, layer range, model id, weights hash and address. The
gateway serves one model per run and checks each field against it, then either rejects
the stage with INVALID_ARGUMENT, naming the field, or accepts it and returns a new epoch.

Epochs come from one counter in the gateway, starting at 1 (0 means the gateway as a
sender). Every accepted registration gets the next value, so a stage that registers
again always receives a higher epoch than before, and no two incarnations share one.
The counter lives only in the gateway process; only workers are restarted.

The heartbeat stream and wiring this into the gateway are separate stories.
"""

import threading
from dataclasses import dataclass

import grpc

from pllm import config
from pllm.model import ModelConfig
from pllm.models import MODELS_DIR, PINNED
from pllm.proto import protocol_pb2, protocol_pb2_grpc

STAGES = (0, 1)


@dataclass(frozen=True)
class ExpectedModel:
    weights_sha256: bytes
    num_layers: int
    split_point: int

    def layer_range(self, stage: int) -> tuple[int, int]:
        k = self.split_point
        return (0, k) if stage == 0 else (k, self.num_layers)


def expected_model(model: str, split_point: int | None = None) -> ExpectedModel:
    """What the gateway expects for the one model it serves: the pinned weights hash, the
    layer count and the split point (the committed one unless a sweep passes another).

    Fails at gateway start, not at the first Register, if the model is not pinned or its
    config is not on this host.
    """
    if model not in PINNED:
        raise ValueError(f"{model!r} is not a pinned model")
    path = MODELS_DIR / model / "config.json"
    if not path.exists():
        raise ValueError(f"{path} is missing on the gateway host; download {model} first")
    n = ModelConfig.from_file(path).num_layers
    k = config.split_point(model) if split_point is None else split_point
    if not 1 <= k <= n - 1:
        raise ValueError(f"split point {k} for {model} outside 1 to {n - 1}")
    return ExpectedModel(bytes.fromhex(PINNED[model].files["model.safetensors"]), n, k)


class Registration(ValueError):
    """A registration that does not match what the gateway expects."""


@dataclass(frozen=True)
class Registered:
    stage: int
    epoch: int
    layer_start: int
    layer_end: int
    model_id: str
    addr: str


class Registry:
    """Checks registrations and hands out epochs. Safe to call from many threads."""

    def __init__(self, model_id: str, expected: ExpectedModel):
        self.model_id = model_id
        self.expected = expected
        self.stages: dict[int, Registered] = {}  # latest accepted per stage
        self._last_epoch = 0
        self._lock = threading.Lock()

    def register(self, req: protocol_pb2.RegisterReq) -> Registered:
        model = self.expected
        if req.model_id != self.model_id:
            raise Registration(
                f"model_id {req.model_id!r} does not match {self.model_id!r}, "
                "the model this gateway serves"
            )
        if req.weights_sha256 != model.weights_sha256:
            raise Registration(f"weights_sha256 does not match the pinned {req.model_id}")
        if req.stage not in STAGES:
            raise Registration(f"stage {req.stage} is not 0 or 1")
        want = model.layer_range(req.stage)
        if (req.layer_start, req.layer_end) != want:
            raise Registration(
                f"layer range {req.layer_start}:{req.layer_end} for stage {req.stage} "
                f"does not match the split, which needs {want[0]}:{want[1]}"
            )
        if not req.addr:
            raise Registration("addr is empty")
        with self._lock:
            self._last_epoch += 1
            entry = Registered(
                req.stage, self._last_epoch, *want, model_id=req.model_id, addr=req.addr
            )
            self.stages[req.stage] = entry
        return entry


class ControlService(protocol_pb2_grpc.ControlServicer):
    def __init__(self, registry: Registry):
        self.registry = registry

    def Register(self, request, context):
        try:
            entry = self.registry.register(request)
        except Registration as err:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(err))
        return protocol_pb2.RegisterResp(epoch=entry.epoch)


def register(
    channel: grpc.Channel,
    stage: int,
    layers: tuple[int, int],
    model_id: str,
    weights_sha256: str,
    addr: str,
    timeout: float = 10.0,
) -> int:
    """Register a stage with the gateway on channel and return its epoch.

    weights_sha256 is the hex digest of the model.safetensors the stage loaded, as
    computed by models.sha256_of, not the pinned value. Raises grpc.RpcError with
    INVALID_ARGUMENT if the gateway rejects it.
    """
    stub = protocol_pb2_grpc.ControlStub(channel)
    req = protocol_pb2.RegisterReq(
        stage=stage,
        layer_start=layers[0],
        layer_end=layers[1],
        model_id=model_id,
        weights_sha256=bytes.fromhex(weights_sha256),
        addr=addr,
    )
    return stub.Register(req, timeout=timeout).epoch
