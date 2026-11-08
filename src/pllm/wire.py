"""How activations travel on the wire (R3.4) and the gRPC options every process shares.

A tensor is sent as its dtype name, its shape and its raw little-endian bytes in row-major
order, never as JSON or text. Only float32 is sent: both hosts compute in fp32, and the
bytes are copied unchanged, so a received tensor is bitwise equal to the one sent.

Every gRPC server and channel takes its options from channel_options (decision #20), so
the receive limit comes from config and not from gRPC's 4 MiB default.
"""

import math
import sys

import torch

from pllm import config
from pllm.proto import protocol_pb2

DTYPE = "float32"
# frombuffer reads native order, and the wire is little-endian; both laptops are.
assert sys.byteorder == "little", "the wire format assumes a little-endian host"


class BadTensor(ValueError):
    """A tensor on the wire that does not describe what it carries."""


def tensor_to_proto(tensor: torch.Tensor) -> protocol_pb2.Tensor:
    if tensor.dtype != torch.float32:
        raise BadTensor(f"only {DTYPE} is sent on the wire, got {tensor.dtype}")
    flat = tensor.detach().contiguous().reshape(-1)
    return protocol_pb2.Tensor(dtype=DTYPE, shape=list(tensor.shape), data=flat.numpy().tobytes())


def tensor_from_proto(message: protocol_pb2.Tensor) -> torch.Tensor:
    if message.dtype != DTYPE:
        raise BadTensor(f"dtype {message.dtype!r} is not {DTYPE!r}")
    if not message.shape or any(dim < 1 for dim in message.shape):
        raise BadTensor(f"shape {list(message.shape)} must have only positive dimensions")
    count = math.prod(message.shape)
    if len(message.data) != count * 4:
        raise BadTensor(
            f"shape {list(message.shape)} needs {count * 4} bytes, got {len(message.data)}"
        )
    # bytearray: frombuffer shares memory, and the bytes of a message are read-only.
    flat = torch.frombuffer(bytearray(message.data), dtype=torch.float32)
    return flat.view(*message.shape)


def channel_options(hidden_size: int) -> list[tuple[str, int]]:
    """Options for every gRPC server and channel: the receive limit from config."""
    return [("grpc.max_receive_message_length", config.max_message_bytes(hidden_size))]
