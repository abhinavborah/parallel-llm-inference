"""Regenerate the protocol code: uv run python -m pllm.proto.generate

Needs grpcio-tools from the dev group. The generated files are committed;
tests/test_protocol.py fails if they drift from protocol.proto.
"""

import sys
from pathlib import Path

from grpc_tools import protoc

SRC = Path(__file__).resolve().parents[2]
PROTO = Path("pllm") / "proto" / "protocol.proto"


def generate(out_dir: Path = SRC) -> None:
    """Write protocol_pb2.py, .pyi and _pb2_grpc.py under out_dir/pllm/proto/."""
    # -I src makes the generated import package-qualified (from pllm.proto import ...)
    args = [
        "grpc_tools.protoc",
        f"-I{SRC}",
        f"--python_out={out_dir}",
        f"--pyi_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        PROTO.as_posix(),  # relative to -I, so the recorded file name has no OS separators
    ]
    if protoc.main(args) != 0:
        raise RuntimeError("protoc failed")


if __name__ == "__main__":
    generate(Path(sys.argv[1]) if len(sys.argv) > 1 else SRC)
