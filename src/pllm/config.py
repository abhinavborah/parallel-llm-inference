"""Host addresses and ports for every process, read from one TOML file.

No process hard-codes an address. The file is config/hosts.toml unless the
PLLM_CONFIG environment variable names another one.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "hosts.toml"
ROLES = ("gateway", "stage0", "stage1")


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class HostConfig:
    network: str
    gateway: Endpoint
    stage0: Endpoint
    stage1: Endpoint


def config_path() -> Path:
    return Path(os.environ.get("PLLM_CONFIG", DEFAULT_PATH))


def load(path: str | Path | None = None) -> HostConfig:
    path = Path(path) if path is not None else config_path()
    with open(path, "rb") as f:
        data = tomllib.load(f)

    network = data.get("network")
    if not isinstance(network, str) or not network:
        raise ValueError(f"{path}: 'network' must name the network in use")

    endpoints = {role: _endpoint(data, role, path) for role in ROLES}
    seen = {}
    for role, endpoint in endpoints.items():
        if endpoint in seen:
            raise ValueError(f"{path}: [{role}] and [{seen[endpoint]}] share {endpoint.address}")
        seen[endpoint] = role

    return HostConfig(network=network, **endpoints)


def split_point(model: str, path: str | Path | None = None) -> int:
    """The split point k for model from the [split] table: stage 0 has layers 0 to k - 1.

    Read from the committed file, not from PLLM_CONFIG: each laptop points PLLM_CONFIG at
    its own copy for addresses, and both stages must use the same k.
    """
    path = Path(path) if path is not None else DEFAULT_PATH
    with open(path, "rb") as f:
        k = tomllib.load(f).get("split", {}).get(model)
    # bool is a subclass of int, so rule it out explicitly
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"{path}: [split] needs a positive whole number for {model!r}")
    return k


FRAMING_MARGIN = 64 * 1024  # bytes added to the largest activation message


def max_message_bytes(hidden_size: int, path: str | Path | None = None) -> int:
    """The receive limit for every gRPC server and channel (decision #20, R3.17).

    The largest message is a prefill of max_batch x max_prompt_tokens positions, each
    hidden_size fp32 values, plus a margin for the message fields. Read from the committed
    file, not from PLLM_CONFIG, so every process on both laptops uses the same bound.
    """
    path = Path(path) if path is not None else DEFAULT_PATH
    with open(path, "rb") as f:
        limits = tomllib.load(f).get("limits", {})
    values = []
    for key in ("max_batch", "max_prompt_tokens"):
        value = limits.get(key)
        # bool is a subclass of int, so rule it out explicitly
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{path}: [limits] needs a positive whole number for {key!r}")
        values.append(value)
    if hidden_size < 1:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    return values[0] * values[1] * hidden_size * 4 + FRAMING_MARGIN


def _endpoint(data: dict, role: str, path: Path) -> Endpoint:
    section = data.get(role)
    if not isinstance(section, dict):
        raise ValueError(f"{path}: missing [{role}] section")
    host = section.get("host")
    port = section.get("port")
    if not isinstance(host, str) or not host:
        raise ValueError(f"{path}: [{role}] host must be a non-empty string")
    # bool is a subclass of int, so rule it out explicitly
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError(f"{path}: [{role}] port must be an integer from 1 to 65535")
    return Endpoint(host, port)
