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


SUPERVISOR_BASE_PORT = 50061  # stage 0 uses this, stage 1 the next port
DASHBOARD_PORT = 50080


def dashboard_endpoint(path: str | Path | None = None) -> Endpoint:
    """Where the dashboard listens (R9.2).

    A [dashboard] table sets it. Without one it listens on the gateway's host at port
    50080, so a hosts file written before the dashboard existed keeps working.
    """
    path = Path(path) if path is not None else config_path()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if "dashboard" in data:
        return _endpoint(data, "dashboard", path)
    gateway = _endpoint(data, "gateway", path)
    return Endpoint(gateway.host, DASHBOARD_PORT)


def supervisor_endpoint(stage: int, path: str | Path | None = None) -> Endpoint:
    """Where the supervisor of stage 0 or 1 takes fault-control requests.

    A [supervisor0] or [supervisor1] table sets it. Without one, the supervisor shares its
    stage's host on port 50061 or 50062, so a hosts file written before supervisors
    existed keeps working.
    """
    if stage not in (0, 1):
        raise ValueError(f"stage must be 0 or 1, got {stage}")
    path = Path(path) if path is not None else config_path()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    role = f"supervisor{stage}"
    if role in data:
        return _endpoint(data, role, path)
    fallback = _endpoint(data, f"stage{stage}", path)
    return Endpoint(fallback.host, SUPERVISOR_BASE_PORT + stage)


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


@dataclass(frozen=True)
class FlowLimits:
    admission_queue: int  # requests waiting for a running slot at the gateway
    credits: int  # steps in flight in the pipeline at once
    stage_inbox: int  # messages waiting at a stage


def flow_limits(path: str | Path | None = None) -> FlowLimits:
    """The flow-control bounds from [limits] (R5.1, R5.2). Read from the committed file, like
    the receive limit, so the gateway and both stages agree. stage_inbox must be at least
    credits: a stage then never fills its inbox while the gateway keeps to its credits."""
    path = Path(path) if path is not None else DEFAULT_PATH
    with open(path, "rb") as f:
        limits = tomllib.load(f).get("limits", {})
    values = {}
    for key, least in (("admission_queue", 0), ("credits", 1), ("stage_inbox", 1)):
        value = limits.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < least:
            raise ValueError(
                f"{path}: [limits] needs a whole number of at least {least} for {key!r}"
            )
        values[key] = value
    if values["stage_inbox"] < values["credits"]:
        raise ValueError(f"{path}: [limits] stage_inbox must be at least credits")
    return FlowLimits(**values)


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
