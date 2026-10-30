"""Run records: one JSON file per script run.

Each record ties a run to the code, model, machine and network that produced it,
so every reported number can be traced. Records go to runs/, which git ignores.
"""

import json
import platform
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import psutil

from pllm import config
from pllm.config import REPO_ROOT

RUNS_DIR = REPO_ROOT / "runs"


def commit_hash() -> str | None:
    """HEAD of this repository, with a -dirty suffix if the work tree has changes."""
    try:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


def power_state() -> dict | None:
    """Battery charge and charger state, or None on a machine without a battery."""
    battery = psutil.sensors_battery()
    if battery is None:
        return None
    return {"plugged_in": battery.power_plugged, "percent": battery.percent}


@contextmanager
def record(
    script: str,
    *,
    model: str | None = None,
    revision: str | None = None,
    split: int | None = None,
    threads: int | None = None,
    network: str | None = None,
    rtt_ms: float | None = None,
    runs_dir: str | Path | None = None,
) -> Iterator[dict]:
    """Write a run record when the block ends, with status ok or error.

    The network defaults to the one named in the hosts file; a missing or invalid
    hosts file stops the script before it runs. The yielded dict can be extended
    with results before the block ends.
    """
    if network is None:
        network = config.load().network
    started = datetime.now(UTC)
    rec = {
        "script": script,
        "started": started.isoformat(),
        "commit": commit_hash(),
        "model": model,
        "revision": revision,
        "split": split,
        "threads": threads,
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "network": network,
        "rtt_ms": rtt_ms,
        "available_memory_bytes": psutil.virtual_memory().available,
        "power": power_state(),
    }
    status = "error"
    try:
        yield rec
        status = "ok"
    finally:
        rec["status"] = status
        rec["ended"] = datetime.now(UTC).isoformat()
        out_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{started:%Y%m%dT%H%M%S%fZ}-{script}.json"
        path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()
