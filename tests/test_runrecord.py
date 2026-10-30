import json
import shutil
import subprocess

import pytest

from pllm import config, runrecord
from pllm.config import REPO_ROOT

FIELDS = {
    "script",
    "started",
    "ended",
    "status",
    "commit",
    "model",
    "revision",
    "split",
    "threads",
    "host",
    "os",
    "python",
    "network",
    "rtt_ms",
    "available_memory_bytes",
    "power",
}


def only_record(runs_dir):
    files = list(runs_dir.glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_record_holds_every_field(tmp_path):
    with runrecord.record(
        "probe",
        model="smollm2-135m",
        revision="abc",
        split=15,
        threads=4,
        network="single-host",
        rtt_ms=0.4,
        runs_dir=tmp_path,
    ) as rec:
        rec["tokens"] = 7

    data = only_record(tmp_path)
    assert FIELDS <= data.keys()
    assert data["status"] == "ok"
    assert data["split"] == 15
    assert data["tokens"] == 7
    assert data["available_memory_bytes"] > 0


def test_network_defaults_to_the_hosts_file(tmp_path, monkeypatch):
    hosts = tmp_path / "hosts.toml"
    text = config.DEFAULT_PATH.read_text(encoding="utf-8")
    hosts.write_text(text.replace('"single-host"', '"test-lan"'), encoding="utf-8")
    monkeypatch.setenv("PLLM_CONFIG", str(hosts))

    with runrecord.record("probe", runs_dir=tmp_path / "runs"):
        pass
    assert only_record(tmp_path / "runs")["network"] == "test-lan"


def test_record_is_written_with_error_status_when_the_run_fails(tmp_path):
    with pytest.raises(RuntimeError):
        with runrecord.record("probe", runs_dir=tmp_path):
            raise RuntimeError("boom")
    assert only_record(tmp_path)["status"] == "error"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_commit_hash_names_head():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    assert runrecord.commit_hash().removesuffix("-dirty") == head


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize(
    "path",
    ["models/smollm2-135m/model.safetensors", "runs/x-probe.json", "config/hosts.local.toml"],
)
def test_git_ignores_weights_run_outputs_and_local_hosts(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT)
    assert result.returncode == 0
