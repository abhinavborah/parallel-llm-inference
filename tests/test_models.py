import hashlib
import json
import re

import pytest

from pllm import models, runrecord

CONTENT = {"config.json": b"{}", "model.safetensors": b"weights"}
FAKE = models.PinnedModel(
    repo="org/fake",
    revision="0" * 40,
    files={name: hashlib.sha256(data).hexdigest() for name, data in CONTENT.items()},
)


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace the network download with one that writes CONTENT; returns the call log."""
    calls = []

    def hf_hub_download(repo, filename, *, revision, local_dir):
        calls.append((repo, filename, revision))
        path = local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(CONTENT[filename])
        return str(path)

    monkeypatch.setattr(models, "hf_hub_download", hf_hub_download)
    monkeypatch.setitem(models.PINNED, "fake", FAKE)
    return calls


def test_every_pin_is_a_commit_and_sha256():
    for model in models.PINNED.values():
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision)
        assert "model.safetensors" in model.files
        for digest in model.files.values():
            assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_verify_accepts_matching_and_rejects_other_content(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    models.verify(path, hashlib.sha256(b"abc").hexdigest())
    with pytest.raises(models.ChecksumError):
        models.verify(path, hashlib.sha256(b"abd").hexdigest())


def test_download_fetches_every_file_at_the_pinned_revision(tmp_path, fake_hub):
    target = models.download("fake", tmp_path)
    assert sorted(f for _, f, _ in fake_hub) == sorted(CONTENT)
    assert {(r, rev) for r, _, rev in fake_hub} == {("org/fake", "0" * 40)}
    assert (target / "model.safetensors").read_bytes() == b"weights"


def test_download_deletes_a_file_with_a_wrong_checksum(tmp_path, fake_hub, monkeypatch):
    monkeypatch.setitem(CONTENT, "model.safetensors", b"tampered")
    with pytest.raises(models.ChecksumError):
        models.download("fake", tmp_path)
    assert not (tmp_path / "model.safetensors").exists()


def test_main_downloads_and_writes_a_run_record(tmp_path, fake_hub, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    models.main(["fake"])

    (record_file,) = (tmp_path / "runs").glob("*-download.json")
    rec = json.loads(record_file.read_text(encoding="utf-8"))
    assert rec["status"] == "ok"
    assert rec["model"] == "fake"
    assert rec["revision"] == "0" * 40
    assert rec["files_verified"] == sorted(CONTENT)
