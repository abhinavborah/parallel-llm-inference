"""Pinned model files and a download that verifies every checksum.

Each model is pinned to a repository commit and each file to its sha256, so both
laptops load identical weights. Files go to models/<name>/, which git ignores.

Usage: uv run python -m pllm.models smollm2-135m [qwen3-0.6b]
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from pllm import runrecord
from pllm.config import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class PinnedModel:
    repo: str
    revision: str
    files: dict[str, str]  # file name to sha256


PINNED = {
    "smollm2-135m": PinnedModel(
        repo="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        files={
            "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
            "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
            "tokenizer_config.json": (
                "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"
            ),
            "model.safetensors": (
                "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
            ),
        },
    ),
    "qwen3-0.6b": PinnedModel(
        repo="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        files={
            "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            "tokenizer_config.json": (
                "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
            ),
            "model.safetensors": (
                "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
            ),
        },
    ),
}


class ChecksumError(Exception):
    pass


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: str | Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ChecksumError(f"{path}: sha256 is {actual}, expected {expected}")


def download(name: str, dest: str | Path | None = None) -> Path:
    """Download every pinned file of a model and verify it; a bad file is deleted."""
    model = PINNED[name]
    target = Path(dest) if dest is not None else MODELS_DIR / name
    for filename, expected in model.files.items():
        path = Path(
            hf_hub_download(model.repo, filename, revision=model.revision, local_dir=target)
        )
        try:
            verify(path, expected)
        except ChecksumError:
            path.unlink()
            raise
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download pinned model files.")
    parser.add_argument("models", nargs="+", choices=sorted(PINNED))
    args = parser.parse_args(argv)
    for name in args.models:
        with runrecord.record("download", model=name, revision=PINNED[name].revision) as rec:
            path = download(name)
            rec["files_verified"] = sorted(PINNED[name].files)
        print(f"{name}: all files verified in {path}")


if __name__ == "__main__":
    main()
