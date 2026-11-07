import json

import pytest

from pllm import prompts, runrecord
from pllm.models import MODELS_DIR, PINNED, ChecksumError


@pytest.fixture(scope="module")
def data():
    return prompts.load()


def test_load_rejects_a_changed_file(tmp_path):
    raw = bytearray(prompts.PROMPTS_PATH.read_bytes())
    raw[-3] ^= 1
    changed = tmp_path / "prompt_set.json"
    changed.write_bytes(raw)
    with pytest.raises(ChecksumError):
        prompts.load(changed)


def test_tokenizers_and_revisions_match_the_pins(data):
    assert data["add_special_tokens"] is False
    assert data["lengths"] == list(prompts.LENGTHS) == [16, 128, 512]
    assert data["tokenizers"] == {
        spec.files["tokenizer.json"]: {"model": name, "repo": spec.repo, "revision": spec.revision}
        for name, spec in PINNED.items()
    }
    assert len(data["tokenizers"]) == len(PINNED), "pinned tokenizers must differ"


@pytest.mark.parametrize("model", sorted(PINNED))
def test_fifty_prompts_per_length_nested(data, model):
    by_length = {n: prompts.prompts(model, n, data) for n in prompts.LENGTHS}
    for n, ids in by_length.items():
        assert len(ids) == 50
        assert all(len(i) == n for i in ids)
    for short, long in zip(by_length[16], by_length[512], strict=True):
        assert long[:16] == short
    assert len({tuple(i) for i in by_length[16]}) == 50


def test_unknown_length_is_rejected(data):
    with pytest.raises(ValueError):
        prompts.prompts("smollm2-135m", 64, data)


@pytest.mark.parametrize("model", sorted(PINNED))
def test_every_cut_round_trips(data, model):
    if not (MODELS_DIR / model / "tokenizer.json").exists():
        pytest.skip("tokenizer not downloaded")
    tokenizer = prompts.load_tokenizer(model)
    for n in prompts.LENGTHS:
        for i, ids in enumerate(prompts.prompts(model, n, data)):
            assert prompts.round_trips(tokenizer, ids), (model, n, i)
            text = data["passages"][i]["text"]
            assert text.startswith(tokenizer.decode(ids)), (model, n, i)


class CharTokenizer:
    """One id per character; offset makes two tokenizers give different ids."""

    def __init__(self, offset=0, broken=False):
        self.offset, self.broken = offset, broken

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) + self.offset for c in text]
        return type("Encoding", (), {"ids": ids})()

    def decode(self, ids):
        text = "".join(chr(i - self.offset) for i in ids)
        return text + "x" if self.broken else text  # broken: never round-trips


def synthetic_book(paragraphs=110):
    body = "\n\n".join(
        (prompts.FIRST_PARAGRAPH if i == 0 else f"Paragraph {i}") + " " + "word " * 120
        for i in range(paragraphs)
    )
    return f"header\n*** START OF THE BOOK ***\nTitle page\n\n{body}\n*** END OF THE BOOK ***\n"


def fake_tokenizers(broken=False):
    keys = [spec.files["tokenizer.json"] for spec in PINNED.values()]
    return {keys[0]: CharTokenizer(), keys[1]: CharTokenizer(offset=7, broken=broken)}


def test_build_is_deterministic_and_dump_round_trips():
    raw = synthetic_book()
    first = prompts.build(raw, fake_tokenizers())
    assert prompts.dump(first) == prompts.dump(prompts.build(raw, fake_tokenizers()))
    assert json.loads(prompts.dump(first)) == first
    passages = first["passages"]
    assert len(passages) == prompts.PASSAGES
    assert all(len(ids) == max(prompts.LENGTHS) for p in passages for ids in p["ids"].values())
    assert passages[0]["text"].startswith(prompts.FIRST_PARAGRAPH)  # front matter dropped
    # one passage per line keeps a diff readable
    lines = prompts.dump(first).splitlines()
    assert sum(line.startswith('{"text"') for line in lines) == prompts.PASSAGES


def test_build_fails_when_a_slot_has_no_round_trip():
    with pytest.raises(ValueError, match="no passage in slot 0"):
        prompts.build(synthetic_book(), fake_tokenizers(broken=True))


def test_a_rebuild_with_the_wrong_book_writes_an_error_record(tmp_path, monkeypatch):
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    book = tmp_path / "book.txt"
    book.write_text(synthetic_book(), encoding="utf-8")
    with pytest.raises(ChecksumError):
        prompts.main([str(book)])
    (record_file,) = (tmp_path / "runs").glob("*-prompt_set.json")
    rec = json.loads(record_file.read_text(encoding="utf-8"))
    assert rec["status"] == "error" and rec["book_sha256"] == prompts.BOOK["sha256"]
