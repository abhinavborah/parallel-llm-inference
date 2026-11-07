import pytest

from pllm import prompts
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
