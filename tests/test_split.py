"""The in-process two-stage split against the single-process model (R2.5).

Equality here is bitwise (torch.equal), on one host, fp32, batch 1.
"""

import re

import pytest
import torch
from safetensors import safe_open
from test_model import tiny_reference

from pllm import split
from pllm.model import Model
from pllm.models import MODELS_DIR, PINNED, verify
from pllm.reference import generate
from pllm.split import EMBED, LM_HEAD, NORM, InProcessSplit, Stage

LAYERS = 5
SMOLLM2_DIR = MODELS_DIR / "smollm2-135m"


def drive(model, tokens):
    """Prefill, decode, then truncate into the decoded part and append; return all logits."""
    cache = model.new_cache()
    steps = [model.forward(tokens[:12], 0, cache)]
    for pos in range(12, 18):
        steps.append(model.forward(tokens[pos : pos + 1], pos, cache))
    steps.append(model.forward(tokens[3:7], 14, cache))
    return steps


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("qwen3", [False, True])
@pytest.mark.parametrize("tie", [True, False])
def test_every_split_point_equals_the_single_process_model(tmp_path, tie, qwen3):
    _, model = tiny_reference(tmp_path, tie, qwen3, layers=LAYERS)
    tokens = torch.randint(0, 128, (20,), generator=torch.Generator().manual_seed(2))
    expected = drive(model, tokens)
    for k in range(1, LAYERS):
        assert same(drive(InProcessSplit.load(tmp_path, k), tokens), expected), f"k={k}"


@pytest.mark.parametrize("tie", [True, False])
def test_generation_through_the_split_equals_the_single_process_run(tmp_path, tie):
    _, model = tiny_reference(tmp_path, tie, layers=LAYERS)
    ids = [5, 9, 100, 3, 42, 7]
    expected = generate(model, ids, 12)
    for k in range(1, LAYERS):
        assert generate(InProcessSplit.load(tmp_path, k), ids, 12) == expected, f"k={k}"


@pytest.fixture
def reads(monkeypatch):
    """The names of the tensors split.Stage.load reads from the checkpoint, in order."""
    real, read = split.safe_open, []

    class RecordingOpen:
        def __init__(self, *args, **kwargs):
            self._file = real(*args, **kwargs)

        def __enter__(self):
            self._opened = self._file.__enter__()
            return self

        def __exit__(self, *exc):
            return self._file.__exit__(*exc)

        def keys(self):
            return self._opened.keys()

        def get_tensor(self, name):
            read.append(name)
            return self._opened.get_tensor(name)

    monkeypatch.setattr(split, "safe_open", RecordingOpen)
    return read


def layer_of(name):
    match = re.match(r"model\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


@pytest.mark.parametrize("tie", [True, False])
def test_each_stage_reads_and_holds_only_its_parts(tmp_path, reads, tie):
    tiny_reference(tmp_path, tie, layers=LAYERS)
    with safe_open(tmp_path / "model.safetensors", framework="pt") as f:
        on_disk = set(f.keys())
    assert (LM_HEAD in on_disk) == (not tie)
    head = EMBED if tie else LM_HEAD

    for k in range(1, LAYERS):
        reads.clear()
        stage0 = Stage.load(tmp_path, 0, k)
        assert sorted(reads) == sorted(
            {n for n in on_disk if layer_of(n) is not None and layer_of(n) < k} | {EMBED}
        )
        assert stage0.embed is not None
        assert stage0.norm is None and stage0.lm_head is None
        assert len(stage0.layers) == len(stage0.new_cache().keys) == k

        reads.clear()
        stage1 = Stage.load(tmp_path, k, LAYERS)
        assert sorted(reads) == sorted(
            {n for n in on_disk if layer_of(n) is not None and layer_of(n) >= k} | {NORM, head}
        )
        assert stage1.embed is None
        assert stage1.norm is not None and stage1.lm_head is not None
        assert len(stage1.layers) == len(stage1.new_cache().keys) == LAYERS - k


def test_split_points_outside_the_model_are_rejected(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    for k in (0, LAYERS, -1, LAYERS + 3):
        with pytest.raises(ValueError, match="split point"):
            InProcessSplit.load(tmp_path, k)
    for start, end in ((0, 0), (3, 3), (4, 2), (0, LAYERS + 1), (-1, 2)):
        with pytest.raises(ValueError, match="outside"):
            Stage.load(tmp_path, start, end)


def test_stages_that_do_not_meet_are_rejected(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    with pytest.raises(ValueError, match="covering the whole model"):
        InProcessSplit(Stage.load(tmp_path, 0, 2), Stage.load(tmp_path, 3, LAYERS))
    with pytest.raises(ValueError, match="covering the whole model"):
        InProcessSplit(Stage.load(tmp_path, 1, 3), Stage.load(tmp_path, 3, LAYERS))


def test_the_second_stage_rejects_what_is_not_hidden_states(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    stage1 = Stage.load(tmp_path, 2, LAYERS)
    cache = stage1.new_cache()
    with pytest.raises(ValueError, match="hidden states"):
        stage1.forward(torch.tensor([1, 2]), 0, cache)  # token ids
    with pytest.raises(ValueError, match="hidden states"):
        stage1.forward(torch.zeros(2, 63), 0, cache)  # wrong width
    with pytest.raises(ValueError, match="hidden states"):
        stage1.forward(torch.zeros(2, 64, dtype=torch.float64), 0, cache)


def test_the_first_stage_rejects_what_is_not_token_ids(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    stage0 = Stage.load(tmp_path, 0, 2)
    cache = stage0.new_cache()
    with pytest.raises(ValueError, match="token ids"):
        stage0.forward(torch.zeros(2, 64), 0, cache)  # hidden states
    with pytest.raises(ValueError, match="token ids"):
        stage0.forward(torch.tensor([[1, 2]]), 0, cache)  # wrong rank


def test_a_stage_keeps_the_start_pos_rule(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    stage = Stage.load(tmp_path, 0, 2)
    with pytest.raises(ValueError, match="start_pos"):
        stage.forward(torch.tensor([1]), 1, stage.new_cache())


@pytest.mark.skipif(
    not (SMOLLM2_DIR / "model.safetensors").exists(), reason="model weights not downloaded"
)
def test_smollm2_splits_equal_the_single_process_model():
    for filename, expected in PINNED["smollm2-135m"].files.items():
        verify(SMOLLM2_DIR / filename, expected)
    model = Model.load(SMOLLM2_DIR)
    n = model.config.num_layers
    tokens = torch.randint(
        0, model.config.vocab_size, (20,), generator=torch.Generator().manual_seed(2)
    )
    expected = drive(model, tokens)
    del model
    for k in (1, n // 4, n // 2, 3 * n // 4, n - 1):
        assert same(drive(InProcessSplit.load(SMOLLM2_DIR, k), tokens), expected), f"k={k}"
