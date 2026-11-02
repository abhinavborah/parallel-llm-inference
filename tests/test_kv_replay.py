"""KV truncation and chunk shape: the evidence the replay design rests on.

All runs are fp32, CPU, batch 1, on one host. Two findings:

1. Truncating the cache to the end of an earlier message and appending is bitwise
   equal to a fresh run that sent the same messages up to that point. Truncating
   inside a message is only agreement (about 1e-7), so stages truncate only at
   message boundaries.
2. Prefill of P+m tokens is not bitwise equal to prefill of P followed by m
   single-token decodes, because the matrix products reduce over different shapes.
   Replay therefore cannot resend prompt and emitted tokens as one message; it must
   reuse the original message shapes, which is bitwise equal.
"""

import pytest
import torch
from test_model import tiny_reference

from pllm.model import Model
from pllm.models import MODELS_DIR, PINNED, verify

SMOLLM2_DIR = MODELS_DIR / "smollm2-135m"
AGREEMENT = 1e-5

# Measured on torch 2.14.1, macOS arm64, 8 threads, for (P, m) = (16, 8): largest
# key or value difference 2.09e-7, largest logit difference 4.77e-7.
EXPECTED_BITWISE_EQUAL = False


def run(model, tokens, chunks):
    """Send tokens as consecutive messages of the given sizes; return logits and cache."""
    cache = model.new_cache()
    logits, pos = [], 0
    for size in chunks:
        logits.append(model.forward(tokens[pos : pos + size], pos, cache))
        pos += size
    return torch.cat(logits), cache


def pairs(a, b):
    return zip(a.keys + a.values, b.keys + b.values, strict=True)


def kv_max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in pairs(a, b))


def kv_equal(a, b):
    return all(torch.equal(x, y) for x, y in pairs(a, b))


@pytest.fixture
def model(tmp_path):
    return tiny_reference(tmp_path)[1]


@pytest.mark.parametrize("decodes_kept", [0, 2])
def test_truncate_at_a_message_boundary_then_append_equals_a_fresh_run(model, decodes_kept):
    torch.manual_seed(1)
    tokens = torch.randint(0, 128, (24,))
    appended = torch.tensor([3, 4, 5])
    p = 16 + decodes_kept

    _, cache = run(model, tokens, [16] + [1] * 8)
    redo = model.forward(appended, p, cache)

    _, fresh_cache = run(model, tokens, [16] + [1] * decodes_kept)
    fresh = model.forward(appended, p, fresh_cache)

    assert len(cache) == p + len(appended)
    assert torch.equal(redo, fresh)
    assert kv_equal(cache, fresh_cache)


def test_truncate_inside_a_message_only_agrees(model):
    torch.manual_seed(1)
    tokens = torch.randint(0, 128, (24,))
    appended = torch.tensor([3, 4, 5])

    _, cache = run(model, tokens, [24])
    redo = model.forward(appended, 10, cache)
    _, fresh_cache = run(model, tokens, [10])
    fresh = model.forward(appended, 10, fresh_cache)

    assert (redo - fresh).abs().max() < AGREEMENT
    assert kv_max_diff(cache, fresh_cache) < AGREEMENT


@pytest.mark.parametrize(("prefill", "decodes"), [(16, 8), (31, 17)])
def test_prefill_against_prefill_then_single_token_decodes(model, prefill, decodes):
    torch.manual_seed(1)
    tokens = torch.randint(0, 128, (prefill + decodes,))
    whole, whole_cache = run(model, tokens, [prefill + decodes])
    stepped, stepped_cache = run(model, tokens, [prefill] + [1] * decodes)

    assert kv_max_diff(whole_cache, stepped_cache) < AGREEMENT
    assert (whole - stepped).abs().max() < AGREEMENT
    assert kv_equal(whole_cache, stepped_cache) == EXPECTED_BITWISE_EQUAL


def test_replay_with_the_original_message_shapes_is_bitwise_equal(model):
    torch.manual_seed(1)
    tokens = torch.randint(0, 128, (24,))
    chunks = [8, 8] + [1] * 8
    first, first_cache = run(model, tokens, chunks)
    again, again_cache = run(model, tokens, chunks)

    assert torch.equal(first, again)
    assert kv_equal(first_cache, again_cache)
    # A chunked prompt replayed as one message does not reproduce the cache.
    _, merged_cache = run(model, tokens, [16] + [1] * 8)
    assert not kv_equal(first_cache, merged_cache)


@pytest.mark.skipif(
    not (SMOLLM2_DIR / "model.safetensors").exists(), reason="model weights not downloaded"
)
def test_smollm2_prefill_against_decodes_and_boundary_truncation():
    for filename, expected in PINNED["smollm2-135m"].files.items():
        verify(SMOLLM2_DIR / filename, expected)
    model = Model.load(SMOLLM2_DIR)
    torch.manual_seed(1)
    tokens = torch.randint(0, model.config.vocab_size, (24,))

    _, whole_cache = run(model, tokens, [24])
    stepped, stepped_cache = run(model, tokens, [16] + [1] * 8)
    assert kv_max_diff(whole_cache, stepped_cache) < 1e-3
    assert kv_equal(whole_cache, stepped_cache) == EXPECTED_BITWISE_EQUAL

    redo = model.forward(tokens[16:20], 16, stepped_cache)
    _, fresh_cache = run(model, tokens, [16])
    fresh = model.forward(tokens[16:20], 16, fresh_cache)
    assert torch.equal(redo, fresh)
    assert kv_equal(stepped_cache, fresh_cache)
