import ast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from pllm.config import REPO_ROOT
from pllm.model import Model, ModelConfig


def tiny_reference(tmp_path, tie=True, qwen3=False, layers=2):
    """A small random Llama or Qwen3 saved the way Hugging Face saves it, plus its reference."""
    torch.manual_seed(0)
    # head_dim 32, not hidden_size // heads = 16, so a head_dim read wrongly shows up.
    config_class, model_class, extra = (
        (Qwen3Config, Qwen3ForCausalLM, {"head_dim": 32})
        if qwen3
        else (LlamaConfig, LlamaForCausalLM, {})
    )
    config = config_class(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        # Not the 10000 default, so a theta that is read wrongly shows up as a mismatch.
        rope_theta=500000.0,
        tie_word_embeddings=tie,
        attn_implementation="eager",
        **extra,
    )
    reference = model_class(config).to(torch.float32).eval()
    reference.save_pretrained(tmp_path)
    return reference, Model.load(tmp_path)


def reference_logits(reference, tokens):
    with torch.inference_mode():
        return reference(tokens[None]).logits[0]


@pytest.mark.parametrize("qwen3", [False, True])
@pytest.mark.parametrize("tie", [True, False])
def test_prefill_matches_the_reference(tmp_path, tie, qwen3):
    reference, model = tiny_reference(tmp_path, tie, qwen3)
    tokens = torch.randint(0, 128, (12,))
    ours = model.forward(tokens, 0, model.new_cache())
    assert (ours - reference_logits(reference, tokens)).abs().max() < 1e-5


@pytest.mark.parametrize("qwen3", [False, True])
def test_cached_decoding_matches_a_full_pass(tmp_path, qwen3):
    reference, model = tiny_reference(tmp_path, qwen3=qwen3)
    tokens = torch.randint(0, 128, (10,))
    cache = model.new_cache()
    steps = [model.forward(tokens[:6], 0, cache)]
    for pos in range(6, 10):
        steps.append(model.forward(tokens[pos : pos + 1], pos, cache))
    assert len(cache) == 10
    assert (torch.cat(steps) - reference_logits(reference, tokens)).abs().max() < 1e-5


def test_start_pos_truncates_the_cache_before_appending(tmp_path):
    _, model = tiny_reference(tmp_path)
    tokens = torch.randint(0, 128, (8,))
    cache = model.new_cache()
    model.forward(tokens, 0, cache)
    replaced = torch.tensor([5, 6])
    redo = model.forward(replaced, 3, cache)
    assert len(cache) == 5

    fresh = model.forward(torch.cat((tokens[:3], replaced)), 0, model.new_cache())
    assert (redo - fresh[3:]).abs().max() < 1e-5


def test_start_pos_past_the_cache_is_rejected(tmp_path):
    _, model = tiny_reference(tmp_path)
    with pytest.raises(ValueError, match="start_pos"):
        model.forward(torch.tensor([1]), 1, model.new_cache())


def test_unsupported_configuration_is_rejected():
    base = {
        "vocab_size": 8,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
    }
    ModelConfig.from_dict(base)
    with pytest.raises(ValueError, match="rope_scaling"):
        ModelConfig.from_dict(base | {"rope_scaling": {"type": "linear", "factor": 2.0}})
    with pytest.raises(ValueError, match="model_type"):
        ModelConfig.from_dict(base | {"model_type": "mistral"})
    with pytest.raises(ValueError, match="rope_type"):
        ModelConfig.from_dict(
            base | {"rope_parameters": {"rope_type": "llama3", "rope_theta": 1.0}}
        )
    with pytest.raises(ValueError, match="rope_interleaved"):
        ModelConfig.from_dict(base | {"rope_interleaved": True})
    with pytest.raises(ValueError, match="use_sliding_window"):
        ModelConfig.from_dict(base | {"model_type": "qwen3", "use_sliding_window": True})
    with pytest.raises(ValueError, match="layer_types"):
        ModelConfig.from_dict(base | {"layer_types": ["sliding_attention"]})
    assert ModelConfig.from_dict(base | {"model_type": "qwen3"}).qk_norm
    assert not ModelConfig.from_dict(base).qk_norm


def test_rope_theta_is_read_from_either_config_layout():
    base = {
        "vocab_size": 8,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
    }
    # transformers 4.x layout, as in the SmolLM2 config
    assert ModelConfig.from_dict(base | {"rope_theta": 100000}).rope_theta == 100000
    # transformers 5.x layout
    rope = {"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}}
    assert ModelConfig.from_dict(base | rope).rope_theta == 500000.0


def test_production_code_never_imports_the_reference_library():
    for path in (REPO_ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "transformers" for n in names), path
