import json

import pytest
import torch

from pllm import reference, runrecord
from pllm.model import Model, ModelConfig
from pllm.models import MODELS_DIR


def tiny_model():
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    weights = {
        "model.embed_tokens.weight": torch.randn(64, 32),
        "model.norm.weight": torch.ones(32),
    }
    shapes = {
        "input_layernorm": (32,),
        "post_attention_layernorm": (32,),
        "self_attn.q_proj": (32, 32),
        "self_attn.k_proj": (16, 32),
        "self_attn.v_proj": (16, 32),
        "self_attn.o_proj": (32, 32),
        "mlp.gate_proj": (64, 32),
        "mlp.up_proj": (64, 32),
        "mlp.down_proj": (32, 64),
    }
    for i in range(config.num_layers):
        for name, shape in shapes.items():
            weights[f"model.layers.{i}.{name}.weight"] = torch.randn(shape)
    return Model(config, weights)


def test_generate_matches_argmax_of_a_full_pass():
    model = tiny_model()
    prompt = [3, 14, 15, 9, 26]
    tokens, top = reference.generate(model, prompt, 6)
    assert len(tokens) == len(top) == 6

    # Oracle: no cache reuse, a full pass over the whole sequence for every step.
    sequence = list(prompt)
    for token, logit in zip(tokens, top, strict=True):
        last = model.forward(torch.tensor(sequence), 0, model.new_cache())[-1]
        assert token == last.argmax().item()
        assert logit == pytest.approx(last.max().item(), abs=1e-5)
        sequence.append(token)


def test_generate_rejects_empty_input():
    with pytest.raises(ValueError):
        reference.generate(tiny_model(), [], 4)
    with pytest.raises(ValueError):
        reference.generate(tiny_model(), [1], 0)


def test_only_cpu_is_accepted():
    with pytest.raises(SystemExit):
        reference.main(["smollm2-135m", "--text", "hi", "--device", "cuda"])


@pytest.mark.skipif(
    not (MODELS_DIR / "smollm2-135m" / "model.safetensors").exists(),
    reason="model weights not downloaded",
)
def test_smoke_on_smollm2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path)
    threads = torch.get_num_threads()
    try:
        reference.main(["smollm2-135m", "--length", "16", "--new-tokens", "4", "--threads", "1"])
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(threads)

    out = json.loads(capsys.readouterr().out)
    assert len(out["tokens"]) == len(out["top_logits"]) == 4
    (record_file,) = tmp_path.glob("*-reference.json")
    rec = json.loads(record_file.read_text(encoding="utf-8"))
    assert rec["status"] == "ok"
    assert rec["threads"] == 1
    assert rec["device"] == "cpu"
    assert len(rec["prompt"]) == 16
    assert rec["tokens"] == out["tokens"]
