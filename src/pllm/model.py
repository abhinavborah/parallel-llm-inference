"""Llama-family forward pass on CPU in fp32, with an explicit KV cache.

Weights are read from a Hugging Face safetensors file and upcast to fp32. The
cache holds keys and values for every layer indexed by position: a call with
start_pos first truncates the cache to start_pos and then appends the new
positions, which is the rule the stages use when a prompt is replayed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        # Configs saved by transformers 4.x keep rope_theta at the top level; 5.x moves
        # it, with the rope type, into rope_parameters.
        rope = cfg.get("rope_parameters") or {}
        rope_theta = rope.get("rope_theta") or cfg.get("rope_theta") or 10000.0
        # Fail loudly on variants this forward pass does not implement.
        unsupported = {
            "model_type": cfg.get("model_type", "llama") != "llama",
            "rope_scaling": cfg.get("rope_scaling") is not None,
            "rope_type": rope.get("rope_type", "default") != "default",
            "rope_interleaved": cfg.get("rope_interleaved", False),
            "attention_bias": cfg.get("attention_bias", False),
            "mlp_bias": cfg.get("mlp_bias", False),
            "hidden_act": cfg.get("hidden_act", "silu") != "silu",
        }
        bad = [key for key, is_bad in unsupported.items() if is_bad]
        if bad:
            raise ValueError(f"unsupported model configuration: {', '.join(bad)}")
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=rope_theta,
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Layer:
    attn_norm: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    mlp_norm: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


class KVCache:
    """Keys and values per layer, each shaped (kv_heads, positions, head_dim)."""

    def __init__(self, config: ModelConfig):
        empty = torch.empty(config.num_kv_heads, 0, config.head_dim)
        self.keys = [empty] * config.num_layers
        self.values = [empty] * config.num_layers

    def __len__(self) -> int:
        return self.keys[0].shape[1]


class Model:
    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor]):
        self.config = config
        self.embed = weights["model.embed_tokens.weight"]
        self.layers = [_layer(weights, i) for i in range(config.num_layers)]
        self.norm = weights["model.norm.weight"]
        self.lm_head = self.embed if config.tie_word_embeddings else weights["lm_head.weight"]
        half = config.head_dim // 2
        exponents = torch.arange(0, half, dtype=torch.float32) * 2 / config.head_dim
        self.inv_freq = 1.0 / (config.rope_theta**exponents)

    @classmethod
    def load(cls, model_dir: str | Path) -> "Model":
        """Load config.json and model.safetensors from a folder, upcasting to fp32."""
        model_dir = Path(model_dir)
        config = ModelConfig.from_file(model_dir / "config.json")
        weights = {}
        with safe_open(model_dir / "model.safetensors", framework="pt") as f:
            for name in f.keys():
                weights[name] = f.get_tensor(name).to(torch.float32)
        return cls(config, weights)

    def new_cache(self) -> KVCache:
        return KVCache(self.config)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, cache: KVCache) -> torch.Tensor:
        """Run tokens at positions start_pos onward; return logits shaped (tokens, vocab).

        The cache is truncated to start_pos before the new keys and values are added,
        so start_pos may not be past the end of the cache.
        """
        if not 0 <= start_pos <= len(cache):
            raise ValueError(f"start_pos {start_pos} outside cache of length {len(cache)}")
        positions = torch.arange(start_pos, start_pos + len(tokens))
        cos, sin = self._rope(positions)

        x = self.embed[tokens]
        for i, layer in enumerate(self.layers):
            x = x + self._attention(
                i, layer, _rms_norm(x, layer.attn_norm, self.config), cos, sin, start_pos, cache
            )
            h = _rms_norm(x, layer.mlp_norm, self.config)
            x = x + F.linear(
                F.silu(F.linear(h, layer.w_gate)) * F.linear(h, layer.w_up), layer.w_down
            )
        return F.linear(_rms_norm(x, self.norm, self.config), self.lm_head)

    def _rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions[:, None].to(torch.float32) * self.inv_freq
        angles = torch.cat((freqs, freqs), dim=-1)
        return angles.cos(), angles.sin()

    def _attention(self, i, layer, x, cos, sin, start_pos, cache) -> torch.Tensor:
        cfg = self.config
        n = x.shape[0]
        q = F.linear(x, layer.wq).view(n, cfg.num_heads, cfg.head_dim).transpose(0, 1)
        k = F.linear(x, layer.wk).view(n, cfg.num_kv_heads, cfg.head_dim).transpose(0, 1)
        v = F.linear(x, layer.wv).view(n, cfg.num_kv_heads, cfg.head_dim).transpose(0, 1)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        keys = torch.cat((cache.keys[i][:, :start_pos], k), dim=1)
        values = torch.cat((cache.values[i][:, :start_pos], v), dim=1)
        cache.keys[i], cache.values[i] = keys, values

        group = cfg.num_heads // cfg.num_kv_heads
        keys = keys.repeat_interleave(group, dim=0)
        values = values.repeat_interleave(group, dim=0)
        scores = torch.matmul(q, keys.transpose(1, 2)) * cfg.head_dim**-0.5
        query_pos = torch.arange(start_pos, start_pos + n)[:, None]
        key_pos = torch.arange(keys.shape[1])[None, :]
        scores = scores.masked_fill(key_pos > query_pos, float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), values)
        return F.linear(out.transpose(0, 1).reshape(n, cfg.num_heads * cfg.head_dim), layer.wo)


def _layer(weights: dict[str, torch.Tensor], i: int) -> Layer:
    p = f"model.layers.{i}."
    return Layer(
        attn_norm=weights[p + "input_layernorm.weight"],
        wq=weights[p + "self_attn.q_proj.weight"],
        wk=weights[p + "self_attn.k_proj.weight"],
        wv=weights[p + "self_attn.v_proj.weight"],
        wo=weights[p + "self_attn.o_proj.weight"],
        mlp_norm=weights[p + "post_attention_layernorm.weight"],
        w_gate=weights[p + "mlp.gate_proj.weight"],
        w_up=weights[p + "mlp.up_proj.weight"],
        w_down=weights[p + "mlp.down_proj.weight"],
    )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + config.rms_norm_eps))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
