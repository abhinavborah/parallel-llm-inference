"""Our forward pass against the reference library on the real model weights.

Each model needs its download (uv run python -m pllm.models smollm2-135m qwen3-0.6b)
and is skipped when its weights are absent, as in CI. Free memory needed: about
1.5 GB for SmolLM2-135M and about 5 GB for Qwen3-0.6B (ours and the reference in fp32).
"""

import pytest
import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM, Qwen3ForCausalLM

from pllm.model import Model
from pllm.models import MODELS_DIR, PINNED, verify

REFERENCE_CLASSES = {"smollm2-135m": LlamaForCausalLM, "qwen3-0.6b": Qwen3ForCausalLM}
NEW_TOKENS = 16

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The three primary colours are",
    "In distributed systems, a consensus algorithm",
    "My favourite recipe for pancakes starts with",
    "The quick brown fox",
    "import numpy as np\nx = np.",
    "Photosynthesis is the process by which",
    "Question: What is 12 times 12?\nAnswer:",
    "The largest planet in the solar system is",
    "A haiku about autumn:",
    "SELECT name FROM users WHERE",
    "The theory of relativity states that",
    "To install the package, run",
    "Dear team, I am writing to",
    "The mitochondria is",
    "Translate to French: good morning ->",
    "1, 1, 2, 3, 5, 8,",
]


@pytest.fixture(scope="module", params=sorted(REFERENCE_CLASSES))
def model_dir(request):
    """The verified model folder; a stale or hand-copied file must not produce a result."""
    model_dir = MODELS_DIR / request.param
    if not (model_dir / "model.safetensors").exists():
        pytest.skip(f"{request.param} weights not downloaded")
    for filename, expected in PINNED[request.param].files.items():
        verify(model_dir / filename, expected)
    return model_dir


def test_twenty_prompts(model_dir):
    reference = (
        REFERENCE_CLASSES[model_dir.name]
        .from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
        .eval()
    )
    model = Model.load(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    assert len(PROMPTS) == 20
    worst = 0.0
    for prompt in PROMPTS:
        tokens = torch.tensor(tokenizer.encode(prompt).ids)

        # Logits for every prompt position.
        with torch.inference_mode():
            expected = reference(tokens[None]).logits[0]
        cache = model.new_cache()
        ours = model.forward(tokens, 0, cache)
        worst = max(worst, (ours - expected).abs().max().item())

        # Greedy decoding: ours through the KV cache, the reference by a full pass per step.
        # A plain argmax loop, not generate, which would apply the model's generation
        # settings (such as stopping at the end-of-sequence token).
        generated = []
        next_token = ours[-1].argmax()
        for pos in range(len(tokens), len(tokens) + NEW_TOKENS):
            generated.append(next_token.item())
            next_token = model.forward(next_token[None], pos, cache)[-1].argmax()
        expected_tokens = tokens
        with torch.inference_mode():
            for _ in range(NEW_TOKENS):
                step = reference(expected_tokens[None]).logits[0, -1].argmax()
                expected_tokens = torch.cat((expected_tokens, step[None]))
        assert generated == expected_tokens[len(tokens) :].tolist(), prompt

    print(f"{model_dir.name}: max abs logit difference over {len(PROMPTS)} prompts: {worst:.2e}")
    assert worst < 1e-4
