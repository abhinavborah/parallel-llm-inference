"""Our forward pass against the reference library on the real SmolLM2-135M weights.

Needs the downloaded model (uv run python -m pllm.models smollm2-135m) and about
1.5 GB of free memory, so it is skipped when the weights are absent, as in CI.
"""

import pytest
import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

from pllm.model import Model
from pllm.models import MODELS_DIR

MODEL_DIR = MODELS_DIR / "smollm2-135m"
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

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(), reason="model weights not downloaded"
)


@pytest.fixture(scope="module")
def models():
    reference = LlamaForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return reference, Model.load(MODEL_DIR)


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


def test_twenty_prompts(models, tokenizer):
    reference, model = models
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

    print(f"max abs logit difference over {len(PROMPTS)} prompts: {worst:.2e}")
    assert worst < 1e-4
