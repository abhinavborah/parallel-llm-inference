"""Single-process reference runner: the oracle every distributed run is compared against.

It uses the same forward pass as the stages, in fp32 on CPU. Decoding is greedy for a
fixed number of tokens with no stop at the end-of-sequence token, so every run of a
prompt decodes the same count. Step i gives the chosen token and its logit; the first
step comes from the last prefill position.

Usage:
uv run python -m pllm.reference smollm2-135m --length 16 --index 0 --new-tokens 16 --threads 4
uv run python -m pllm.reference smollm2-135m --text "The capital of France is"
"""

import argparse
import json

import torch

from pllm import prompts, runrecord
from pllm.model import Model
from pllm.models import MODELS_DIR, PINNED, verify

# Model has no device placement, and equality runs are CPU only.
DEVICES = ("cpu",)


def generate(model: Model, ids: list[int], new_tokens: int) -> tuple[list[int], list[float]]:
    """Prefill ids, then decode greedily; return the new tokens and the top logit per step."""
    if not ids or new_tokens < 1:
        raise ValueError("need at least one prompt token and one new token")
    cache = model.new_cache()
    logits = model.forward(torch.tensor(ids), 0, cache)[-1]
    tokens, top = [], []
    for pos in range(len(ids), len(ids) + new_tokens):
        value, best = logits.max(dim=-1)
        tokens.append(best.item())
        top.append(value.item())
        if len(tokens) < new_tokens:
            logits = model.forward(best[None], pos, cache)[-1]
    return tokens, top


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the single-process reference.")
    parser.add_argument("model", choices=sorted(PINNED))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="raw prompt text, tokenised without special tokens")
    source.add_argument("--length", type=int, choices=prompts.LENGTHS, help="prompt set length")
    parser.add_argument("--index", type=int, default=0, help="prompt set entry, 0 to 49")
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--device", choices=DEVICES, default="cpu")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    model_dir = MODELS_DIR / args.model
    for filename, expected in PINNED[args.model].files.items():
        verify(model_dir / filename, expected)
    tokenizer = prompts.load_tokenizer(args.model)
    if args.text is not None:
        ids = prompts.encode(tokenizer, args.text)
    else:
        ids = prompts.prompts(args.model, args.length)[args.index]
    model = Model.load(model_dir)

    spec = PINNED[args.model]
    with runrecord.record(
        "reference", model=args.model, revision=spec.revision, threads=args.threads
    ) as rec:
        tokens, top = generate(model, ids, args.new_tokens)
        rec.update(device=args.device, prompt=ids, tokens=tokens, top_logits=top)
    print(json.dumps({"tokens": tokens, "top_logits": top, "text": tokenizer.decode(tokens)}))


if __name__ == "__main__":
    main()
