"""X1a: exact equality of the two-stage split with the single-process model, on one host.

For every prompt of one length and every split point, the split decodes greedily and is
compared with the single-process reference: the tokens must be identical and the logits
are compared over the full vocabulary at every step. One JSON line is printed per prompt
and split, then a summary line. The run passes at 100 percent exact token match, and the
exit code says so.

Equality needs one thread count for the reference and the split (R8.9), so --threads is
required and checked before every decode. The reference runs first for all prompts and its
model is freed before the splits load, which keeps the peak memory near one model plus
the logits of the reference run.

The run record keeps each prompt's reference tokens and top logits with a sha256, so
the two hosts' references can be compared later.

Usage:
uv run python -m pllm.x1a smollm2-135m --threads 6 --length 128 > runs/x1a.jsonl
"""

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Iterator

import torch

from pllm import prompts, runrecord
from pllm.model import Model, ModelConfig
from pllm.models import MODELS_DIR, PINNED, verify
from pllm.reference import decode
from pllm.split import InProcessSplit


def split_points(num_layers: int) -> list[int]:
    """The tested split points 1, n/4, n/2, 3n/4 and n-1, without repeats."""
    n = num_layers
    return sorted({k for k in (1, n // 4, n // 2, 3 * n // 4, n - 1) if 1 <= k <= n - 1})


def compare(
    expected: tuple[list[int], list[torch.Tensor]], got: tuple[list[int], list[torch.Tensor]]
) -> tuple[bool, float]:
    """Whether the tokens are identical, and the largest logit difference over the steps.

    Steps after the first differing token follow different prefixes and are not compared.
    """
    max_diff = 0.0
    for step, (want, have) in enumerate(zip(expected[1], got[1], strict=True)):
        diff = (want - have).abs().max().item()
        max_diff = math.inf if math.isnan(diff) else max(max_diff, diff)
        if expected[0][step] != got[0][step]:
            break
    return expected[0] == got[0], max_diff


def reference_entries(
    traces: list[tuple[list[int], list[torch.Tensor]]],
    ids_list: list[list[int]],
    index_of: list[int],
) -> list[dict]:
    """Tokens and top logit per step for each prompt, the summary stored for comparison."""
    return [
        {
            "index": index,
            "prompt_tokens": len(ids),
            "prompt_sha256": hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest(),
            "tokens": tokens,
            "top_logits": [row.max().item() for row in rows],
        }
        for index, ids, (tokens, rows) in zip(index_of, ids_list, traces, strict=True)
    ]


def checksum(content: object) -> str:
    """sha256 of JSON-able content in a canonical form."""
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_sweep(
    load_reference: Callable[[], Model],
    load_split: Callable[[int], InProcessSplit],
    ids_list: list[list[int]],
    points: list[int],
    new_tokens: int,
    threads: int,
    index_of: list[int] | None = None,
    identity: dict | None = None,
) -> Iterator[dict]:
    """Yield one result per prompt and split, then a summary with the reference entries.

    identity (model, prompt set, length and so on) is covered by the reference checksum,
    so two checksums are equal only for the same inputs and the same outputs.
    """
    index_of = index_of if index_of is not None else list(range(len(ids_list)))

    def traced(model, ids):
        if torch.get_num_threads() != threads:
            raise RuntimeError(f"thread count changed to {torch.get_num_threads()}, not {threads}")
        return decode(model, ids, new_tokens)

    model = load_reference()
    expected = [traced(model, ids) for ids in ids_list]
    del model
    entries = reference_entries(expected, ids_list, index_of)

    total = matched = 0
    worst = 0.0
    for k in points:
        split = load_split(k)
        for index, ids, want in zip(index_of, ids_list, expected, strict=True):
            same, diff = compare(want, traced(split, ids))
            total += 1
            matched += same
            worst = max(worst, diff)
            yield {
                "type": "result",
                "index": index,
                "split": k,
                "tokens_match": same,
                "max_logit_diff": diff,
            }
        del split
    yield {
        "type": "summary",
        "comparisons": total,
        "matched": matched,
        "match_percent": 100.0 * matched / total if total else 0.0,
        "max_logit_diff": worst,
        "split_points": points,
        "threads": threads,
        "new_tokens": new_tokens,
        "interop_threads": torch.get_num_interop_threads(),
        "reference_sha256": checksum({"identity": identity or {}, "entries": entries}),
        "reference": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X1a: split against single process.")
    parser.add_argument("model", choices=sorted(PINNED))
    # Required, since equality needs the same count for the reference and the split.
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--length", type=int, choices=prompts.LENGTHS, default=128)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--count", type=int, default=prompts.PASSAGES, help="first N prompts")
    args = parser.parse_args(argv)
    if not 1 <= args.count <= prompts.PASSAGES:
        parser.error(f"--count must be 1 to {prompts.PASSAGES}")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    spec = PINNED[args.model]
    model_dir = MODELS_DIR / args.model
    with runrecord.record(
        "x1a", model=args.model, revision=spec.revision, threads=args.threads
    ) as rec:
        for filename, expected in spec.files.items():
            verify(model_dir / filename, expected)
        ids_list = prompts.prompts(args.model, args.length)[: args.count]
        points = split_points(ModelConfig.from_file(model_dir / "config.json").num_layers)
        rec.update(length=args.length, new_tokens=args.new_tokens, split_points=points)
        summary = {}
        for line in run_sweep(
            lambda: Model.load(model_dir),
            lambda k: InProcessSplit.load(model_dir, k),
            ids_list,
            points,
            args.new_tokens,
            args.threads,
            identity={
                "model": args.model,
                "revision": spec.revision,
                "length": args.length,
                "new_tokens": args.new_tokens,
                "threads": args.threads,
                "prompt_set_sha256": prompts.PROMPTS_SHA256,
            },
        ):
            if line["type"] == "summary":
                summary = line
                line = {key: value for key, value in line.items() if key != "reference"}
            print(json.dumps(line), flush=True)
        rec["reference"] = summary["reference"]
        rec["reference_sha256"] = summary["reference_sha256"]
        rec["summary"] = {k: v for k, v in summary.items() if k != "reference"}
    return 0 if summary["matched"] == summary["comparisons"] else 1


if __name__ == "__main__":
    sys.exit(main())
