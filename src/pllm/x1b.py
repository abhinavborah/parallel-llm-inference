"""X1b: agreement of the two-laptop pipeline with each host's single-process reference (R7.8).

The references are X1a run records, one per host (R2.12); Host B's is copied into runs/ on
Host A. Each record is checked before any request is sent: script x1a with status ok, its
reference checksum recomputed over the model, revision, length, token count, thread count,
prompt set and entries, the same settings in every record, the gateway's model and revision,
and each stored prompt hash equal to the prompt set's ids. The records must come from two
different hosts; --trial allows one host and the run record says trial. Length, token count
and prompts come from the records.

Each prompt goes through a running gateway once, one request at a time, for the records'
token count. The first is sent once both stages beat in the gateway's metrics file. Per
prompt and reference:

- first_divergence: the first step whose token differs, null if none;
- agreed_tokens: the steps before it (all of them if none), out of compared_tokens, the
  tokens streamed;
- max_logit_diff: the largest difference of the chosen token's logit over the steps up to
  and including the first divergence. The wire and the X1a record carry only that logit, so
  this is not the full-vocabulary difference X1a reports.

The gateway stops at end of sequence and the reference never does: a stop on the token the
reference also chose agrees, a stop on another token diverges there. A stream that ends
early on a token that is not end of sequence is a failed request.

Per reference the summary gives sequence agreement (prompts whose tokens all agree) and
token agreement (agreed over compared tokens). Across hosts the result is agreement, never
exact equality, so nothing is asserted about it: the exit code is 0 when every request
completed. One JSON line per prompt and reference, then one summary per reference, go to a
.jsonl file next to the run record in runs/ and to stdout. A non-finite difference is null
with non_finite true.

Usage:
uv run python -m pllm.x1b smollm2-135m --metrics runs/<gateway>.metrics.jsonl \
    --reference runs/<host A>-x1a.json runs/<host B>-x1a.json
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import grpc

from pllm import client, config, prompts, report, runrecord, x1a
from pllm.gateway import eos_ids
from pllm.models import MODELS_DIR, PINNED

SHARED = ("model", "revision", "length", "new_tokens")  # the same in every reference
WAIT_S = 120.0
POLL_S = 0.5
GRACE_S = 1.0  # after both stages beat, for the route updates that follow registration


class BadReference(ValueError):
    """An X1a record that cannot serve as a reference."""


def wait_until_ready(path: Path, wait: float, poll: float = POLL_S, grace: float = GRACE_S):
    """Whether both stages beat in the gateway's metrics file within wait seconds.

    The gateway logs beats of live epochs only, so a beat from each stage means both are
    registered. A request sent before that is refused and leaves a request line with no rid.
    """
    until = time.monotonic() + wait
    while True:
        if {0, 1} <= {b["stage"] for b in report.load(path).beats}:
            time.sleep(grace)
            return True
        if time.monotonic() >= until:
            return False
        time.sleep(poll)


def prompt_sha256(ids: list[int]) -> str:
    """As x1a stores it per prompt."""
    return hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()


def load_reference(path: Path) -> dict:
    """An X1a run record whose reference checksum still matches its contents."""
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise BadReference(f"{path}: not a readable record ({err})") from None
    if not isinstance(rec, dict) or rec.get("script") != "x1a" or rec.get("status") != "ok":
        raise BadReference(f"{path}: not an x1a record with status ok")
    try:
        identity = {
            "model": rec["model"],
            "revision": rec["revision"],
            "length": rec["length"],
            "new_tokens": rec["new_tokens"],
            "threads": rec["threads"],
            "prompt_set_sha256": prompts.PROMPTS_SHA256,
        }
        stored = x1a.checksum({"identity": identity, "entries": rec["reference"]})
    except (KeyError, TypeError, ValueError) as err:
        raise BadReference(f"{path}: the record lacks its reference ({err})") from None
    if stored != rec.get("reference_sha256"):
        raise BadReference(f"{path}: the reference checksum does not match its contents")
    return rec


def check_references(records: list[dict], prompt_ids: list[list[int]]) -> None:
    """Every record has the same settings and prompts, and each prompt is the prompt set's."""
    first = records[0]
    indices = [entry["index"] for entry in first["reference"]]
    for rec in records[1:]:
        for key in SHARED:
            if rec[key] != first[key]:
                raise BadReference(f"the references differ in {key}: {first[key]} and {rec[key]}")
        if [entry["index"] for entry in rec["reference"]] != indices:
            raise BadReference("the references cover different prompts")
    for rec in records:
        for entry in rec["reference"]:
            index = entry["index"]
            if not 0 <= index < len(prompt_ids):
                raise BadReference(f"prompt index {index} is not in the prompt set")
            if entry["prompt_sha256"] != prompt_sha256(prompt_ids[index]):
                raise BadReference(f"prompt {index} is not the prompt set's")


def compare(ref_tokens, ref_logits, tokens, logits) -> dict:
    """Agreement of one streamed result with one reference."""
    n = len(tokens)
    first = next((i for i in range(n) if tokens[i] != ref_tokens[i]), None)
    diff = 0.0
    for i in range(n if first is None else first + 1):
        step = abs(logits[i] - ref_logits[i])
        diff = math.inf if math.isnan(step) else max(diff, step)
    return {
        "compared_tokens": n,
        "agreed_tokens": n if first is None else first,
        "first_divergence": first,
        **x1a.diff_fields(diff),
    }


def serve(call: Callable[[list[int]], Iterator], ids: list[int], new_tokens: int, eos) -> dict:
    """One request through the gateway: its rid, tokens and top logits, or why it failed."""
    got = {"rid": None, "tokens": [], "top_logits": [], "error": None}
    try:
        for token in call(ids):
            got["rid"] = token.rid
            got["tokens"].append(token.token)
            got["top_logits"].append(token.top_logit)
    except grpc.RpcError as err:
        got["error"] = f"{err.code().name}: {err.details()}"
        return got
    n = len(got["tokens"])
    if n > new_tokens:
        got["error"] = f"{n} tokens streamed, more than the {new_tokens} asked for"
    elif n < new_tokens and (not n or got["tokens"][-1] not in eos):
        got["error"] = f"the stream ended after {n} tokens without an end of sequence token"
    return got


def summary_line(name: str, rec: dict, results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    compared = sum(r["compared_tokens"] for r in ok)
    agreed = sum(r["agreed_tokens"] for r in ok)
    diverged = [r["first_divergence"] for r in ok if r["first_divergence"] is not None]
    worst = max((math.inf if r["non_finite"] else r["max_logit_diff"] for r in ok), default=0.0)
    return {
        "type": "summary",
        "reference": name,
        "reference_host": rec["host"],
        "prompts": len(results),
        "failed_requests": len(results) - len(ok),
        "prompts_agreeing": len(ok) - len(diverged),
        "sequence_agreement_percent": 100.0 * (len(ok) - len(diverged)) / len(ok) if ok else 0.0,
        "agreed_tokens": agreed,
        "compared_tokens": compared,
        "token_agreement_percent": 100.0 * agreed / compared if compared else 0.0,
        "first_divergence_min": min(diverged, default=None),
        "first_divergence_median": statistics.median(diverged) if diverged else None,
        **x1a.diff_fields(worst),
    }


def run(call, references: dict[str, dict], prompt_ids, new_tokens: int, eos) -> Iterator[dict]:
    """Serve every prompt once, then yield one line per prompt and reference, then one
    summary per reference. references maps a name to its X1a record. The served results
    come last, as a line of type served."""
    entries = next(iter(references.values()))["reference"]
    served = [
        {"index": e["index"], **serve(call, prompt_ids[e["index"]], new_tokens, frozenset(eos))}
        for e in entries
    ]
    for name, rec in references.items():
        results = []
        for want, got in zip(rec["reference"], served, strict=True):
            line = {"type": "result", "reference": name, "index": got["index"], "rid": got["rid"]}
            if got["error"] is not None:
                line["error"] = got["error"]
            else:
                line.update(
                    compare(want["tokens"], want["top_logits"], got["tokens"], got["top_logits"])
                )
            results.append(line)
            yield line
        yield summary_line(name, rec, results)
    yield {"type": "served", "results": served}


def x1a_passed(rec: dict) -> bool | None:
    s = rec.get("summary") or {}
    if "matched" not in s:
        return None
    return s["matched"] == s["comparisons"] and not s["non_finite"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X1b: pipeline agreement with each reference.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--metrics", type=Path, required=True, help="the gateway's metrics file")
    parser.add_argument(
        "--reference", type=Path, nargs="+", required=True, help="X1a run records, one per host"
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per request")
    parser.add_argument("--wait", type=float, default=WAIT_S, help="seconds for both stages")
    parser.add_argument("--rtt-ms", type=float, help="round-trip time from pllm.netcheck")
    parser.add_argument(
        "--trial", action="store_true", help="allow references from one host; the record says so"
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.wait <= 0:
        parser.error("--timeout and --wait must be positive")
    if not args.metrics.is_file():
        parser.error("--metrics must name the running gateway's metrics file")
    info = report.load(args.metrics).info
    if info.get("model") != args.model:
        parser.error(f"the gateway serves {info.get('model')}, not {args.model}")
    try:
        records = [load_reference(path) for path in args.reference]
        if records[0]["model"] != args.model:
            raise BadReference(f"the references are for {records[0]['model']}, not {args.model}")
        if records[0]["revision"] != info.get("revision"):
            raise BadReference(
                f"the references are for revision {records[0]['revision']}, "
                f"the gateway serves {info.get('revision')}"
            )
        prompt_ids = prompts.prompts(args.model, records[0]["length"])
        check_references(records, prompt_ids)
    except (BadReference, ValueError) as err:
        parser.error(str(err))
    names = [path.name for path in args.reference]
    if len(set(names)) != len(names):
        parser.error("two references have the same file name")
    if len({r["host"] for r in records}) < 2 and not args.trial:
        parser.error("the references must come from two hosts; --trial allows one")
    first = records[0]
    tokenizer = prompts.load_tokenizer(args.model)
    for entry in first["reference"]:
        if not prompts.round_trips(tokenizer, prompt_ids[entry["index"]]):
            parser.error(f"prompt {entry['index']} does not survive the gateway's tokenisation")

    address = config.load().gateway.address

    def call(ids):
        return client.generate(address, tokenizer.decode(ids), first["new_tokens"], args.timeout)

    if not wait_until_ready(args.metrics, args.wait):
        parser.error(f"no heartbeat from both stages within {args.wait} s: is the system up?")
    spec = PINNED[args.model]
    with runrecord.record(
        "x1b", model=args.model, revision=spec.revision, split=info.get("split"), rtt_ms=args.rtt_ms
    ) as rec:
        rec["argv"] = sys.argv
        rec.update(length=first["length"], new_tokens=first["new_tokens"])
        rec["metrics_file"] = args.metrics.name
        rec["trial"] = args.trial
        rec["references"] = [
            {
                "file": name,
                "host": r["host"],
                "os": r.get("os"),
                "threads": r["threads"],
                "reference_sha256": r["reference_sha256"],
                "x1a_passed": x1a_passed(r),
            }
            for name, r in zip(names, records, strict=True)
        ]
        lines_path = runrecord.output_path(rec, ".jsonl")
        rec["results_file"] = lines_path.name
        summaries = []
        with lines_path.open("w", encoding="utf-8") as out:
            references = dict(zip(names, records, strict=True))
            eos = eos_ids(MODELS_DIR / args.model)
            for line in run(call, references, prompt_ids, first["new_tokens"], eos):
                if line["type"] == "served":
                    rec["served"] = line["results"]
                    continue
                if line["type"] == "summary":
                    summaries.append(line)
                text = json.dumps(line, allow_nan=False)
                out.write(text + "\n")
                print(text, flush=True)
        rec["summary"] = summaries
    return 0 if all(s["failed_requests"] == 0 for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
