"""X3: serialized bytes per message on each hop, for one prompt length (R7.10).

A client of a gateway started fresh for this run. Once both stages beat in the gateway's
metrics file (a request sent earlier is refused, and its line would spoil the run), it sends
--count prompts of --length tokens one at a time with --new-tokens (greedy decoding stops at
end of sequence), then asks for the gateway to be stopped while the stages are up, and waits
for the end of its metrics file.

The Stage 0 to Stage 1 hop never passes the gateway: Stage 0 returns its byte counts when its
stream ends, and the gateway writes their totals for the whole run when it stops. So one
gateway run gives one prompt length's numbers, and the file must hold this run's requests
only. A restarted gateway is not rejoined by the stages, so each length needs both laptops
started again.

Per hop and phase (prefill, decode, and replay if any): messages, bytes, payload bytes,
positions, bytes per position, and the overhead per message, bytes minus payload over
messages. The gateway's two hops come from its hop lines, one per message, so they also give
the largest overhead; Stage 0 to Stage 1 has totals only, so only the mean. Bytes are the
protobuf serialized size: gRPC's 5 bytes per message and HTTP/2 framing are not in them.

Only Stage 0 to Stage 1 carries activations (the gateway sends token ids and receives one
token and its logit per step), so the check is on that hop, per phase:

- payload bytes equal positions x hidden size x 4 B exactly;
- the mean overhead per message is at most OVERHEAD_BOUND;
- its positions equal those the gateway sent to Stage 0 (every step forwarded once).

The run fails, with the reasons in the summary, if the file has no run_end in time, Stage 0's
totals are incomplete, a line is unreadable, a request in the file is not this run's, a
request failed, or a request decoded fewer than --new-tokens tokens (it stopped at end of
sequence). One JSON line per hop and phase, then a summary with each request's decoded
token count, go to a .jsonl file next to the run record in runs/ and to stdout. The exit
code is 0 only when the check passes.

Usage:
uv run python -m pllm.x3 smollm2-135m --length 128 --metrics runs/<gateway>.metrics.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pllm import client, config, metrics, prompts, report, runrecord, x1b, x2
from pllm.gateway import eos_ids
from pllm.model import ModelConfig
from pllm.models import MODELS_DIR, PINNED

NEW_TOKENS = 128
COUNT = 5
WAIT_S = 600.0
POLL_S = 0.5
OVERHEAD_BOUND = 128  # bytes per activation message; 45 to 49 measured on loopback
HOPS = (metrics.GATEWAY_STAGE0, metrics.STAGE0_STAGE1, metrics.STAGE1_GATEWAY)
STOP_NOW = (
    "all requests done: stop the gateway now (Ctrl-C in its own terminal) and leave the "
    "stages running until this run ends"
)


def wait_for_end(path: Path, wait: float, poll: float = POLL_S) -> report.Run:
    """The metrics file once it has its run_end line, or as it is after wait seconds."""
    until = time.monotonic() + wait
    while True:
        run = report.load(path)
        if run.end is not None or time.monotonic() >= until:
            return run
        time.sleep(poll)


def row(hop, msg, phase, messages, size, payload, positions, overheads=None) -> dict:
    return {
        "type": "hop",
        "hop": hop,
        "msg": msg,
        "phase": phase,
        "messages": messages,
        "bytes": size,
        "payload_bytes": payload,
        "positions": positions,
        "bytes_per_message": size / messages if messages else None,
        "bytes_per_position": size / positions if positions else None,
        "overhead_mean_bytes": (size - payload) / messages if messages else None,
        "overhead_max_bytes": max(overheads) if overheads else None,
        "source": "stage 0 totals" if overheads is None else "gateway hop lines",
    }


def rows(run: report.Run, hops: list[dict]) -> list[dict]:
    """One row per hop, message type and phase: the gateway's hops from their hop lines,
    Stage 0 to Stage 1 from its totals."""
    groups: dict[tuple, list[dict]] = {}
    for h in hops:
        groups.setdefault((h["hop"], h["msg"], h["phase"]), []).append(h)
    out = [
        row(
            *key,
            len(lines),
            sum(h["bytes"] for h in lines),
            sum(h["payload_bytes"] for h in lines),
            sum(h["positions"] for h in lines),
            [h["bytes"] - h["payload_bytes"] for h in lines],
        )
        for key, lines in groups.items()
    ]
    for b in run.bytes:
        if b["hop"] == metrics.STAGE0_STAGE1:
            out.append(
                row(
                    *(b[k] for k in ("hop", "msg", "phase", "messages", "bytes")),
                    b["payload_bytes"],
                    b["positions"],
                )
            )
    return sorted(out, key=lambda r: (HOPS.index(r["hop"]), metrics.PHASES.index(r["phase"])))


def problems(run: report.Run, rids: set[str], failed: int, table: list[dict], hidden: int, short=0):
    """Why the run does not show bytes equal to hidden size x 4 B x tokens, if it does not.
    short counts the completed requests that decoded fewer tokens than asked for."""
    if run.end is None:
        return ["the metrics file has no run_end: the gateway was not stopped, or not cleanly"]
    found = []
    if not run.end.get("stage0_stage1_complete"):
        found.append("Stage 0 to Stage 1 totals are incomplete: a stream gave no byte report")
    if run.bad_lines:
        found.append(f"{run.bad_lines} unreadable line(s) in the metrics file")
    foreign = {r.get("rid") for r in run.requests} - rids
    if foreign:
        found.append(f"{len(foreign)} request(s) in the file are not this run's")
    if failed:
        found.append(f"{failed} request(s) failed at the client")
    if short:
        found.append(f"{short} request(s) decoded fewer tokens than asked for (end of sequence)")
    done = {r["rid"] for r in run.requests if r.get("outcome") in report.COMPLETED}
    if rids - done:
        found.append(f"{len(rids - done)} request(s) did not complete at the gateway")
    sent = {}
    for r in table:
        if r["hop"] == metrics.GATEWAY_STAGE0:
            sent[r["phase"]] = sent.get(r["phase"], 0) + r["positions"]
    forwarded = [r for r in table if r["hop"] == metrics.STAGE0_STAGE1]
    if {r["phase"] for r in forwarded} != set(sent):
        found.append("Stage 0 to Stage 1 has other phases than the gateway sent")
    for r in forwarded:
        expected = r["positions"] * hidden * 4
        if r["payload_bytes"] != expected:
            found.append(f"{r['phase']}: payload {r['payload_bytes']} B, not {expected} B")
        if r["overhead_mean_bytes"] > OVERHEAD_BOUND:
            found.append(
                f"{r['phase']}: {r['overhead_mean_bytes']:.1f} B overhead per message, "
                f"above {OVERHEAD_BOUND} B"
            )
        if r["positions"] != sent.get(r["phase"]):
            found.append(f"{r['phase']}: {r['positions']} positions forwarded, not as sent")
    return found


def summary_line(table: list[dict], found: list[str], hidden: int) -> dict:
    return {
        "type": "summary",
        "hidden_size": hidden,
        "expected_payload_bytes_per_position": hidden * 4,
        "overhead_bound_bytes": OVERHEAD_BOUND,
        "overhead": [
            {
                "hop": r["hop"],
                "phase": r["phase"],
                "mean_bytes": r["overhead_mean_bytes"],
                "max_bytes": r["overhead_max_bytes"],
            }
            for r in table
        ],
        "passed": not found,
        "problems": found,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X3: bytes per message on each hop.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--length", type=int, choices=prompts.LENGTHS, required=True)
    parser.add_argument("--metrics", type=Path, required=True, help="the fresh gateway's file")
    parser.add_argument("--count", type=int, default=COUNT, help="first N prompts")
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument(
        "--wait", type=float, default=WAIT_S, help="seconds for the stages, then for the stop"
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    parser.add_argument("--rtt-ms", type=float, help="round-trip time from pllm.netcheck")
    args = parser.parse_args(argv)
    if not 1 <= args.count <= prompts.PASSAGES:
        parser.error(f"--count must be 1 to {prompts.PASSAGES}")
    if args.new_tokens < 1 or args.wait <= 0 or args.timeout <= 0:
        parser.error("--new-tokens, --wait and --timeout must be positive")
    if not args.metrics.is_file():
        parser.error("--metrics must name the running gateway's metrics file")
    start = report.load(args.metrics)
    if start.info.get("model") != args.model:
        parser.error(f"the gateway serves {start.info.get('model')}, not {args.model}")
    if start.requests or start.end is not None or start.bad_lines:
        parser.error("the metrics file is not fresh (requests, an end or unreadable lines)")

    spec = PINNED[args.model]
    model_dir = MODELS_DIR / args.model
    hidden = ModelConfig.from_file(model_dir / "config.json").hidden_size
    ids_list = prompts.prompts(args.model, args.length)[: args.count]
    tokenizer = prompts.load_tokenizer(args.model)
    if not all(prompts.round_trips(tokenizer, ids) for ids in ids_list):
        parser.error("a prompt does not survive the gateway's tokenisation")
    address = config.load().gateway.address
    eos = frozenset(eos_ids(model_dir))

    def call(ids):
        return client.generate(address, tokenizer.decode(ids), args.new_tokens, args.timeout)

    if not x1b.wait_until_ready(args.metrics, args.wait):
        parser.error(f"no heartbeat from both stages within {args.wait} s: is the system up?")
    with runrecord.record(
        "x3",
        model=args.model,
        revision=spec.revision,
        split=start.info.get("split"),
        rtt_ms=args.rtt_ms,
    ) as rec:
        rec["argv"] = sys.argv
        rec.update(
            length=args.length,
            count=args.count,
            new_tokens=args.new_tokens,
            hidden_size=hidden,
            metrics_file=args.metrics.name,
        )
        lines_path = runrecord.output_path(rec, ".jsonl")
        rec["results_file"] = lines_path.name
        served = [x1b.serve(call, ids, args.new_tokens, eos) for ids in ids_list]
        rec["requests"] = [
            {"rid": s["rid"], "output_tokens": len(s["tokens"]), "error": s["error"]}
            for s in served
        ]
        print(STOP_NOW, file=sys.stderr, flush=True)
        run = wait_for_end(args.metrics, args.wait)
        rids = {s["rid"] for s in served if s["rid"]}
        failed = sum(s["error"] is not None for s in served)
        decoded = [len(s["tokens"]) for s in served]
        short = sum(s["error"] is None and len(s["tokens"]) < args.new_tokens for s in served)
        table = rows(run, x2.read_hops(args.metrics, rids))
        summary = summary_line(table, problems(run, rids, failed, table, hidden, short), hidden)
        summary.update(length=args.length, requests=len(served), decoded_tokens=decoded)
        with lines_path.open("w", encoding="utf-8") as out:
            for line in [*table, summary]:
                text = json.dumps(line, allow_nan=False)
                out.write(text + "\n")
                print(text, flush=True)
        rec["summary"] = summary
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
