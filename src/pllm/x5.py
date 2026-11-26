"""X5: crash recovery (R7.12). One stage is killed with requests in flight, then the other.

The runner is a client of a running system, on one laptop or two. It runs on the gateway's
laptop, since it reads the gateway's metrics file, and it kills a worker through that
stage's supervisor endpoint ([supervisor0] and [supervisor1] of the hosts file).

Reference. Before any fault, prompts of the prompt set run one at a time, in order, and the
first REQUESTS of them that give NEW_TOKENS tokens (none stops at end of sequence) are kept
with their tokens. That fault-free run of the same running system is the reference. When
every address of the hosts file is this laptop's, tokens and top logits must match it bit
for bit (one host, fp32, batch 1). Across two laptops the tokens must match, and no
bit-exact claim is made.

One repeat of one stage: wait until both stages are registered and beating and a short
request completes; send the REQUESTS requests together; when every one has FAULT_AT tokens
and none has ended, POST /kill to the stage's supervisor; wait for the requests to end and
for the supervisor to report its new worker. The repeat is then judged from the part of
the gateway's metrics file written since the requests were sent.

Times. detection, respawn, weight load and replay prefill are the four parts of the
gateway's recovery lines (R6.12, pllm.gateway.recovery_parts): detection and replay
prefill are intervals on the gateway's clock, weight load is a duration the new worker
measured on its own clock, and respawn is the gateway's declaration to registration
interval less that load. total is the dead epoch's last beat to the request's first new
token. detection is an upper bound on fault to detection, since the gateway cannot see the
fault. supervisor_respawn_s is the supervisor's own duration, from noticing the exit to its
new worker listening, weight load included; it overlaps the parts and is not one of them.
No time stamps of two clocks are ever subtracted.

A repeat fails, and lists why, when a check fails or cannot be made: the fault did not take
effect (a request ended before it, no death declared, no new epoch or worker, a request
that did not replay from at least FAULT_AT tokens), another death or request in the
window, a missing or incomplete recovery line, a time that is unknown, a divergence, a
token streamed twice or out of order, output unlike the reference, or a dropped result
with no discard line. The exit code is 0 only when every planned repeat of every stage ran
and passed. Fewer than 5 repeats, or another load than 4 requests of 64 tokens with the
fault at token 20, needs --trial, and the record then says so.

One JSON line for the plan, one per reference request, one per repeat, one aggregate per
stage with the spread over its passed repeats (x2.spread), then a summary, go to a .jsonl
file next to the run record in runs/ and to stdout.

Usage (the gateway's laptop, with the system already running):
uv run python -m pllm.x5 smollm2-135m --metrics runs/<gateway>.metrics.jsonl --rtt-ms <ms>
"""

import argparse
import json
import signal
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import grpc

from pllm import client, config, launch, prompts, report, runrecord, x2
from pllm.metrics import STAGE1_GATEWAY
from pllm.models import PINNED

REQUESTS = 4
NEW_TOKENS = 64
FAULT_AT = 20
REPEATS = 5
LENGTH = 128
STAGE_ORDER = (1, 0)  # each repeat kills stage 1, then stage 0, so drift spreads over both
POLL_SECONDS = 0.02
SETTLE_S = 1.5  # after the requests end, for lines still on their way to the file
HTTP_SECONDS = 10.0
PROBE_TOKENS = 2
PARTS = ("detection_ms", "respawn_ms", "weight_load_ms", "replay_prefill_ms")
SPREAD = (*PARTS, "total_ms", "total_ms_max", "supervisor_respawn_s")
SINGLE_HOST = (
    "fault-free run of the same running system on one host (fp32, batch 1): tokens and top "
    "logits are compared bit for bit"
)
TWO_LAPTOPS = (
    "fault-free run of the same running system across two laptops: tokens are compared; no "
    "bit-exact claim"
)

# call(ids, max_tokens) yields the gateway's Tokens for one request
Call = Callable[[list[int], int], Iterator]


class CheckFailed(Exception):
    """Something a repeat needs did not happen or could not be read."""


# ------------------------------------------------------------------ the supervisors


class ControlError(Exception):
    """A request to a supervisor or standby that did not succeed. status is the HTTP
    status, or None when no answer came."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def http_json(endpoint: config.Endpoint, method: str, path: str, timeout: float) -> dict:
    """One request to a control port, with no proxy: both laptops are on one network."""
    url = f"http://{endpoint.address}{path}"
    request = urllib.request.Request(
        url,
        data=b"" if method == "POST" else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")
        raise ControlError(f"{method} {url}: {err.code} {detail}", err.code) from None
    except (OSError, ValueError) as err:
        raise ControlError(f"{method} {url}: {err}") from None
    if not isinstance(body, dict):
        raise ControlError(f"{method} {url}: the answer is not a JSON object")
    return body


class Control:
    """The supervisors of both stages, by their endpoints."""

    def __init__(self, endpoints: dict[int, config.Endpoint], timeout: float = HTTP_SECONDS):
        self.endpoints, self.timeout = endpoints, timeout

    def status(self, stage: int) -> dict:
        return http_json(self.endpoints[stage], "GET", "/status", self.timeout)

    def fault(self, stage: int, action: str) -> int:
        """kill, suspend or resume the stage's worker; the pid the supervisor acted on."""
        body = http_json(self.endpoints[stage], "POST", f"/{action}", self.timeout)
        if body.get("ok") is not True or not isinstance(body.get("pid"), int):
            raise ControlError(f"stage {stage} {action}: unexpected answer {body}")
        return body["pid"]


def require_running(stage: int, status: dict) -> dict:
    """The status of a supervisor whose worker can take a fault."""
    ok = (
        status.get("state") == "running"
        and status.get("ready") is True
        and isinstance(status.get("pid"), int)
        and isinstance(status.get("restarts"), int)
    )
    if not ok:
        raise CheckFailed(f"stage {stage}'s supervisor has no running worker: {status}")
    return status


# ------------------------------------------------------------------ the metrics file


class MetricsTail:
    """The gateway's metrics file, read as it grows. Only whole lines are taken."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        self.bad = 0
        self._offset = 0

    def poll(self) -> list[dict]:
        with self.path.open("rb") as file:
            file.seek(self._offset)
            data = file.read()
        end = data.rfind(b"\n") + 1  # a line still being written is left for the next poll
        self._offset += end
        for raw in data[:end].splitlines():
            try:
                row = json.loads(raw)
            except ValueError:
                row = None
            if isinstance(row, dict):
                self.rows.append(row)
            else:
                self.bad += 1
        return self.rows

    def mark(self) -> int:
        """A position in the file: since(mark) gives the lines written after this call."""
        return len(self.poll())

    def since(self, mark: int) -> list[dict]:
        return self.poll()[mark:]


def of_type(rows: list[dict], kind: str) -> list[dict]:
    return [r for r in rows if r.get("type") == kind]


def deaths(rows: list[dict]) -> list[dict]:
    return [r for r in of_type(rows, "detected") if r.get("declared")]


def live_epochs(rows: list[dict]) -> dict[int, int]:
    """Each stage's latest registered epoch, if it has not been declared dead and has
    beaten. CheckFailed names the first stage for which that is not so."""
    out = {}
    dead = {r.get("epoch") for r in deaths(rows)}
    for stage in report.STAGES:
        epochs = [r.get("new_epoch") for r in of_type(rows, "registered") if r["stage"] == stage]
        if not epochs:
            raise CheckFailed(f"stage {stage} has no registration in the metrics file")
        if epochs[-1] in dead:
            raise CheckFailed(f"stage {stage} epoch {epochs[-1]} is dead, with no replacement")
        beats = of_type(rows, "beat")
        if not any(b.get("stage") == stage and b.get("epoch") == epochs[-1] for b in beats):
            raise CheckFailed(f"stage {stage} epoch {epochs[-1]} has not beaten yet")
        out[stage] = epochs[-1]
    return out


# ------------------------------------------------------------------ the load


@dataclass
class Job:
    """One request: its prompt and what the client received."""

    prompt: int  # index in the prompt set
    ids: list[int]
    rid: str | None = None
    tokens: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    error: str | None = None
    done: bool = False
    at_fault: int | None = None  # tokens the client had when the fault was sent

    def fresh(self) -> "Job":
        return Job(self.prompt, self.ids)


def serve_one(call: Call, job: Job, max_tokens: int, changed=None) -> None:
    changed = changed or threading.Condition()
    try:
        for token in call(job.ids, max_tokens):
            with changed:
                job.rid = token.rid
                job.tokens.append(token.token)
                job.steps.append(token.step)
                job.logits.append(token.top_logit)
                changed.notify_all()
    except grpc.RpcError as err:
        job.error = f"{err.code().name}: {err.details()}"
    except Exception as err:  # the request did not finish: never a silent pass
        job.error = f"{type(err).__name__}: {err}"
    finally:
        with changed:
            job.done = True
            changed.notify_all()


class Load:
    """The requests of one repeat, each on its own daemon thread, all sent together."""

    def __init__(self, call: Call, jobs: list[Job], new_tokens: int):
        self.jobs = jobs
        self.changed = threading.Condition()
        self.threads = [
            threading.Thread(
                target=serve_one, args=(call, job, new_tokens, self.changed), daemon=True
            )
            for job in jobs
        ]

    def start(self) -> "Load":
        for thread in self.threads:
            thread.start()
        return self

    def wait_for_tokens(self, count: int, timeout: float) -> None:
        """Wait until every request has count tokens and is still running, and note how
        many each has. CheckFailed if one ends first: the fault would miss it."""
        end = time.monotonic() + timeout
        with self.changed:
            while True:
                ended = [job for job in self.jobs if job.done]
                if ended:
                    raise CheckFailed(
                        "a request ended before the fault point: "
                        + "; ".join(
                            f"prompt {j.prompt} after {len(j.tokens)} tokens ({j.error})"
                            for j in ended
                        )
                    )
                if all(len(job.tokens) >= count for job in self.jobs):
                    for job in self.jobs:
                        job.at_fault = len(job.tokens)
                    return
                if time.monotonic() > end:
                    raise CheckFailed(f"not every request reached {count} tokens in {timeout} s")
                self.changed.wait(0.1)  # a timeout, so Ctrl-C is seen

    def join(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        for thread in self.threads:
            while thread.is_alive() and time.monotonic() < end:
                thread.join(0.1)
        if any(thread.is_alive() for thread in self.threads):
            raise CheckFailed(f"a request had not ended after {timeout} s")


def wait_until(check: Callable[[], object], what: str, timeout: float):
    """The first value of check() that is not None; CheckFailed after timeout."""
    end = time.monotonic() + timeout
    while True:
        value = check()
        if value is not None:
            return value
        if time.monotonic() > end:
            raise CheckFailed(f"timed out after {timeout} s waiting for {what}")
        time.sleep(POLL_SECONDS)


@dataclass
class Context:
    """What every repeat uses."""

    call: Call
    tail: MetricsTail
    control: Control
    single_host: bool
    requests: int = REQUESTS
    new_tokens: int = NEW_TOKENS
    fault_at: int = FAULT_AT
    timeout: float = 300.0  # for the requests of a repeat to reach a point or end
    wait: float = 180.0  # for the system to react or to be healthy again
    settle: float = SETTLE_S
    probe_ids: tuple[int, ...] = (1, 2, 3)  # the health check's prompt

    def healthy(self) -> dict[int, int]:
        """Wait until both stages are live and a short request completes; their epochs.
        A request sent just after a registration can be refused until both route updates
        have arrived, so it is sent again."""
        end = time.monotonic() + self.wait
        why = "never checked"
        while True:
            try:
                live_epochs(self.tail.poll())
                probe = Job(-1, list(self.probe_ids))
                serve_one(self.call, probe, PROBE_TOKENS)
                if probe.error is None and probe.tokens:
                    return live_epochs(self.tail.poll())
                why = f"a probe request failed: {probe.error}"
            except CheckFailed as err:
                why = str(err)
            if time.monotonic() > end:
                raise CheckFailed(f"the system was not healthy after {self.wait} s: {why}")
            time.sleep(0.5)

    def finish(self, load: Load, failures: list[str]) -> None:
        """Wait for the requests to end and for the gateway to write their end lines."""
        try:
            load.join(self.timeout)
            rids = {job.rid for job in load.jobs if job.rid}

            def ended():
                found = {r.get("rid") for r in of_type(self.tail.poll(), "request")}
                return True if rids <= found else None

            wait_until(ended, "the gateway's request lines", self.wait)
        except CheckFailed as err:
            failures.append(str(err))


def references(
    ctx: Context, prompt_ids: list[list[int]], emit: Callable[[dict], None]
) -> list[Job]:
    """The fault-free reference: the first prompts, one request at a time, that give
    new_tokens tokens. CheckFailed if there are too few or the run was not fault free."""
    ctx.healthy()
    mark = ctx.tail.mark()
    kept: list[Job] = []
    for index, ids in enumerate(prompt_ids):
        job = Job(index, ids)
        serve_one(ctx.call, job, ctx.new_tokens)
        if job.error is not None:
            raise CheckFailed(f"the reference request of prompt {index} failed: {job.error}")
        full = len(job.tokens) == ctx.new_tokens and job.steps == list(range(ctx.new_tokens))
        emit(
            {
                "type": "reference",
                "prompt": index,
                "used": full,
                "output_tokens": len(job.tokens),
                "tokens": job.tokens,
                "top_logits": job.logits,
            }
        )
        if full:
            kept.append(job)
        if len(kept) == ctx.requests:
            break
    if len(kept) < ctx.requests:
        raise CheckFailed(
            f"only {len(kept)} of {len(prompt_ids)} prompts give {ctx.new_tokens} tokens; "
            f"{ctx.requests} are needed"
        )
    time.sleep(ctx.settle)
    rows = ctx.tail.since(mark)
    if deaths(rows) or of_type(rows, "recovery"):
        raise CheckFailed("the reference run was not fault free: a death or a replay happened")
    return kept


# ------------------------------------------------------------------ judging one repeat


def _ms(ns) -> float | None:
    return None if ns is None else ns / 1e6


def _mean(values: list) -> float | None:
    """None if any value is unknown: a mean over the known ones would hide it."""
    if not values or any(v is None for v in values):
        return None
    return statistics.fmean(values)


def client_checks(job: Job, ref: Job, new_tokens: int, single_host: bool) -> tuple[dict, list]:
    """What the client received for one request against its reference."""
    failures = []
    name = f"prompt {job.prompt}"
    duplicates = len(job.steps) - len(set(job.steps))
    in_order = job.steps == list(range(new_tokens))
    equal = job.tokens == ref.tokens
    logits_equal = (job.logits == ref.logits) if single_host else None
    if job.error is not None:
        failures.append(f"{name}: the request failed: {job.error}")
    if job.rid is None:
        failures.append(f"{name}: no token came, so the request cannot be checked")
    if duplicates:
        failures.append(f"{name}: {duplicates} duplicate tokens at the client")
    if not in_order:
        failures.append(f"{name}: the client's steps are not 0 to {new_tokens - 1} in order")
    if not equal:
        failures.append(f"{name}: output differs from the fault-free reference")
    if logits_equal is False:
        failures.append(f"{name}: top logits differ from the reference on one host")
    row = {
        "rid": job.rid,
        "prompt": job.prompt,
        "tokens_at_fault": job.at_fault,
        "output_tokens": len(job.tokens),
        "duplicate_tokens": duplicates,
        "output_equals_reference": equal,
        "top_logits_equal_reference": logits_equal,
    }
    return row, failures


def _key(row: dict) -> tuple:
    return row.get("rid"), row.get("attempt"), row.get("step")


def discard_name(row: dict) -> str:
    """The reason of a discard line; for a Nack, with why the stage refused the step."""
    if row.get("source") == "nack":
        return f"{row.get('reason')}:{row.get('nack_reason')}"
    return str(row.get("reason"))


def analyse(
    rows: list[dict],
    jobs: list[Job],
    refs: list[Job],
    *,
    stage: int,
    old_epoch: int | None,
    fault_at: int,
    new_tokens: int,
    single_host: bool,
    death_path: str | None = None,
) -> tuple[dict, list[str]]:
    """Judge one repeat from the metrics lines written since its requests were sent and
    from what the client received. Returns the repeat's numbers and why it fails, if it
    does. death_path is the detection path the death must have, or None for any."""
    failures: list[str] = []
    per_request = []
    for job, ref in zip(jobs, refs, strict=True):
        row, found = client_checks(job, ref, new_tokens, single_host)
        per_request.append(row)
        failures += found
    rids = {job.rid for job in jobs if job.rid}
    if len(rids) != len(jobs):
        failures.append(f"{len(jobs)} requests were sent but {len(rids)} request ids are known")

    # the death and its replacement
    declared = deaths(rows)
    ours = [d for d in declared if d.get("stage") == stage and d.get("epoch") == old_epoch]
    for d in declared:
        if d not in ours:
            failures.append(
                f"another death in the window: stage {d.get('stage')} epoch {d.get('epoch')} "
                f"({d.get('path')})"
            )
    path = ours[0].get("path") if ours else None
    if len(ours) != 1:
        failures.append(
            f"the fault did not take effect: {len(ours)} declarations of stage {stage} "
            f"epoch {old_epoch}"
        )
    elif death_path is not None and path != death_path:
        failures.append(f"the death was detected by {path}, not {death_path}")
    replaced = [
        r
        for r in of_type(rows, "registered")
        if r.get("stage") == stage and r.get("old_epoch") == old_epoch
    ]
    new_epoch = replaced[0].get("new_epoch") if replaced else None
    if new_epoch is None or old_epoch is None or new_epoch <= old_epoch:
        failures.append(f"no new epoch of stage {stage} registered after epoch {old_epoch}")

    # the gateway's request lines
    ends = of_type(rows, "request")
    for r in ends:
        if r.get("rid") not in rids:
            failures.append(f"another request ran in the window: {r.get('rid')}")
    for rid in sorted(rids):
        lines = [r for r in ends if r.get("rid") == rid]
        if len(lines) != 1:
            failures.append(f"{rid}: {len(lines)} request lines in the metrics file")
        elif (
            lines[0].get("outcome") not in report.COMPLETED
            or lines[0].get("output_tokens") != new_tokens
        ):
            failures.append(
                f"{rid}: the gateway ended it as {lines[0].get('outcome')} with "
                f"{lines[0].get('output_tokens')} tokens"
            )

    # results dropped, and one discard line for each one dropped on arrival
    results = [r for r in of_type(rows, "hop") if r.get("hop") == STAGE1_GATEWAY]
    rejected = [r for r in results if r.get("accepted") is False]
    discards = of_type(rows, "discard")
    written = Counter(_key(r) for r in discards if r.get("source") == "result")
    missing = Counter(_key(r) for r in rejected) - written
    if missing:
        failures.append(
            f"{sum(missing.values())} dropped results have no discard line, so their reasons "
            "are unknown"
        )
    accepted = Counter(_key(r) for r in results if r.get("accepted") is True)
    twice = sum(n - 1 for n in accepted.values())
    if twice:
        failures.append(f"the gateway accepted {twice} results twice")
    for r in of_type(rows, "divergence"):
        failures.append(f"{r.get('rid')}: replay diverged at step {r.get('step')}")

    # the recovery lines: the four parts, per request
    recoveries = []
    for r in of_type(rows, "recovery"):
        if all(key in r for key in report.REQUIRED["recovery"]):
            recoveries.append(r)
        else:
            failures.append(f"an incomplete recovery line: {r}")
    run = report.Run(Path("window"), recoveries=recoveries)
    run.dropped = [(r.get("rid"), r.get("attempt")) for r in rejected]
    columns = report.RECOVERY_HEADER
    stale = {
        row[columns.index("rid")]: row[columns.index("stale_discarded")]
        for row in report.recovery_rows(run)
    }
    for r in recoveries:
        if r["rid"] not in rids:
            failures.append(f"another request recovered in the window: {r['rid']}")
    for row in per_request:
        rid = row["rid"]
        lines = [r for r in recoveries if r["rid"] == rid]
        line = lines[0] if len(lines) == 1 else {}
        if len(lines) != 1:
            failures.append(
                f"prompt {row['prompt']} ({rid}): {len(lines)} recovery lines, not 1: the "
                "fault did not take effect on it as planned"
            )
        parts = {
            "detection_ms": _ms(line.get("detection_ns")),
            "respawn_ms": _ms(line.get("respawn_ns")),
            "weight_load_ms": _ms(line.get("weight_load_ns")),
            "replay_prefill_ms": _ms(line.get("replay_prefill_ns")),
            "total_ms": _ms(line.get("total_ns")),
        }
        row.update(
            parts,
            attempt=line.get("attempt"),
            overlap=line.get("overlap"),
            recomputed_tokens=line.get("recomputed_tokens"),
            divergences=line.get("divergences"),
            stale_discarded=stale.get(rid),
        )
        if not line:
            continue
        if line.get("recovered") is not True:
            failures.append(f"{rid}: not recovered ({line.get('outcome')})")
        if (line.get("stage"), line.get("old_epoch"), line.get("new_epoch")) != (
            stage,
            old_epoch,
            new_epoch,
        ):
            failures.append(
                f"{rid}: its recovery names stage {line.get('stage')} epoch "
                f"{line.get('old_epoch')} to {line.get('new_epoch')}, not stage {stage} "
                f"epoch {old_epoch} to {new_epoch}"
            )
        unknown = [name for name, value in parts.items() if value is None]
        if unknown:
            failures.append(f"{rid}: unknown {', '.join(unknown)} in its recovery line")
        if line.get("divergences") != 0:
            failures.append(f"{rid}: divergences {line.get('divergences')}")
        recomputed = line.get("recomputed_tokens")
        if not isinstance(recomputed, int) or not fault_at <= recomputed < new_tokens:
            failures.append(
                f"{rid}: replayed {recomputed} tokens, so the stage did not die between "
                f"token {fault_at} and token {new_tokens}"
            )

    totals = [row["total_ms"] for row in per_request]
    data = {
        "old_epoch": old_epoch,
        "new_epoch": new_epoch,
        "path": path,
        # one death and one registration, so the same for every request: any one of them
        **{name: per_request[0][name] for name in PARTS[:3]},
        "overlap": per_request[0]["overlap"],
        "replay_prefill_ms": _mean([row["replay_prefill_ms"] for row in per_request]),
        "total_ms": _mean(totals),
        "total_ms_max": None if _mean(totals) is None else max(totals),
        "duplicate_tokens": sum(row["duplicate_tokens"] for row in per_request),
        "discards_by_reason": dict(sorted(Counter(map(discard_name, discards)).items())),
        "requests": per_request,
    }
    return data, failures


def repeat_line(experiment: str, stage: int, repeat: int, data: dict, failures: list) -> dict:
    return {
        "type": "repeat",
        "experiment": experiment,
        "stage": stage,
        "repeat": repeat,
        "passed": not failures,
        "failures": failures,
        **data,
    }


# ------------------------------------------------------------------ one X5 repeat


def wait_respawn(ctx: Context, stage: int, before: dict) -> dict:
    """The supervisor's status once its next worker listens."""

    def respawned():
        try:
            status = ctx.control.status(stage)
        except ControlError:
            return None
        fresh = (
            status.get("state") == "running"
            and status.get("ready") is True
            and status.get("pid") not in (None, before["pid"])
            and isinstance(status.get("restarts"), int)
            and status["restarts"] > before["restarts"]
        )
        return status if fresh else None

    return wait_until(respawned, f"stage {stage}'s supervisor to report a new worker", ctx.wait)


def x5_repeat(
    ctx: Context, stage: int, repeat: int, refs: list[Job], epochs: dict[int, int]
) -> dict:
    jobs = [ref.fresh() for ref in refs]
    old = epochs[stage]
    failures: list[str] = []
    supervisor = {"pid_before": None, "pid_after": None, "supervisor_respawn_s": None}
    load, rows = None, []
    mark, bad = ctx.tail.mark(), ctx.tail.bad
    try:
        before = require_running(stage, ctx.control.status(stage))
        supervisor["pid_before"] = before["pid"]
        load = Load(ctx.call, jobs, ctx.new_tokens).start()
        load.wait_for_tokens(ctx.fault_at, ctx.timeout)
        pid = ctx.control.fault(stage, "kill")
        if pid != before["pid"]:
            failures.append(f"the supervisor killed pid {pid}, not pid {before['pid']}")
        after = wait_respawn(ctx, stage, before)
        supervisor["pid_after"] = after["pid"]
        supervisor["supervisor_respawn_s"] = after.get("last_respawn_seconds")
        if after["restarts"] != before["restarts"] + 1:
            failures.append(
                f"the worker was restarted {after['restarts'] - before['restarts']} times"
            )
        if supervisor["supervisor_respawn_s"] is None:
            failures.append("the supervisor reports no respawn time")
    except (CheckFailed, ControlError) as err:
        failures.append(str(err))
    if load is not None:
        ctx.finish(load, failures)
        time.sleep(ctx.settle)
        rows = ctx.tail.since(mark)
    if ctx.tail.bad != bad:
        failures.append(f"{ctx.tail.bad - bad} unreadable metrics lines in the window")
    data, found = analyse(
        rows,
        jobs,
        refs,
        stage=stage,
        old_epoch=old,
        fault_at=ctx.fault_at,
        new_tokens=ctx.new_tokens,
        single_host=ctx.single_host,
    )
    return repeat_line("x5", stage, repeat, {**data, **supervisor}, failures + found)


# ------------------------------------------------------------------ the experiment


def aggregate(experiment: str, lines: list[dict], names=SPREAD) -> list[dict]:
    """One line per stage: the spread of each time over the repeats that passed."""
    out = []
    for stage in sorted({line["stage"] for line in lines}, reverse=True):
        mine = [line for line in lines if line["stage"] == stage]
        passed = [line for line in mine if line["passed"]]
        counted = {}
        for key in ("discards_by_reason", "late_by_reason"):
            if any(key in line for line in mine):
                total: Counter = Counter()
                for line in passed:
                    total.update(line.get(key) or {})
                counted[key] = dict(sorted(total.items()))
        out.append(
            {
                "type": "aggregate",
                "experiment": experiment,
                "stage": stage,
                "repeats": len(mine),
                "passed_repeats": len(passed),
                "paths": sorted({line["path"] for line in passed if line.get("path")}),
                **{name: x2.spread([line.get(name) for line in passed]) for name in names},
                **counted,
            }
        )
    return out


def summary_line(experiment: str, lines: list[dict], planned: int, error: str | None) -> dict:
    """planned is the number of repeats that must pass. A line marked replaced is an attempt
    that showed nothing and was run again: it is listed and does not count as a repeat."""
    counted = [line.get("duplicate_tokens") for line in lines]
    kept = [line for line in lines if not line.get("replaced")]

    def listed(line: dict) -> dict:
        return {"stage": line["stage"], "repeat": line["repeat"], "failures": line["failures"]}

    failed = [listed(line) for line in kept if not line["passed"]]
    return {
        "type": "summary",
        "experiment": experiment,
        "planned_repeats": planned,
        "ran_repeats": len(lines),
        "passed_repeats": sum(line["passed"] for line in lines),
        "failed": failed,
        "replaced": [listed(line) for line in lines if line.get("replaced")],
        "error": error,
        # unknown, not zero, unless every planned repeat counted it
        "duplicate_tokens": sum(counted) if len(kept) == planned and None not in counted else None,
        "passed": error is None and planned > 0 and len(kept) == planned and not failed,
    }


def run_experiment(
    ctx: Context,
    experiment: str,
    stages: list[int],
    repeats: int,
    prompt_ids: list[list[int]],
    one_repeat: Callable,
    emit: Callable[[dict], None],
    aggregates: Callable[[list[dict]], list[dict]],
    again: Callable[[dict], bool] | None = None,
    extra_attempts: int = 0,
) -> tuple[list[dict], dict]:
    """The reference, then every stage in each repeat, then aggregates and the summary.
    A system that does not become healthy again ends the run: what did not run fails it.
    A repeat for which again(line) is true is marked replaced and run again, at most
    extra_attempts times for each stage in the whole run."""
    lines: list[dict] = []
    error = None
    emit(
        {
            "type": "plan",
            "experiment": experiment,
            "stages": stages,
            "repeats": repeats,
            "requests": ctx.requests,
            "new_tokens": ctx.new_tokens,
            "fault_at": ctx.fault_at,
            "single_host": ctx.single_host,
            "reference": SINGLE_HOST if ctx.single_host else TWO_LAPTOPS,
        }
    )
    ctx.probe_ids = tuple(prompt_ids[0][:8])
    try:
        refs = references(ctx, prompt_ids, emit)
        left = dict.fromkeys(stages, extra_attempts)
        for repeat in range(repeats):
            for stage in stages:
                retry = True
                while retry:
                    line = one_repeat(ctx, stage, repeat, refs, ctx.healthy())
                    retry = again is not None and left[stage] > 0 and again(line)
                    if retry:
                        left[stage] -= 1
                        line["replaced"] = True
                    lines.append(line)
                    emit(line)
    except (CheckFailed, ControlError) as err:
        error = str(err)
    for line in aggregates(lines):
        emit(line)
    summary = summary_line(experiment, lines, repeats * len(stages), error)
    emit(summary)
    return lines, summary


# ------------------------------------------------------------------ command line


def local_to_this_laptop(
    hosts: config.HostConfig,
    controls: dict[int, config.Endpoint],
    local: set[str] | None = None,
) -> bool:
    """True when every process and supervisor of the hosts file is on this laptop."""
    local = launch.this_laptop() if local is None else local
    endpoints = [getattr(hosts, role) for role in config.ROLES] + list(controls.values())
    try:
        return all(socket.gethostbyname(end.host) in local for end in endpoints)
    except OSError:  # a name that does not resolve is not this laptop's
        return False


@contextmanager
def interrupts_raise():
    """Make a termination signal or a closed terminal raise KeyboardInterrupt, as Ctrl-C
    does, so every cleanup runs. A forced kill cannot be caught."""
    previous = {}

    def interrupt(*_):
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):  # not every system has each
            if hasattr(signal, name):
                number = getattr(signal, name)
                previous[number] = signal.signal(number, interrupt)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def parser_for(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--metrics", type=Path, required=True, help="the gateway's metrics file")
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=report.STAGES,
        default=list(STAGE_ORDER),
        help="the stages to fault, in the order of one repeat (default 1 0)",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--requests", type=int, default=REQUESTS, help="in flight at once")
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument("--fault-at", type=int, default=FAULT_AT, help="tokens before the fault")
    parser.add_argument("--length", type=int, choices=prompts.LENGTHS, default=LENGTH)
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    parser.add_argument(
        "--wait", type=float, default=180.0, help="seconds for the system to react or recover"
    )
    parser.add_argument("--settle", type=float, default=SETTLE_S, help="seconds")
    parser.add_argument("--rtt-ms", type=float, help="round trip from pllm.netcheck")
    parser.add_argument(
        "--trial", action="store_true", help="allow fewer repeats or another load; recorded"
    )
    return parser


def check_args(parser: argparse.ArgumentParser, args) -> tuple[config.HostConfig, dict, bool, dict]:
    """The checks every fault experiment makes before it sends anything. Returns the hosts,
    the supervisors' endpoints, whether it is one host, and the gateway's run line."""
    if len(set(args.stages)) != len(args.stages):
        parser.error("--stages names a stage twice")
    if args.repeats < 1 or args.requests < 1 or not 0 < args.fault_at < args.new_tokens:
        parser.error("--repeats and --requests must be positive, 0 < --fault-at < --new-tokens")
    if args.requests > prompts.PASSAGES:
        parser.error(f"--requests can be at most {prompts.PASSAGES}")
    if not args.trial:
        if args.repeats < REPEATS:
            parser.error(f"a result needs at least {REPEATS} repeats; --trial allows fewer")
        if (args.requests, args.new_tokens, args.fault_at) != (REQUESTS, NEW_TOKENS, FAULT_AT):
            parser.error(
                f"the experiment is {REQUESTS} requests of {NEW_TOKENS} tokens with the "
                f"fault at token {FAULT_AT}; --trial allows another load"
            )
    if min(args.timeout, args.wait) <= 0 or args.settle < 0:
        parser.error("--timeout and --wait must be positive, --settle not negative")
    if not args.metrics.is_file():
        parser.error("--metrics must be the running gateway's metrics file")
    run = report.load(args.metrics)
    if run.info.get("model") != args.model:
        parser.error(f"the gateway serves {run.info.get('model')}, not {args.model}")
    if run.end is not None:
        parser.error("the metrics file has ended: name the running gateway's file")
    if not run.info.get("discard_lines"):
        parser.error(
            "the gateway does not write discard lines, so dropped outputs would be unknown: "
            "start it from this code"
        )
    try:
        hosts = config.load()
        controls = {stage: config.supervisor_endpoint(stage) for stage in report.STAGES}
    except (ValueError, OSError) as err:
        parser.error(str(err))
    single = local_to_this_laptop(hosts, controls)
    if not single and args.rtt_ms is None:
        parser.error("a run across laptops records its round trip: give --rtt-ms")
    return hosts, controls, single, run.info


def gateway_call(model: str, address: str, timeout: float) -> Call:
    tokenizer = prompts.load_tokenizer(model)

    def call(ids, max_tokens):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        return client.generate(address, text, max_tokens, timeout)

    return call


def record_and_run(script: str, args, info: dict, single: bool, run: Callable) -> int:
    """Write the run record and the result lines around run(emit), which returns the
    repeat lines and the summary."""
    spec = PINNED[args.model]
    with runrecord.record(
        script,
        model=args.model,
        revision=spec.revision,
        split=config.split_point(args.model),
        rtt_ms=args.rtt_ms,
    ) as rec:
        rec.update(
            argv=sys.argv,
            trial=args.trial,
            stages=args.stages,
            repeats=args.repeats,
            requests=args.requests,
            new_tokens=args.new_tokens,
            fault_at=args.fault_at,
            length=args.length,
            timeout_s=args.timeout,
            wait_s=args.wait,
            settle_s=args.settle,
            single_host=single,
            reference=SINGLE_HOST if single else TWO_LAPTOPS,
            metrics_file=args.metrics.name,
            gateway_commit=info.get("commit"),
            heartbeat_timeout_s=info.get("heartbeat_timeout_s"),
            **getattr(args, "record", {}),  # an experiment's own settings
        )
        lines_path = runrecord.output_path(rec, ".jsonl")
        rec["results_file"] = lines_path.name
        with lines_path.open("w", encoding="utf-8") as out:

            def emit(line: dict) -> None:
                text = json.dumps(line, allow_nan=False)
                out.write(text + "\n")
                out.flush()
                print(text, flush=True)

            with interrupts_raise():
                _, summary = run(emit)
        rec["summary"] = summary
    if not summary["passed"]:
        if summary["error"]:
            print(f"failed: {summary['error']}", file=sys.stderr)
        for failed in summary["failed"]:
            for reason in failed["failures"]:
                print(
                    f"failed: stage {failed['stage']} repeat {failed['repeat']}: {reason}",
                    file=sys.stderr,
                )
    return 0 if summary["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = parser_for("X5: kill each stage with requests in flight and measure recovery.")
    args = parser.parse_args(argv)
    hosts, controls, single, info = check_args(parser, args)
    prompt_ids = prompts.prompts(args.model, args.length)
    ctx = Context(
        gateway_call(args.model, hosts.gateway.address, args.timeout),
        MetricsTail(args.metrics),
        Control(controls),
        single,
        requests=args.requests,
        new_tokens=args.new_tokens,
        fault_at=args.fault_at,
        timeout=args.timeout,
        wait=args.wait,
        settle=args.settle,
    )

    def run(emit):
        return run_experiment(
            ctx,
            "x5",
            args.stages,
            args.repeats,
            prompt_ids,
            x5_repeat,
            emit,
            lambda lines: aggregate("x5", lines),
        )

    return record_and_run("x5", args, info, single, run)


if __name__ == "__main__":
    sys.exit(main())
