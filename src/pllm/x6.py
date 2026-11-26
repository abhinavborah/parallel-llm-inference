"""X6: stale outputs after a suspend (R7.13, R6.8). One stage is suspended past the
heartbeat timeout, a replacement serves, the old process is resumed; then the other stage.

The runner is a client of a running system, as pllm.x5 is, with the same reference, load
and checks. A supervisor does not restart a suspended worker, so each faulted stage's laptop
also runs pllm.standby, which starts the replacement when the runner asks.

One repeat of one stage: send the 4 requests; at FAULT_AT tokens POST /suspend to the
stage's supervisor; wait for the gateway to declare the stage's epoch dead by the heartbeat
timeout; ask the standby to start the replacement; wait until it has registered and the
gateway has accepted a result of a replayed request after that; check that the old process
is still there; POST /resume; wait for a late output of it. The repeat is judged from the
metrics lines written since the requests were sent. Then the runner restores the system: it
kills the old worker through its supervisor, which starts a new one that takes over from
the replacement, and stops the replacement.

The suspended worker is resumed on every way out, errors and Ctrl-C included. A runner
that is killed outright cannot do that: resume by hand through the supervisor.

Late outputs are the resumed process's, read from the discard lines written after the
resume was sent (a position in the file, not a clock): for stage 1, results carrying the
old epoch; for stage 0, whose output goes to stage 1, stage 1's BAD_SENDER Nacks for an
attempt from before the fault. late_by_reason counts them by the gateway's reason;
discards_by_reason counts every drop in the window.

A stage can send a late output only if it held one when it was suspended. Stage 0 holds
the steps the gateway goes on sending it, since the gateway half-closes its old stream.
Stage 1 loses what it had not yet read when stage 0 moves its route to the replacement
and closes the old channel, so it sends a late result only if it was suspended with a step
in hand. A repeat in which it held none has not shown rejection, and fails.
--extra-attempts N (default 0) lets the runner run such a repeat again, at most N times
per stage, when that is its only failure. Every attempt stays in the output, marked
replaced, and is listed in the summary. A repeat with any other failure is never run
again.

A repeat fails where pllm.x5 would, and when: the death is not detected by the heartbeat
timeout; the replacement does not serve; the old process is gone before the resume or
after its late output; the stage was suspended for as long as the keepalive timeout (8 s)
or longer, after which its connections close and late outputs can be lost; no late output
arrived, so rejection was not shown; a late output was accepted, or for stage 0 stage 1
ran it; or the system could not be restored.

held_s, the time from sending the suspend to sending the resume, is on the runner's clock.
detection, respawn, weight load and replay prefill are the gateway's recovery parts
(pllm.x5). Here respawn runs to the replacement's registration and so includes the
runner's reaction and the standby's start: it is not a supervisor's respawn.

Usage (the gateway's laptop; the system and a standby per faulted stage already running):
uv run python -m pllm.x6 smollm2-135m --metrics runs/<gateway>.metrics.jsonl --rtt-ms <ms>
"""

import signal
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

from pllm import config, prompts, standby, x5
from pllm.metrics import GATEWAY_STAGE0, STAGE1_GATEWAY
from pllm.wire import KEEPALIVE_TIMEOUT_SECONDS
from pllm.x5 import CheckFailed, ControlError, Job, of_type

LATE_WAIT_S = 10.0
NOT_SHOWN = (
    "no late output of the resumed stage reached the gateway, so its rejection was not shown"
)
RESUME_TRIES = 10
RESUME_PAUSE_S = 1.0
STANDBY_STOP_SECONDS = 45.0  # longer than the supervisor's own 30 s stop
SPREAD = (*x5.PARTS, "total_ms", "total_ms_max", "held_s", "late_outputs")


class ResumeFailed(ControlError):
    """A suspended stage that may still be suspended."""


class StandbyControl:
    """The standbys of the stages, by their endpoints."""

    def __init__(self, endpoints: dict[int, config.Endpoint]):
        self.endpoints = endpoints

    def status(self, stage: int) -> dict:
        return x5.http_json(self.endpoints[stage], "GET", "/status", x5.HTTP_SECONDS)

    def post(self, stage: int, action: str) -> None:
        timeout = STANDBY_STOP_SECONDS if action == "stop" else x5.HTTP_SECONDS
        body = x5.http_json(self.endpoints[stage], "POST", f"/{action}", timeout)
        if body.get("ok") is not True:
            raise ControlError(f"standby {stage} {action}: unexpected answer {body}")


@dataclass
class Context(x5.Context):
    standby: StandbyControl | None = None
    late_wait: float = LATE_WAIT_S


# ------------------------------------------------------------------ suspend and resume


@contextmanager
def signals_ignored():
    """Ignore Ctrl-C and termination for a moment, so a second one cannot stop a resume."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):  # not all everywhere
            if hasattr(signal, name):
                number = getattr(signal, name)
                previous[number] = signal.signal(number, signal.SIG_IGN)
    try:
        yield
    finally:
        for number, handler in previous.items():
            if handler is not None:
                signal.signal(number, handler)


class Suspension:
    def __init__(self, control: x5.Control, stage: int):
        self.control, self.stage = control, stage
        self.pid: int | None = None
        self.began = time.monotonic()
        self.resumed = False
        self.held_s: float | None = None  # from sending the suspend to sending the resume

    def resume(self) -> int | None:
        """Resume the worker, trying again until the supervisor answers. Returns the pid it
        resumed, or None if it was resumed already or no worker is there. Safe to repeat."""
        if self.resumed:
            return None
        last = None
        with signals_ignored():
            for attempt in range(RESUME_TRIES):
                sent = time.monotonic()
                try:
                    pid = self.control.fault(self.stage, "resume")
                except ControlError as err:
                    if err.status == 409:  # no worker: nothing is left suspended
                        self.resumed = True
                        return None
                    last = err
                    if attempt < RESUME_TRIES - 1:
                        time.sleep(RESUME_PAUSE_S)
                    continue
                self.resumed = True
                self.held_s = sent - self.began
                return pid
        address = self.control.endpoints[self.stage].address
        message = (
            f"STAGE {self.stage} MAY STILL BE SUSPENDED. Resume it by hand: curl -X POST "
            f'-H "Content-Type: application/json" http://{address}/resume ({last})'
        )
        print(message, file=sys.stderr, flush=True)
        raise ResumeFailed(message)


@contextmanager
def suspended(control: x5.Control, stage: int):
    """Suspend a stage's worker for the length of the block, and resume it on every way
    out. The resume is also sent when the suspend itself fails, since a request whose
    answer was lost may have been applied."""
    hold = Suspension(control, stage)
    try:
        hold.pid = control.fault(stage, "suspend")
        yield hold
    finally:
        hold.resume()


# ------------------------------------------------------------------ reading the window


def attempts_before(rows: list[dict], rids: set[str]) -> dict[str, int]:
    """Each request's highest attempt among the steps the gateway sent in these rows."""
    out: dict[str, int] = {}
    for r in of_type(rows, "hop"):
        if r.get("hop") == GATEWAY_STAGE0 and r.get("rid") in rids:
            out[r["rid"]] = max(out.get(r["rid"], 0), r["attempt"])
    return out


def death_index(rows: list[dict], stage: int, epoch: int) -> int | None:
    for i, r in enumerate(rows):
        if (
            r.get("type") == "detected"
            and r.get("declared")
            and (r.get("stage"), r.get("epoch")) == (stage, epoch)
        ):
            return i
    return None


def replacement_index(rows: list[dict], stage: int, old_epoch: int) -> int | None:
    for i, r in enumerate(rows):
        if r.get("type") == "registered" and (r.get("stage"), r.get("old_epoch")) == (
            stage,
            old_epoch,
        ):
            return i
    return None


def served(rows: list[dict], old_attempts: dict[str, int]) -> bool:
    """True once the gateway has accepted a result under an attempt newer than the one
    from before the fault. Pass the rows written after the replacement registered."""
    return any(
        r.get("hop") == STAGE1_GATEWAY
        and r.get("accepted") is True
        and r.get("rid") in old_attempts
        and r["attempt"] > old_attempts[r["rid"]]
        for r in of_type(rows, "hop")
    )


def late_outputs(
    rows: list[dict], stage: int, old_epoch: int, old_attempts: dict[str, int]
) -> list[dict]:
    """The discard lines of the old process's outputs. Pass the rows written after the
    resume was sent."""
    out = []
    for r in of_type(rows, "discard"):
        if r.get("rid") not in old_attempts:
            continue
        if stage == 1:
            ours = r.get("source") == "result" and r.get("epoch") == old_epoch
        else:
            ours = (
                r.get("source") == "nack"
                and r.get("stage") == 1
                and r.get("nack_reason") == "BAD_SENDER"
                and r.get("attempt") <= old_attempts[r["rid"]]
            )
        if ours:
            out.append(r)
    return out


def analyse_stale(
    rows: list[dict],
    *,
    stage: int,
    old_epoch: int,
    rids: set[str],
    resume_at: int | None,
    held_s: float | None,
) -> tuple[dict, list[str]]:
    """The part of a repeat that is X6's own. rows is the repeat's window and resume_at
    the position in it when the resume was sent, None if the replacement never served and
    the old process was resumed only to clean up."""
    failures: list[str] = []
    died = death_index(rows, stage, old_epoch)
    old_attempts = {} if died is None else attempts_before(rows[:died], rids)
    if not rids or set(old_attempts) != rids:
        failures.append("the attempt each request had before the fault is unknown")
    registered = replacement_index(rows, stage, old_epoch)
    if (
        resume_at is None
        or registered is None
        or not served(rows[registered:resume_at], old_attempts)
    ):
        failures.append("the stage was not resumed after its replacement had served")
        resume_at = None
    if held_s is None:
        failures.append("how long the stage was suspended is unknown")
    elif held_s >= KEEPALIVE_TIMEOUT_SECONDS:
        failures.append(
            f"the stage was suspended for {held_s:.1f} s, not under the "
            f"{KEEPALIVE_TIMEOUT_SECONDS:.0f} s keepalive timeout: late outputs can be lost "
            "with the closed connections"
        )
    data = {"held_s": held_s, "late_outputs": None, "late_by_reason": {}}
    if resume_at is None:
        return data, failures
    after = rows[resume_at:]
    late = late_outputs(after, stage, old_epoch, old_attempts)
    data["late_outputs"] = len(late)
    data["late_by_reason"] = dict(sorted(Counter(map(x5.discard_name, late)).items()))
    if not late:
        failures.append(NOT_SHOWN)

    def old(r: dict) -> bool:
        return r.get("rid") in old_attempts and r["attempt"] <= old_attempts[r["rid"]]

    results = [r for r in of_type(after, "hop") if r.get("hop") == STAGE1_GATEWAY]
    accepted = sum(r.get("accepted") is True and old(r) for r in results)
    if accepted:
        failures.append(f"{accepted} late outputs of an old attempt were accepted")
    if stage == 0 and any(old(r) for r in results):
        failures.append("stage 1 ran a late step of the old stage 0 instead of refusing it")
    return data, failures


# ------------------------------------------------------------------ one repeat


def restore(
    ctx: Context, stage: int, old_epoch: int, mark: int, wait: bool, problems: list[str]
) -> None:
    """Put the stage back on its own supervisor. With a replacement registered, kill the
    old worker, which takes no more beats, so that its supervisor starts one that takes
    over, and only then stop the replacement. Without one, stop the standby; the resumed
    old process, if it was declared dead, leaves and is respawned by itself. With wait
    False (an interrupt), every request is sent but nothing is waited for."""

    def replacement() -> int | None:
        rows = ctx.tail.since(mark)
        found = replacement_index(rows, stage, old_epoch)
        return None if found is None else rows[found]["new_epoch"]

    def attempt(what: str, action) -> None:
        try:
            action()
        except ControlError as err:
            if err.status != 409:  # 409: no worker, so it is being respawned already
                problems.append(f"restore: {what}: {err}")
        except (CheckFailed, OSError) as err:
            problems.append(f"restore: {what}: {err}")

    def not_left_suspended() -> None:
        # a suspend that arrived late, after the resume
        if ctx.control.status(stage).get("state") == "suspended":
            ctx.control.fault(stage, "resume")

    attempt("check the old worker", not_left_suspended)
    new_epoch = replacement()
    if new_epoch is None:
        attempt("stop the standby", lambda: ctx.standby.post(stage, "stop"))
        new_epoch = replacement()  # it may have registered just before it was stopped
        if new_epoch is None:
            return
    attempt("kill the old worker", lambda: ctx.control.fault(stage, "kill"))

    def taken_over():
        epochs = [
            r.get("new_epoch")
            for r in of_type(ctx.tail.since(mark), "registered")
            if r.get("stage") == stage
        ]
        return True if epochs and epochs[-1] > new_epoch else None

    if wait:
        attempt(
            "the supervisor's new worker taking over",
            lambda: x5.wait_until(taken_over, "it to register", ctx.wait),
        )
    attempt("stop the standby", lambda: ctx.standby.post(stage, "stop"))


def restore_always(
    ctx: Context, stage: int, old_epoch: int, mark: int, wait: bool, problems: list[str]
) -> None:
    """restore, run to its end. After an interrupt nothing is waited for and further
    signals are ignored; an interrupt while it waits starts it again that way."""
    if wait:
        try:
            restore(ctx, stage, old_epoch, mark, True, problems)
            return
        except KeyboardInterrupt:
            with signals_ignored():
                restore(ctx, stage, old_epoch, mark, False, problems)
            raise
    with signals_ignored():
        restore(ctx, stage, old_epoch, mark, False, problems)
    for problem in problems:  # the repeat's line will not be written: say it now
        print(problem, file=sys.stderr, flush=True)


def inject(
    ctx: Context, stage: int, old: int, mark: int, jobs: list[Job], before: dict, seen: dict
) -> None:
    """Suspend, have the replacement serve, resume. What happened is put in seen as it
    happens, so that a failure part way keeps what is known."""
    failures = seen["failures"]
    with suspended(ctx.control, stage) as hold:
        seen["hold"] = hold
        if hold.pid != before["pid"]:
            failures.append(f"the supervisor suspended pid {hold.pid}, not pid {before['pid']}")

        def declared():
            rows = ctx.tail.since(mark)
            found = death_index(rows, stage, old)
            return None if found is None else (found, rows[found])

        died, death = x5.wait_until(declared, f"epoch {old} to be declared dead", ctx.wait)
        if death.get("path") != "heartbeat_timeout":
            raise CheckFailed(f"the death was detected by {death.get('path')}")
        rids = {job.rid for job in jobs}
        old_attempts = attempts_before(ctx.tail.since(mark)[:died], rids)
        ctx.standby.post(stage, "start")

        def replaced_and_serving():
            rows = ctx.tail.since(mark)
            found = replacement_index(rows, stage, old)
            if found is None or not served(rows[found:], old_attempts):
                return None
            return True

        x5.wait_until(replaced_and_serving, "the replacement to serve", ctx.wait)
        status = ctx.control.status(stage)
        if (status.get("state"), status.get("pid")) != ("suspended", hold.pid):
            raise CheckFailed(f"the old process is not there to resume: {status}")
        seen["free_ram_two_workers"] = status.get("free_ram_bytes")
        resume_mark = ctx.tail.mark()
        resumed = hold.resume()
        seen["resume_at"] = resume_mark - mark
        if resumed != hold.pid:
            failures.append(f"the supervisor resumed pid {resumed}, not pid {hold.pid}")

    def late():
        rows = ctx.tail.since(resume_mark)
        return True if late_outputs(rows, stage, old, old_attempts) else None

    try:
        x5.wait_until(late, "a late output of the resumed stage", ctx.late_wait)
    except CheckFailed:
        pass  # analyse_stale finds none in the window either, and fails the repeat
    status = ctx.control.status(stage)
    seen["old_alive_after_resume"] = status.get("pid") == hold.pid
    if not seen["old_alive_after_resume"]:
        failures.append(f"the old process did not stay after its resume: {status}")


def x6_repeat(
    ctx: Context, stage: int, repeat: int, refs: list[Job], epochs: dict[int, int]
) -> dict:
    jobs = [ref.fresh() for ref in refs]
    old = epochs[stage]
    seen = {
        "failures": [],
        "hold": None,
        "resume_at": None,
        "old_alive_after_resume": None,
        "free_ram_before": None,
        "free_ram_two_workers": None,
    }
    failures = seen["failures"]
    load, rows = None, []
    mark, bad = ctx.tail.mark(), ctx.tail.bad
    interrupted = True
    try:
        try:
            before = x5.require_running(stage, ctx.control.status(stage))
            idle = ctx.standby.status(stage)
            if idle.get("state") != "idle":
                raise CheckFailed(f"stage {stage}'s standby is not idle: {idle}")
            seen["free_ram_before"] = before.get("free_ram_bytes")
            load = x5.Load(ctx.call, jobs, ctx.new_tokens).start()
            load.wait_for_tokens(ctx.fault_at, ctx.timeout)
            inject(ctx, stage, old, mark, jobs, before, seen)
        except (CheckFailed, ControlError) as err:
            earlier = err.__context__
            if isinstance(err, ResumeFailed) and isinstance(earlier, CheckFailed):
                failures.append(str(earlier))
            failures.append(str(err))
        if load is not None:
            ctx.finish(load, failures)
            time.sleep(ctx.settle)
            rows = ctx.tail.since(mark)
        interrupted = False
    finally:
        # after the window is read, so what restoring does is not in it
        if seen["hold"] is not None or load is not None:
            restore_always(ctx, stage, old, mark, not interrupted, failures)
    if ctx.tail.bad != bad:
        failures.append(f"{ctx.tail.bad - bad} unreadable metrics lines")
    data, found = x5.analyse(
        rows,
        jobs,
        refs,
        stage=stage,
        old_epoch=old,
        fault_at=ctx.fault_at,
        new_tokens=ctx.new_tokens,
        single_host=ctx.single_host,
        death_path="heartbeat_timeout",
    )
    hold = seen["hold"]
    stale, more = analyse_stale(
        rows,
        stage=stage,
        old_epoch=old,
        rids={job.rid for job in jobs if job.rid},
        resume_at=seen["resume_at"],
        held_s=None if hold is None else hold.held_s,
    )
    data.update(
        stale,
        old_alive_after_resume=seen["old_alive_after_resume"],
        # the stage laptop's available memory, from its supervisor
        stage_free_ram_bytes={
            "before": seen["free_ram_before"],
            "two_workers": seen["free_ram_two_workers"],
        },
    )
    return x5.repeat_line("x6", stage, repeat, data, failures + found + more)


def only_not_shown(line: dict) -> bool:
    """True when the one thing wrong with a repeat is that no late output came."""
    return line["failures"] == [NOT_SHOWN]


# ------------------------------------------------------------------ command line


def main(argv: list[str] | None = None) -> int:
    parser = x5.parser_for("X6: stale outputs after each stage is suspended and resumed.")
    parser.add_argument(
        "--late-wait", type=float, default=LATE_WAIT_S, help="seconds for a late output"
    )
    parser.add_argument(
        "--extra-attempts",
        type=int,
        default=0,
        help="per stage: run again a repeat whose only failure is that no late output came",
    )
    args = parser.parse_args(argv)
    if args.late_wait <= 0 or args.extra_attempts < 0:
        parser.error("--late-wait must be positive and --extra-attempts not negative")
    hosts, controls, single, info = x5.check_args(parser, args)
    try:
        standbys = StandbyControl({s: standby.standby_endpoint(s) for s in args.stages})
        for stage in args.stages:
            status = standbys.status(stage)
            if status.get("state") != "idle":
                parser.error(f"stage {stage}'s standby is not idle: {status}")
    except (ValueError, OSError, ControlError) as err:
        parser.error(f"each faulted stage needs pllm.standby running on its laptop: {err}")
    prompt_ids = prompts.prompts(args.model, args.length)
    ctx = Context(
        x5.gateway_call(args.model, hosts.gateway.address, args.timeout),
        x5.MetricsTail(args.metrics),
        x5.Control(controls),
        single,
        requests=args.requests,
        new_tokens=args.new_tokens,
        fault_at=args.fault_at,
        timeout=args.timeout,
        wait=args.wait,
        settle=args.settle,
        standby=standbys,
        late_wait=args.late_wait,
    )
    args.record = {"late_wait_s": args.late_wait, "extra_attempts": args.extra_attempts}

    def run(emit):
        return x5.run_experiment(
            ctx,
            "x6",
            args.stages,
            args.repeats,
            prompt_ids,
            x6_repeat,
            emit,
            lambda lines: x5.aggregate("x6", lines, SPREAD),
            only_not_shown,
            args.extra_attempts,
        )

    return x5.record_and_run("x6", args, info, single, run)


if __name__ == "__main__":
    sys.exit(main())
