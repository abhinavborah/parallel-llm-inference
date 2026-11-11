"""The gateway: Frontend.Generate, the decode loop and the request table (R4.1 to R4.3, R4.6).

One gRPC server hosts four services:

- Control.Register: stages register here and get their epoch (pllm.control).
- Control.Heartbeat: each registered stage keeps one stream here (R3.10). The gateway keeps
  the latest Beat of each stage with the time it arrived on this process's monotonic
  clock, and the Nacks the stages report (R3.12). A Nack for the step a running request is
  waiting on (same attempt and step) fails that request at once with UNAVAILABLE, naming
  the stage and the reason; a Nack for another step, such as one a resumed old stage sent
  late, leaves it running. Failing at once replaces waiting for the deadline; replaying
  instead comes with the recovery stories (#38, #39). A Nack for a request that has ended
  needs nothing (a late step of a freed attempt). It sends back the Free messages queued
  for that stage, and a route update naming the other stage's address and epoch when the
  stream opens and each time that stage registers: Stage 0 uses it as its next hop
  (R3.13), Stage 1 as the sender epoch it accepts (R3.11). A stream whose first Beat is
  not from the stage's latest registration is refused with FAILED_PRECONDITION, and so is
  an open stream once its stage registers again. A stage whose epoch was declared dead,
  with no replacement yet, gets ABORTED when it is next heard from, by a Beat or a new
  stream; it exits and its supervisor starts a new one. ABORTED is not sent at the declaration: a
  suspended stage would read it on resume and exit even if a replacement had registered.
- Frontend.Generate: a client sends a prompt, max_tokens and a timeout and receives the
  tokens in order. The gateway tokenises the prompt, gives the request an id and attempt
  1, sends the whole prompt as one message at start_pos 0, then drives greedy decode:
  each result's token goes back to Stage 0 as the next step, one position further on.
  A request ends on an end-of-sequence token or after max_tokens tokens.
- Results.Push: the last stage pushes each step's result here; the gateway hands it to the
  request it belongs to.

The gateway keeps one stream open to the registered Stage 0 and reopens it when Stage 0
registers again with another address or epoch. Requests share that stream.

Deadlines are judged here only, on this process's monotonic clock (decision #32): no
clock is compared across machines. Each step sent to Stage 0 carries budget_ms, the time
left on the request's deadline, and Stage 0 passes on what is left of it to Stage 1 (R5.4);
a stage drops a step whose budget ran out before it ran (pllm.stage_service). A waiting
request checks its deadline at least every POLL_SECONDS and fails with DEADLINE_EXCEEDED
once it has passed, whatever a stage does, so no request hangs (R5.6). The tolerance is that
interval plus thread scheduling and the error's way to the client; the tests assert 0.5 s on
loopback.

When a request ends, for any reason, it leaves the table and a Free for it is queued for
each stage (R4.6); that stage's heartbeat stream delivers it. A Free queued while a stage
has no stream waits for one, and the queue is emptied when the stage registers again,
because a new process holds none of those caches. A request that ends early (deadline,
cancel, failure) can still have a step on the data stream; the stage remembers the freed
attempt and refuses that step (pllm.kvstore), so it cannot create the cache again.

Every request and every message the gateway sends or receives is written to the run's
metrics file, stamped by one clock (pllm.metrics). Deadlines stay on the monotonic clock,
which is never reported.

Failure detection is pllm.detection. A heartbeat stream that breaks, a stage silent past the
heartbeat timeout and a step past the step timeout are each recorded on the recorder's
clock, in the log and as a line of the metrics file. Both timeouts are gateway flags.
The first two declare the stage's epoch dead: requests are refused with UNAVAILABLE until a
new registration gives the stage a higher epoch, and the neighbour then gets a route update.
Only the heartbeat stream names a dead stage: Stage 0 also ends its data stream when Stage 1
dies or a route moves. Requests in flight behave as before; replaying them is #39.

Many requests in flight (R4.5): every request goes to Stage 0 on the one stream, so the
stages see the requests' steps interleaved in whatever order they arrive, and Stage 0 works
on one request while Stage 1 works on another. Each request sends its next step only after
its own result, so its steps stay strictly in order; nothing orders one request against
another. A request that ends early (cancel, deadline, a refused step) ends only itself. A
stage ends the shared stream when a message is malformed, when a step fails inside the stage
(including running out of memory), or when its next hop fails or a route update moves it;
every request on the stream then fails with UNAVAILABLE, and recovering them is the recovery
stories. A request whose prompt and max_tokens exceed the model's context length is refused
at admission, so no valid request grows a cache past the model's context. That bounds each
request, not Host B's memory: with an even split, one request at the cap holds about 180 MiB
of Stage 0 cache for SmolLM2 (8192 positions) but about 4.4 GiB for Qwen3-0.6B (40960
positions), more than Host B has. Admission by Host B's KV budget (R5.3) is #34. Equality under
concurrency rests on each stage handling its stream one step at a time: requests never share
a model call within a stage. One long prefill at a stage delays every other request's next
step there (head-of-line blocking). Each Generate holds a
server thread for its life, so serve() caps running requests at max_workers -
RESERVED_THREADS; without the cap, a full pool leaves no thread for Results.Push and every
running request stalls until its deadline. Also not here: replay (#39).

Flow control (R5.1, R5.2). Admission: a request that finds every running slot taken waits
in a first-in, first-out queue of max_queued places; one that finds the queue full is
refused at once with RESOURCE_EXHAUSTED. A slot that frees goes to the head of the queue.
Time in the queue counts against the request's deadline, and a request whose deadline
passes there fails with DEADLINE_EXCEEDED; one whose client goes away leaves the queue. A
waiting request holds a server thread too, so serve() adds max_queued threads to the pool.
Credits: a request takes one of `credits` before it sends each step and gives it back when
the step's result has come (or the request ends). Requests waiting for a credit get one in
arrival order, and a request giving its credit back waits behind them for its next step.
Each request has at most one step in the pipeline, and that step is at one stage at a
time, so at most `credits` steps are at any stage. A request that is waiting for a credit
when its deadline passes fails with DEADLINE_EXCEEDED. queue_peak and in_flight_peak record
the deepest the queue has been and the most steps in flight at once. A step whose request
failed at its deadline can still be at a stage after its credit came back; the stage's
bounded inbox (pllm.stage_service) is the hard bound there.

Result filter and exactly-once delivery (R4.7, R4.8): a result is accepted only when its
request is running and its attempt, its step and its epoch all match the request's attempt,
the step the request waits on and the live epoch of the last stage. Anything else is dropped
and counted by reason in discards:

- unknown_request: no running request has that id (a late result after the request ended).
- stale_attempt: another attempt of the request.
- stale_epoch: sent by a last stage that is not the registered one (an old incarnation).
- duplicate: a step the request has already received (the token went to the client once).
- ahead: a step the request has not asked for yet.
- stale_nack: a Nack that names no step a running request waits on (the request ended, or
  another attempt or step), counted when it arrives or when a queued one is taken.

Only an accepted result moves the request on and reaches the client, so each (request,
step) is streamed once and in order. A result is checked when it arrives and again when the
request takes it, since a result can be queued just before the step it was for completes or
before a new last stage registers. A replay (#39) must compare the results of steps below
len(tokens) with the stored history and not stream them again; the filter itself stays as
it is, since each replayed step is asked for in turn under the new attempt.
"""

import argparse
import collections
import json
import logging
import queue
import signal
import sys
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import grpc

from pllm import config, metrics, prompts, runrecord, wire
from pllm.control import Registration, Registry, expected_model
from pllm.detection import HEARTBEAT_TIMEOUT_SECONDS, STEP_TIMEOUT_SECONDS, Detector
from pllm.heartbeat import HEARTBEAT_SECONDS
from pllm.model import ModelConfig
from pllm.models import MODELS_DIR, PINNED
from pllm.proto import protocol_pb2, protocol_pb2_grpc
from pllm.stage_service import _DONE, _drain, budget_ms

log = logging.getLogger("pllm.gateway")

POLL_SECONDS = 0.05  # how often a waiting request checks its deadline, client and link
CLOSE_WAIT_SECONDS = 5.0  # how long close() waits for Stage 0's byte counters, in all
MAX_NACKS = 1024  # Nacks kept for the recovery stories; the count covers all of them
MAX_UNSENT = 4  # a UTF-8 character spans at most four byte-level tokens
REPLACEMENT = chr(0xFFFD)  # what an incomplete UTF-8 character decodes to
STAGES = (0, 1)
# Server threads kept free of Generate: the Results.Push stream from Stage 1, which holds
# its thread for its life; one for Control.Register and for refusing requests; and one
# heartbeat stream per stage plus one replaced heartbeat per stage still closing.
RESERVED_THREADS = 2 + 2 * len(STAGES)
MAX_RUNNING = 14  # Generate calls at once with the default pool
MAX_QUEUED = 8  # requests waiting for a running slot; [limits] admission_queue sets it


@dataclass
class Request:
    """One row of the request table. tokens is the authoritative history of the request."""

    rid: str
    attempt: int
    deadline: float  # time.monotonic() value
    prompt_len: int
    tokens: list[int] = field(default_factory=list)
    expected_step: int = 0
    results: queue.Queue = field(default_factory=queue.Queue)
    max_tokens: int = 0
    arrived_ns: int | None = None  # the three *_ns values are on the recorder's clock
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    step_due_ns: int | None = None  # detector clock; None once the step's record is written

    def row(self) -> dict:
        return {
            "rid": self.rid,
            "attempt": self.attempt,
            "tokens": list(self.tokens),
            "expected_step": self.expected_step,
            "deadline_s": round(self.deadline - time.monotonic(), 3),
        }


class Link:
    """The gateway's client stream to one Stage 0 incarnation."""

    def __init__(self, addr: str, epoch: int, options):
        self.addr, self.epoch = addr, epoch
        self.channel = grpc.insecure_channel(addr, options=options)
        self.outbox: queue.Queue = queue.Queue()
        self.ended = False
        self.call = protocol_pb2_grpc.StageStub(self.channel).Steps.future(_drain(self.outbox))

    def failure(self) -> str | None:
        """Why the stream ended, or None while it is open. An ended link fails its requests
        at once, even if its stage 0 is hung and never finishes the stream."""
        if not self.call.done():
            return "replaced or closed by the gateway" if self.ended else None
        try:
            self.call.result()
        except grpc.RpcError as err:
            return f"{err.code().name}: {err.details()}"
        except grpc.FutureCancelledError:
            return "cancelled"
        return "closed by stage 0"

    def end(self) -> None:
        """Half-close the stream. Stage 0 then finishes and sends its byte counters; closing
        the channel now would cancel the stream and lose them."""
        self.ended = True
        self.outbox.put(_DONE)

    def report(self) -> list[list] | None:
        """Stage 0's byte counters for this stream, or None if it gave no valid report."""
        return metrics.parse_counters(self.call.trailing_metadata()) if self.call.done() else None

    def wait(self, seconds: float) -> None:
        try:
            self.call.exception(timeout=max(0.0, seconds))
        except (grpc.FutureTimeoutError, grpc.FutureCancelledError):
            pass


class Session:
    """One open heartbeat stream from one stage incarnation."""

    def __init__(self, stage: int, epoch: int):
        self.stage, self.epoch = stage, epoch
        self.outbox: queue.Queue = queue.Queue()  # route updates for this stream
        self.superseded = threading.Event()
        self.fenced = threading.Event()  # a Beat arrived after this epoch was declared dead
        self.closed = threading.Event()


class Gateway(
    protocol_pb2_grpc.ControlServicer,
    protocol_pb2_grpc.FrontendServicer,
    protocol_pb2_grpc.ResultsServicer,
):
    """tokenizer needs encode(text) -> ids and decode(ids) -> text; eos_ids end a request."""

    recorder = metrics.Recorder()  # writes nothing; main sets a real one, like max_running

    def __init__(
        self,
        registry: Registry,
        tokenizer,
        eos_ids,
        hidden_size: int,
        max_bytes: int,
        max_positions: int | None = None,
        max_queued: int = MAX_QUEUED,
        credits: int | None = None,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT_SECONDS,
        step_timeout: float = STEP_TIMEOUT_SECONDS,
    ):
        """max_positions is the model's context length: a request whose prompt and
        max_tokens together exceed it is refused. None leaves it unchecked. max_queued is
        the admission queue's size and credits the steps in flight at once (None: no
        limit beyond the running slots). The two timeouts, in seconds, go to the
        failure detector."""
        self.registry = registry
        self.max_positions = max_positions
        self.tokenizer = tokenizer
        self.eos_ids = frozenset(eos_ids)
        self.hidden_size = hidden_size
        self.max_bytes = max_bytes
        self.options = wire.channel_options(hidden_size)
        self.requests: dict[str, Request] = {}
        self.frees = {stage: queue.Queue() for stage in STAGES}
        self.dropped = 0  # results and Nacks dropped, all reasons together
        self.discards: collections.Counter = collections.Counter()  # dropped, by reason
        self.nacks: collections.deque = collections.deque(maxlen=MAX_NACKS)
        self.nack_count = 0
        self._beats: dict[int, tuple[protocol_pb2.Beat, float]] = {}
        self._sessions: dict[int, Session] = {}
        self.max_running: int | None = None  # Generate calls at once; serve() sets it
        self._running = 0
        self.max_queued = max_queued
        self._waiting: collections.deque = collections.deque()  # one Event per waiting request
        self.queue_peak = 0
        self.credits = credits
        self._in_flight = 0  # steps sent whose result has not come
        self.in_flight_peak = 0
        self._credit_waiting: collections.deque = collections.deque()  # one Event per waiter
        self._lock = threading.Lock()
        # Register and opening a heartbeat stream run one at a time, so a stream cannot
        # pass its epoch check just before a newer registration and then become current.
        self._control = threading.Lock()
        self._link: Link | None = None
        # Looked up at each call: main sets the recorder after this. A new recorder keeps
        # the same clock, or the detector would subtract times from two clocks.
        self.detector = Detector(
            heartbeat_timeout,
            step_timeout,
            clock=lambda: self.recorder.now(),
            sink=lambda record: self.recorder.event(record),
        )
        self._stopping = threading.Event()
        self._watcher: threading.Thread | None = None
        self._retired: list[Link] = []  # half-closed, waiting for stage 0's counters

    # ------------------------------------------------------------------ table and frees

    def table(self) -> list[dict]:
        """A snapshot of the request table: id, attempt, tokens so far, expected step and
        seconds left until the deadline."""
        with self._lock:
            return [request.row() for request in self.requests.values()]

    def frees_for(self, stage: int) -> list[protocol_pb2.Free]:
        """Take the frees waiting for a stage."""
        out = []
        while True:
            try:
                out.append(self.frees[stage].get_nowait())
            except queue.Empty:
                return out

    def _end(self, request: Request, how: str, code=None) -> None:
        with self._lock:
            self.requests.pop(request.rid, None)
        for stage in STAGES:
            self.frees[stage].put(protocol_pb2.Free(rid=request.rid, attempt=request.attempt))
        self.recorder.request(
            how,
            code,
            rid=request.rid,
            attempt=request.attempt,
            prompt_tokens=request.prompt_len,
            max_tokens=request.max_tokens,
            output_tokens=len(request.tokens),
            arrived_ns=request.arrived_ns,
            first_token_ns=request.first_token_ns,
            last_token_ns=request.last_token_ns,
        )
        log.info("request %s ended (%s) after %d tokens", request.rid, how, len(request.tokens))

    def _rejected(
        self, req, context, arrived: int, prompt_tokens: int | None, how: str = "rejected"
    ) -> None:
        """The line of a request that ended before it entered the table."""
        self.recorder.request(
            how,
            context.code(),
            rid=None,
            attempt=None,
            prompt_tokens=prompt_tokens,
            max_tokens=req.max_tokens,
            output_tokens=0,
            arrived_ns=arrived,
            first_token_ns=None,
            last_token_ns=None,
        )

    # ------------------------------------------------------------------ Control

    def Register(self, request, context):
        with self._control:
            previous = self.registry.stages.get(request.stage)
            try:
                entry = self.registry.register(request)
            except Registration as err:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(err))
            neighbour = 1 - entry.stage
            with self._lock:
                old = self._sessions.get(entry.stage)
                other = self._sessions.get(neighbour)
            if old is not None:
                old.superseded.set()  # before the drain: it stops taking frees
            self.frees_for(entry.stage)  # a new process holds none of those caches
            if previous is not None:  # a respawn can beat detection
                self.detector.declare(previous.stage, previous.epoch, "replaced")
            queued = None
            if other is not None:  # Stage 0 learns its next hop, Stage 1 its upstream
                other.outbox.put(self._route(entry))
                queued = self.detector.now()
            old_epoch = previous.epoch if previous is not None else None
            self.detector.registered(entry.stage, old_epoch, entry.epoch, queued)
        return protocol_pb2.RegisterResp(epoch=entry.epoch)

    @staticmethod
    def _route(entry) -> protocol_pb2.ControlMsg:
        """A route update naming entry's stage. The field is called next_stage, but Stage 1
        receives one naming stage 0, its upstream, and takes the epoch as the sender epoch
        it accepts (R3.11); the schema is unchanged."""
        return protocol_pb2.ControlMsg(
            route_update=protocol_pb2.RouteUpdate(
                next_stage=entry.stage, addr=entry.addr, epoch=entry.epoch
            )
        )

    def beats(self) -> dict[int, dict]:
        """The latest Beat of each stage and how many seconds ago it arrived."""
        now = time.monotonic()
        with self._lock:
            return {
                stage: {
                    "epoch": b.epoch,
                    "inbox_depth": b.inbox_depth,
                    "cached_requests": b.cached_requests,
                    "age_s": round(now - at, 3),
                }
                for stage, (b, at) in self._beats.items()
            }

    def _record(self, session: Session, message: protocol_pb2.StageMsg) -> None:
        kind = message.WhichOneof("msg")
        with self._lock:
            if self._sessions.get(session.stage) is not session:
                return  # a stream that has been replaced
            if kind == "beat":
                if self.detector.beat(session.stage, session.epoch):
                    self._beats[session.stage] = (message.beat, time.monotonic())
                else:
                    session.fenced.set()
                return
            if kind != "nack":
                return
            self.nacks.append(message.nack)
            self.nack_count += 1
            n = message.nack
            request = self.requests.get(n.rid)
            running = (
                request is not None
                and request.attempt == n.attempt
                and request.expected_step == n.step
            )
            if not running:  # a request that ended, another attempt or another step
                self._discard("stale_nack")
        log.warning(
            "stage %d nacked %s attempt %d step %d: %s", n.stage, n.rid, n.attempt, n.step, n.reason
        )
        if running:  # its step was refused, so it cannot go on: fail it at once (R3.12)
            request.results.put(n)

    def Heartbeat(self, request_iterator, context):
        try:
            first = next(request_iterator)
        except (StopIteration, grpc.RpcError):
            return
        if first.WhichOneof("msg") != "beat":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "the first message must be a Beat")
        stage, epoch = first.beat.stage, first.beat.epoch
        with self._control:
            entry = self.registry.stages.get(stage)
            if entry is None or entry.epoch != epoch:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"stage {stage} epoch {epoch} is not its latest registration",
                )
            session = Session(stage, epoch)
            with self._lock:
                old = self._sessions.get(stage)
                self._sessions[stage] = session
            if old is not None:
                old.superseded.set()
            neighbour = self.registry.stages.get(1 - stage)
            route = self._route(neighbour) if neighbour is not None else None
        self._record(session, first)

        def read():
            why = "closed"
            try:
                for message in request_iterator:
                    self._record(session, message)
            except grpc.RpcError:  # the stage went away
                why = "broken"
            finally:
                session.closed.set()
                self._stream_ended(session, why)

        threading.Thread(target=read, name=f"heartbeat-{stage}", daemon=True).start()
        log.info("heartbeat open from stage %d epoch %d", stage, epoch)
        try:
            if route is not None:
                yield route
            while not session.closed.is_set() and context.is_active():
                refusal = self._refusal(session)
                if refusal is not None:
                    context.abort(*refusal)
                try:
                    yield session.outbox.get_nowait()
                    continue
                except queue.Empty:
                    pass
                try:
                    free = self.frees[stage].get(timeout=POLL_SECONDS)
                except queue.Empty:
                    continue
                if session.superseded.is_set():  # taken just as a newer stage registered
                    self.frees[stage].put(free)
                    continue
                yield protocol_pb2.ControlMsg(free=free)
        finally:
            with self._lock:
                if self._sessions.get(stage) is session:
                    del self._sessions[stage]
            log.info("heartbeat closed from stage %d epoch %d", stage, epoch)

    def _refusal(self, session: Session) -> tuple[grpc.StatusCode, str] | None:
        """Why an open stream must end, or None. Superseded comes first, under the lock
        Register holds: a stage told ABORTED exits, and its respawn would register a third
        epoch over the replacement."""
        with self._control:
            if session.superseded.is_set():
                return (
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"stage {session.stage} registered again after epoch {session.epoch}",
                )
            if session.fenced.is_set():
                return (
                    grpc.StatusCode.ABORTED,
                    f"stage {session.stage} epoch {session.epoch} was declared dead",
                )
        return None

    def _stream_ended(self, session: Session, why: str) -> None:
        """Declare the stream's epoch dead unless a newer stream or registration replaced it.
        Not judged by _sessions: on a kill the generator can remove the session first."""
        entry = self.registry.stages.get(session.stage)
        current = entry is not None and entry.epoch == session.epoch
        if current and not session.superseded.is_set():
            self.detector.declare(session.stage, session.epoch, "stream_break", why)

    def watch(self) -> None:
        """Start the thread that expires silent stages; close() stops it. It is not a pool
        thread, so RESERVED_THREADS does not count it."""

        def run():
            while not self._stopping.wait(POLL_SECONDS):
                self.detector.expire()

        self._watcher = threading.Thread(target=run, name="failure-detector", daemon=True)
        self._watcher.start()

    # ------------------------------------------------------------------ Stage 0 link

    def _stage0(self, context) -> Link:
        stages = self.registry.stages
        for stage in STAGES:
            if stage not in stages:
                context.abort(grpc.StatusCode.UNAVAILABLE, f"stage {stage} is not registered")
            if self.detector.is_dead(stages[stage].epoch):
                context.abort(
                    grpc.StatusCode.UNAVAILABLE,
                    f"stage {stage} epoch {stages[stage].epoch} is dead; waiting for it to "
                    "register again",
                )
        entry = stages[0]
        with self._lock:
            link = self._link
            moved = link is not None and (link.addr, link.epoch) != (entry.addr, entry.epoch)
            if link is None or link.failure() or moved:
                if link is not None:
                    link.end()
                    self._retired.append(link)
                link = self._link = Link(entry.addr, entry.epoch, self.options)
            done = {old: old.call.done() for old in self._retired}
            self._retired = [old for old, ended in done.items() if not ended]
        self._collect([old for old, ended in done.items() if ended])  # never waits

        return link

    def _collect(self, links: list[Link]) -> None:
        for link in links:
            self.recorder.link_ended(link.report())
            link.channel.close()

    def close(self) -> None:
        """End every stream to Stage 0 and add up its byte counters, waiting at most
        CLOSE_WAIT_SECONDS in all. Results.Push must still be up, so that Stage 1 can finish
        and Stage 0 can end cleanly. The gateway stays usable, without its heartbeat watch,
        and calling this again is harmless."""
        self._stopping.set()
        if self._watcher is not None:
            self._watcher.join()
        with self._lock:
            if self._link is not None:
                self._link.end()
                self._retired.append(self._link)
                self._link = None
            links, self._retired = self._retired, []
        until = time.monotonic() + CLOSE_WAIT_SECONDS
        for link in links:
            link.wait(until - time.monotonic())
        self._collect(links)

    # ------------------------------------------------------------------ Results.Push

    def _why_not(self, result: protocol_pb2.StepResult, request: Request | None) -> str | None:
        """Why result cannot be accepted for request, or None if it can (R4.7)."""
        if request is None:
            return "unknown_request"
        if result.attempt != request.attempt:
            return "stale_attempt"
        last = self.registry.stages.get(STAGES[-1])
        if last is None or result.epoch != last.epoch:
            return "stale_epoch"
        if result.step < request.expected_step:
            return "duplicate"
        if result.step > request.expected_step:
            return "ahead"
        return None

    def _discard(self, reason: str) -> None:
        """Count a dropped result or Nack. Call with the lock held."""
        self.dropped += 1
        self.discards[reason] += 1

    def Push(self, request_iterator, context):
        for result in request_iterator:
            with self._lock:
                request = self.requests.get(result.rid)
                reason = self._why_not(result, request)
                if reason is not None:
                    self._discard(reason)
            self.recorder.hop(metrics.STAGE1_GATEWAY, result, accepted=reason is None)
            if reason is None:
                request.results.put(result)
            else:
                log.warning(
                    "dropped result %s attempt %d step %d epoch %d: %s",
                    result.rid,
                    result.attempt,
                    result.step,
                    result.epoch,
                    reason,
                )
        return protocol_pb2.Empty()

    # ------------------------------------------------------------------ Frontend.Generate

    def _admit(self, req: protocol_pb2.GenerateReq, context) -> list[int]:
        if req.max_tokens < 1:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "max_tokens must be at least 1")
        if req.timeout_ms < 1:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "timeout_ms must be at least 1")
        ids = self.tokenizer.encode(req.prompt)
        if not ids:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "the prompt has no tokens")
        # Positions past the context length do not raise at a stage (rotary positions are
        # computed as needed), but each one grows the request's cache on both stages, so an
        # unbounded max_tokens could run a stage out of memory and end the shared stream.
        total = len(ids) + req.max_tokens
        if self.max_positions is not None and total > self.max_positions:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"the prompt has {len(ids)} tokens and max_tokens is {req.max_tokens}; "
                f"together {total}, above the model's context length of {self.max_positions}",
            )
        # The prefill is the largest message: Stage 0 sends len(ids) x hidden fp32 values
        # to Stage 1, and every receiver accepts at most max_bytes (decision #20).
        needed = len(ids) * self.hidden_size * 4 + config.FRAMING_MARGIN
        if needed > self.max_bytes:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"the prompt has {len(ids)} tokens; its prefill needs {needed} bytes, above "
                f"the receive limit of {self.max_bytes} bytes set by [limits] in the hosts file",
            )
        return ids

    def Generate(self, req, context):
        arrived = self.recorder.now()
        ids = None
        try:
            ids = self._admit(req, context)
            deadline = time.monotonic() + req.timeout_ms / 1000  # time in the queue counts
            entered = self._enter(deadline, context)
        except Exception:
            self._rejected(req, context, arrived, len(ids) if ids else None)
            raise
        if not entered:  # the client went away while the request was queued
            self._rejected(req, context, arrived, len(ids), "cancelled by the client")
            return
        try:
            yield from self._generate(req, ids, deadline, context, arrived)
        finally:
            self._leave()

    # ------------------------------------------------------------------ flow control

    def _enter(self, deadline: float, context) -> bool:
        """Take a running slot, waiting in the admission queue if every slot is taken
        (R5.1). Refuses at once when the queue is full; False if the client went away."""
        with self._lock:
            if self.max_running is None or (self._running < self.max_running and not self._waiting):
                self._running += 1
                return True
            full = len(self._waiting) >= self.max_queued
            if not full:
                ticket = threading.Event()  # set when a leaving request hands over its slot
                self._waiting.append(ticket)
                self.queue_peak = max(self.queue_peak, len(self._waiting))
        if full:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"the gateway is running {self.max_running} requests and {self.max_queued} "
                "more are waiting, its limits; try again later",
            )
        while not ticket.wait(POLL_SECONDS):
            expired = time.monotonic() >= deadline
            gone = not context.is_active()
            if expired or gone:
                with self._lock:
                    handed = ticket.is_set()  # the slot came just now: pass it on
                    if not handed:
                        self._waiting.remove(ticket)
                if handed:
                    self._leave()
                if gone:
                    return False
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "the request passed its deadline waiting for a running slot",
                )
        return True

    def _leave(self) -> None:
        """Give up a running slot: to the head of the queue if a request waits."""
        with self._lock:
            if self._waiting:
                self._waiting.popleft().set()  # the slot passes on; _running is unchanged
            else:
                self._running -= 1

    def _take_credit(self, request: Request, context) -> bool:
        """Take a credit to send the request's next step (R5.2), waiting first in, first out
        behind the requests already waiting. False if the client went away; a deadline
        that passes while waiting fails the request."""
        with self._lock:
            if self.credits is None or (
                self._in_flight < self.credits and not self._credit_waiting
            ):
                self._in_flight += 1
                self.in_flight_peak = max(self.in_flight_peak, self._in_flight)
                return True
            ticket = threading.Event()  # set when a returned credit is handed to this request
            self._credit_waiting.append(ticket)
        while not ticket.wait(POLL_SECONDS):
            expired = time.monotonic() >= request.deadline
            gone = not context.is_active()
            if expired or gone:
                with self._lock:
                    handed = ticket.is_set()  # the credit came just now: pass it on
                    if not handed:
                        self._credit_waiting.remove(ticket)
                if handed:
                    self._give_credit()
                if gone:
                    return False
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    f"request {request.rid} passed its deadline waiting for a credit "
                    f"at step {request.expected_step}",
                )
        return True

    def _give_credit(self) -> None:
        """Give a credit back: to the first waiting request if one waits."""
        with self._lock:
            if self._credit_waiting:
                self._credit_waiting.popleft().set()  # passes on; _in_flight is unchanged
            else:
                self._in_flight -= 1

    def _generate(self, req, ids: list[int], deadline: float, context, arrived: int):
        try:
            link = self._stage0(context)
        except Exception:
            self._rejected(req, context, arrived, len(ids))
            raise
        request = Request(
            rid=uuid.uuid4().hex[:12],
            attempt=1,
            deadline=deadline,
            prompt_len=len(ids),
            max_tokens=req.max_tokens,
            arrived_ns=arrived,
        )
        with self._lock:
            self.requests[request.rid] = request
        how = "failed"
        try:
            send, start = ids, 0
            unsent: list[int] = []  # tokens whose text is not complete yet
            while True:
                step = request.expected_step
                if not self._take_credit(request, context):
                    how = "cancelled by the client"
                    return
                request.step_due_ns = self.detector.now() + self.detector.step_timeout_ns
                message = protocol_pb2.StepRequest(
                    rid=request.rid,
                    attempt=request.attempt,
                    step=step,
                    start_pos=start,
                    length=len(send),
                    target_epoch=link.epoch,
                    sender_epoch=0,
                    tokens=protocol_pb2.TokenIds(ids=send),
                    budget_ms=budget_ms(request.deadline, time.monotonic()),
                )
                # Before the put, so that a fast result cannot appear to precede its send.
                self.recorder.hop(metrics.GATEWAY_STAGE0, message)
                link.outbox.put(message)
                try:
                    result = self._wait(request, link, context)
                finally:
                    self._give_credit()  # the result came, or the request is ending
                if result is None:
                    how = "cancelled by the client"
                    return
                chosen = result.token
                with self._lock:
                    request.tokens.append(chosen)
                    request.expected_step += 1
                if chosen in self.eos_ids:
                    how = "end of sequence"
                elif len(request.tokens) >= req.max_tokens:
                    how = "max_tokens"
                unsent.append(chosen)
                text = self._text(unsent, final=how != "failed")
                if text is not None:
                    unsent = []
                request.last_token_ns = self.recorder.now()
                if request.first_token_ns is None:
                    request.first_token_ns = request.last_token_ns
                yield protocol_pb2.Token(
                    rid=request.rid,
                    step=step,
                    token=chosen,
                    text=text or "",
                    top_logit=result.top_logit,
                )
                if how != "failed":
                    return
                start += len(send)
                send = [chosen]
        except GeneratorExit:  # the client left while a token was on its way to it
            if how == "failed":
                how = "cancelled by the client"
            raise
        finally:
            self._end(request, how, context.code())

    def _text(self, ids: list[int], final: bool) -> str | None:
        """The text of tokens not yet sent, or None to wait for more: a byte-level token can
        end inside a character, which decodes as U+FFFD until the next token completes it."""
        text = self.tokenizer.decode(ids)
        if text.endswith(REPLACEMENT) and not final and len(ids) < MAX_UNSENT:
            return None
        return text

    def _wait(self, request: Request, link: Link, context):
        """The result for the request's expected step, or None if the client went away."""
        while True:
            left = request.deadline - time.monotonic()
            if left <= 0:
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    f"request {request.rid} passed its deadline at step {request.expected_step}",
                )
            if not context.is_active():
                return None
            if request.step_due_ns is not None and self.detector.now() >= request.step_due_ns:
                request.step_due_ns = None  # one record per step
                self.detector.step_deadline(request.rid, request.expected_step)
            try:
                result = request.results.get(timeout=min(left, POLL_SECONDS))
            except queue.Empty:
                pass
            else:
                # Checked again here, for a result and a Nack alike: either can be queued
                # just before the step moved on, and is then for a step already done.
                if isinstance(result, protocol_pb2.Nack):
                    current = (result.attempt, result.step) == (
                        request.attempt,
                        request.expected_step,
                    )
                    if current:
                        context.abort(
                            grpc.StatusCode.UNAVAILABLE,
                            f"stage {result.stage} refused step {result.step} of request "
                            f"{request.rid}: {result.reason}",
                        )
                    reason = "stale_nack"
                else:
                    with self._lock:
                        reason = self._why_not(result, request)
                    if reason is None:
                        return result
                with self._lock:
                    self._discard(reason)
            why = link.failure()
            if why is not None:
                context.abort(grpc.StatusCode.UNAVAILABLE, f"stage 0 stream failed: {why}")


def serve(
    gateway: Gateway, bind: str, max_workers: int = MAX_RUNNING + RESERVED_THREADS
) -> tuple[grpc.Server, int]:
    """Serve Control, Frontend and Results on bind (port 0 picks a free one).

    Every Generate, heartbeat stream and Results.Push stream holds a thread from one
    pool. At most max_workers - RESERVED_THREADS requests run at once and gateway.max_queued
    more wait for a slot, each on a thread of its own; the pool has max_workers +
    max_queued threads, so the other streams always find one. Any more are refused.
    """
    if max_workers <= RESERVED_THREADS:
        raise ValueError(f"max_workers must be above {RESERVED_THREADS}")
    gateway.max_running = max_workers - RESERVED_THREADS
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers + gateway.max_queued),
        options=gateway.options,
    )
    protocol_pb2_grpc.add_ControlServicer_to_server(gateway, server)
    protocol_pb2_grpc.add_FrontendServicer_to_server(gateway, server)
    protocol_pb2_grpc.add_ResultsServicer_to_server(gateway, server)
    port = server.add_insecure_port(bind)
    if port == 0:
        raise OSError(f"could not listen on {bind}")
    server.start()
    gateway.watch()
    log.info("gateway listening on port %d", port)
    return server, port


class Tokenizer:
    """The pinned tokenizer, with raw text in and out (no special tokens)."""

    def __init__(self, model: str):
        self.inner = prompts.load_tokenizer(model)

    def encode(self, text: str) -> list[int]:
        return prompts.encode(self.inner, text)

    def decode(self, ids: list[int]) -> str:
        return self.inner.decode(ids, skip_special_tokens=False)


def max_positions(model_dir) -> int | None:
    """The model's context length from config.json, or None if it does not give one."""
    value = json.loads((model_dir / "config.json").read_text()).get("max_position_embeddings")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def eos_ids(model_dir) -> list[int]:
    eos = json.loads((model_dir / "config.json").read_text()).get("eos_token_id")
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else list(eos)


def stop_event() -> threading.Event:
    """An event set by a stop signal. SIGBREAK exists on Windows only, and is what a launcher
    sends there; with no handler for it the process would die with a failure status."""
    done = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: done.set())
    return done


def run_until_stopped(gateway: Gateway, server: grpc.Server, done: threading.Event) -> None:
    """Serve until done is set, then end the run: first the streams to Stage 0, while
    Results.Push is still up, then the server. The second close() takes a stream opened by a
    request that arrived during the first. The last lines of the file come after every
    request has written its own."""
    while not done.wait(0.5):  # a timeout, so a signal handler can run while this waits
        pass
    gateway.close()
    server.stop(grace=1).wait()
    gateway.close()
    gateway.recorder.finish()


def seconds_above(low: float):
    """An argparse type: a finite number of seconds greater than low."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text}") from None
        if not low < value < float("inf"):
            raise argparse.ArgumentTypeError(f"must be more than {low} seconds")
        return value

    return parse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the gateway for one model.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--split-point", type=int, help="override the committed split point")
    parser.add_argument("--metrics", type=Path, help="metrics file (default: a new one in runs/)")
    parser.add_argument(
        "--heartbeat-timeout",
        type=seconds_above(HEARTBEAT_SECONDS),
        default=HEARTBEAT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"no Beat for this long declares a stage dead (default {HEARTBEAT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--step-timeout",
        type=seconds_above(0),
        default=STEP_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"record a step with no result after this long (default {STEP_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args(argv)
    model_dir = MODELS_DIR / args.model
    try:
        hosts = config.load()
        expected = expected_model(args.model, args.split_point)
        hidden = ModelConfig.from_file(model_dir / "config.json").hidden_size
        tokenizer = Tokenizer(args.model)
        limits = config.flow_limits()
        started = {"started": datetime.now(UTC).isoformat(), "script": "gateway"}
        path = args.metrics or runrecord.output_path(started, ".metrics.jsonl")
        recorder = metrics.Recorder(
            path,
            {
                "commit": runrecord.commit_hash(),
                "model": args.model,
                "revision": PINNED[args.model].revision,
                "split": expected.split_point,
                "network": hosts.network,
                **runrecord.machine(),
                "heartbeat_timeout_s": args.heartbeat_timeout,
                "step_timeout_s": args.step_timeout,
                "argv": sys.argv,
            },
        )
    except (ValueError, OSError) as err:
        parser.error(str(err))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    gateway = Gateway(
        Registry(args.model, expected),
        tokenizer,
        eos_ids(model_dir),
        hidden,
        config.max_message_bytes(hidden),
        max_positions(model_dir),
        max_queued=limits.admission_queue,
        credits=limits.credits,
        heartbeat_timeout=args.heartbeat_timeout,
        step_timeout=args.step_timeout,
    )
    gateway.recorder = recorder
    done = stop_event()
    server, _ = serve(gateway, f"0.0.0.0:{hosts.gateway.port}")
    run_until_stopped(gateway, server, done)


if __name__ == "__main__":
    main()
