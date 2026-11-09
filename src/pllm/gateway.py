"""The gateway: Frontend.Generate, the decode loop and the request table (R4.1 to R4.3, R4.6).

One gRPC server hosts three services:

- Control.Register: stages register here and get their epoch (pllm.control).
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
clock is compared across machines. The remaining budget is not yet carried on each hop;
that is the deadline story (#36), so deadline_unix_ms is sent as 0.

When a request ends, for any reason, it leaves the table and a Free for it is queued for
each stage (R4.6). In the protocol a Free travels to a stage on the heartbeat stream; the
heartbeat story (#28) drains these queues onto it. Until then frees_for(stage) returns them,
nothing delivers them, and the queues grow by two small messages per request. A request that
ends early (deadline, cancel, failure) can still have a step on the data stream; delivering
its Free on the heartbeat stream must not let that step recreate the cache afterwards, which
is a requirement on #28.

Not here: many requests in flight is the next story (#26; the code does not prevent it but
it is not tested). Requests share one stream to Stage 0, and a stage ends that stream when
one request's step fails, so every request on it then fails with UNAVAILABLE; #26 has to
keep one request's failure from ending the others. Each Generate holds a server thread for
its life. Also not here: the full result filter with epochs and exactly-once delivery (#30),
recovery and replay (#38, #39), bounded queues (#33). A result that does not match its
request's attempt and expected step is dropped and counted.
"""

import argparse
import json
import logging
import queue
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field

import grpc

from pllm import config, prompts, wire
from pllm.control import ControlService, Registry, expected_model
from pllm.model import ModelConfig
from pllm.models import MODELS_DIR, PINNED
from pllm.proto import protocol_pb2, protocol_pb2_grpc
from pllm.stage_service import _DONE, _drain

log = logging.getLogger("pllm.gateway")

POLL_SECONDS = 0.05  # how often a waiting request checks its deadline, client and link
MAX_UNSENT = 4  # a UTF-8 character spans at most four byte-level tokens
REPLACEMENT = chr(0xFFFD)  # what an incomplete UTF-8 character decodes to
STAGES = (0, 1)


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
        self.call = protocol_pb2_grpc.StageStub(self.channel).Steps.future(_drain(self.outbox))

    def failure(self) -> str | None:
        """Why the stream ended, or None while it is open."""
        if not self.call.done():
            return None
        try:
            self.call.result()
        except grpc.RpcError as err:
            return f"{err.code().name}: {err.details()}"
        except grpc.FutureCancelledError:
            return "cancelled"
        return "closed by stage 0"

    def close(self) -> None:
        self.outbox.put(_DONE)
        self.channel.close()


class Gateway(protocol_pb2_grpc.FrontendServicer, protocol_pb2_grpc.ResultsServicer):
    """tokenizer needs encode(text) -> ids and decode(ids) -> text; eos_ids end a request."""

    def __init__(self, registry: Registry, tokenizer, eos_ids, hidden_size: int, max_bytes: int):
        self.registry = registry
        self.tokenizer = tokenizer
        self.eos_ids = frozenset(eos_ids)
        self.hidden_size = hidden_size
        self.max_bytes = max_bytes
        self.options = wire.channel_options(hidden_size)
        self.requests: dict[str, Request] = {}
        self.frees = {stage: queue.Queue() for stage in STAGES}
        self.dropped = 0  # results that matched no waiting step
        self._lock = threading.Lock()
        self._link: Link | None = None

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

    def _end(self, request: Request, how: str) -> None:
        with self._lock:
            self.requests.pop(request.rid, None)
        for stage in STAGES:
            self.frees[stage].put(protocol_pb2.Free(rid=request.rid, attempt=request.attempt))
        log.info("request %s ended (%s) after %d tokens", request.rid, how, len(request.tokens))

    # ------------------------------------------------------------------ Stage 0 link

    def _stage0(self, context) -> Link:
        stages = self.registry.stages
        for stage in STAGES:
            if stage not in stages:
                context.abort(grpc.StatusCode.UNAVAILABLE, f"stage {stage} is not registered")
        entry = stages[0]
        with self._lock:
            link = self._link
            moved = link is not None and (link.addr, link.epoch) != (entry.addr, entry.epoch)
            if link is None or link.failure() or moved:
                if link is not None:
                    link.close()
                link = self._link = Link(entry.addr, entry.epoch, self.options)
        return link

    def close(self) -> None:
        with self._lock:
            if self._link is not None:
                self._link.close()
                self._link = None

    # ------------------------------------------------------------------ Results.Push

    def Push(self, request_iterator, context):
        for result in request_iterator:
            with self._lock:
                request = self.requests.get(result.rid)
                wanted = request is not None and (result.attempt, result.step) == (
                    request.attempt,
                    request.expected_step,
                )
                if not wanted:
                    self.dropped += 1
            if wanted:
                request.results.put(result)
            else:
                log.warning("dropped result %s step %d", result.rid, result.step)
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
        ids = self._admit(req, context)
        link = self._stage0(context)
        request = Request(
            rid=uuid.uuid4().hex[:12],
            attempt=1,
            deadline=time.monotonic() + req.timeout_ms / 1000,
            prompt_len=len(ids),
        )
        with self._lock:
            self.requests[request.rid] = request
        how = "failed"
        try:
            send, start = ids, 0
            unsent: list[int] = []  # tokens whose text is not complete yet
            while True:
                step = request.expected_step
                link.outbox.put(
                    protocol_pb2.StepRequest(
                        rid=request.rid,
                        attempt=request.attempt,
                        step=step,
                        start_pos=start,
                        length=len(send),
                        target_epoch=link.epoch,
                        sender_epoch=0,
                        tokens=protocol_pb2.TokenIds(ids=send),
                    )
                )
                result = self._wait(request, link, context)
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
        finally:
            self._end(request, how)

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
            try:
                result = request.results.get(timeout=min(left, POLL_SECONDS))
            except queue.Empty:
                pass
            else:
                # Checked again here: a duplicate queued just before the step moved on.
                if (result.attempt, result.step) == (request.attempt, request.expected_step):
                    return result
                with self._lock:
                    self.dropped += 1
            why = link.failure()
            if why is not None:
                context.abort(grpc.StatusCode.UNAVAILABLE, f"stage 0 stream failed: {why}")


def serve(gateway: Gateway, bind: str, max_workers: int = 16) -> tuple[grpc.Server, int]:
    """Serve Control, Frontend and Results on bind (port 0 picks a free one)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), options=gateway.options
    )
    protocol_pb2_grpc.add_ControlServicer_to_server(ControlService(gateway.registry), server)
    protocol_pb2_grpc.add_FrontendServicer_to_server(gateway, server)
    protocol_pb2_grpc.add_ResultsServicer_to_server(gateway, server)
    port = server.add_insecure_port(bind)
    if port == 0:
        raise OSError(f"could not listen on {bind}")
    server.start()
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


def eos_ids(model_dir) -> list[int]:
    eos = json.loads((model_dir / "config.json").read_text()).get("eos_token_id")
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else list(eos)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the gateway for one model.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--split-point", type=int, help="override the committed split point")
    args = parser.parse_args(argv)
    model_dir = MODELS_DIR / args.model
    try:
        hosts = config.load()
        expected = expected_model(args.model, args.split_point)
        hidden = ModelConfig.from_file(model_dir / "config.json").hidden_size
        tokenizer = Tokenizer(args.model)
    except (ValueError, OSError) as err:
        parser.error(str(err))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    gateway = Gateway(
        Registry(args.model, expected),
        tokenizer,
        eos_ids(model_dir),
        hidden,
        config.max_message_bytes(hidden),
    )
    server, _ = serve(gateway, f"0.0.0.0:{hosts.gateway.port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=1).wait()
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
