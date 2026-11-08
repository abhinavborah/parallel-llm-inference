"""The network service that hosts a stage worker (R3.4).

Each stage serves Stage.Steps: an ordered stream of StepRequest messages from the
previous hop, handled one at a time in arrival order within a stream. A stream that fails
at the next hop is ended with UNAVAILABLE even while the previous hop holds it open.

- Stage 0 receives token ids, runs its layers and forwards the hidden states to stage 1
  as a StepRequest carrying a float32 Tensor.
- The last stage receives hidden states, runs its layers and the LM head, takes the
  greedy token and its logit, and pushes a StepResult to the gateway's Results.Push.

Each request id has its own KV cache, created by its first message at position 0. The
position rules (truncate, append, reject a gap), release and fencing by epoch are separate
stories; here start_pos is passed to the stage unchanged and the epoch fields are carried
but not checked. Caches are not freed until the Free story, and two streams that carry the
same request id at once are not guarded against each other (fencing, not this story). The
outbound queue is unbounded; bounded queues and backpressure come with the queue story.

Chain over a single host or two: every address comes from the caller, which reads it from
the hosts file. Nothing here depends on the operating system (R3.15).
"""

import logging
import queue
import threading
from concurrent import futures

import grpc
import torch

from pllm import wire
from pllm.proto import protocol_pb2, protocol_pb2_grpc
from pllm.worker import StageWorker

log = logging.getLogger("pllm.stage_service")

_DONE = object()  # ends an outbound stream
POLL_SECONDS = 0.05  # how often an idle handler checks that the next hop is still up


def _drain(outbox: queue.Queue):
    """Messages for an outbound stream, in the order they were put."""
    while (item := outbox.get()) is not _DONE:
        yield item


class StageServicer(protocol_pb2_grpc.StageServicer):
    """Serves Stage.Steps for one worker.

    next_address is the next stage's Stage service, or None for the last stage, which
    pushes to results_address instead. own_epoch and next_epoch go into the sender and
    target epoch fields of the forwarded message; they are 0 until registration supplies
    them.
    """

    def __init__(
        self,
        worker: StageWorker,
        *,
        next_address: str | None = None,
        results_address: str | None = None,
        own_epoch: int = 0,
        next_epoch: int = 0,
    ):
        last = worker.stage.last
        if last and results_address is None:
            raise ValueError("the last stage needs the gateway's results address")
        if not last and next_address is None:
            raise ValueError("a stage that is not last needs the next stage's address")
        self.worker = worker
        self.last = last
        self.own_epoch, self.next_epoch = own_epoch, next_epoch
        options = wire.channel_options(worker.stage.config.hidden_size)
        self.channel = grpc.insecure_channel(results_address if last else next_address, options)
        self.caches: dict[str, object] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        self.channel.close()

    # ------------------------------------------------------------------ one hop

    def Steps(self, request_iterator, context):
        outbox: queue.Queue = queue.Queue()
        if self.last:
            call = protocol_pb2_grpc.ResultsStub(self.channel).Push.future(_drain(outbox))
        else:
            call = protocol_pb2_grpc.StageStub(self.channel).Steps.future(_drain(outbox))
        # A reader thread feeds the inbox, so this loop can notice a failed next hop while
        # the previous hop holds its stream open and waits (it would otherwise block in
        # the iterator until its own timeout).
        inbox: queue.Queue = queue.Queue()

        def pump():
            try:
                for request in request_iterator:
                    inbox.put(request)
            except grpc.RpcError:
                pass  # the previous hop cancelled or dropped; this handler is ending too
            finally:
                inbox.put(_DONE)

        threading.Thread(target=pump, daemon=True).start()
        try:
            while not call.done():
                try:
                    request = inbox.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    continue
                if request is _DONE:
                    break
                outbox.put(self._handle(request, context))
        finally:
            outbox.put(_DONE)
        try:
            call.result()
        except grpc.RpcError as err:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"next hop failed: {err.code().name}: {err.details()}",
            )
        return protocol_pb2.Empty()

    def _handle(self, request: protocol_pb2.StepRequest, context):
        inputs = self._inputs(request, context)
        if len(inputs) != request.length:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"length {request.length} but the payload carries {len(inputs)} positions",
            )
        cache = self._cache(request, context)
        try:
            out = self.worker.step(inputs, request.start_pos, cache)
        except (ValueError, IndexError) as err:
            self._drop(request.rid)  # the cache may be half written
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(err))
        except Exception as err:
            self._drop(request.rid)
            log.exception("step %d of %r failed", request.step, request.rid)
            context.abort(grpc.StatusCode.INTERNAL, f"stage failed: {type(err).__name__}")
        if self.last:
            top, best = out[-1].max(dim=-1)  # same greedy choice as the reference runner
            chosen, logit = int(best.item()), float(top.item())
            return protocol_pb2.StepResult(
                rid=request.rid,
                attempt=request.attempt,
                step=request.step,
                epoch=request.target_epoch,
                token=chosen,
                top_logit=logit,
            )
        return protocol_pb2.StepRequest(
            rid=request.rid,
            attempt=request.attempt,
            step=request.step,
            start_pos=request.start_pos,
            length=request.length,
            target_epoch=self.next_epoch,
            sender_epoch=self.own_epoch,
            replay=request.replay,
            deadline_unix_ms=request.deadline_unix_ms,
            activations=wire.tensor_to_proto(out),
        )

    def _inputs(self, request, context) -> torch.Tensor:
        kind = request.WhichOneof("payload")
        first = self.worker.stage.first
        if kind != ("tokens" if first else "activations"):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"stage {'0' if first else '1'} takes {'tokens' if first else 'activations'}, "
                f"got {kind or 'no payload'}",
            )
        if first:
            return torch.tensor(list(request.tokens.ids), dtype=torch.int64)
        try:
            return wire.tensor_from_proto(request.activations)
        except wire.BadTensor as err:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(err))

    def _drop(self, rid: str) -> None:
        """Forget a request's cache after a failed step, so a retry from position 0 starts
        clean and a later position is refused as unknown (the gateway replays)."""
        with self._lock:
            self.caches.pop(rid, None)

    def _cache(self, request, context):
        with self._lock:
            cache = self.caches.get(request.rid)
            if cache is None:
                if request.start_pos != 0:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        f"request {request.rid!r} is unknown here and starts at "
                        f"position {request.start_pos}, not 0",
                    )
                cache = self.caches[request.rid] = self.worker.new_cache()
        return cache


def serve(
    worker: StageWorker,
    bind: str,
    *,
    next_address: str | None = None,
    results_address: str | None = None,
    own_epoch: int = 0,
    next_epoch: int = 0,
    max_workers: int = 8,
) -> tuple[grpc.Server, StageServicer, int]:
    """Start serving a worker on bind (for example '0.0.0.0:50051'; port 0 picks a free
    one). Returns the running server, its servicer and the port it listens on."""
    servicer = StageServicer(
        worker,
        next_address=next_address,
        results_address=results_address,
        own_epoch=own_epoch,
        next_epoch=next_epoch,
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=wire.channel_options(worker.stage.config.hidden_size),
    )
    protocol_pb2_grpc.add_StageServicer_to_server(servicer, server)
    port = server.add_insecure_port(bind)
    if port == 0:
        raise OSError(f"could not listen on {bind}")
    server.start()
    log.info("stage %s listening on port %d", "last" if worker.stage.last else "first", port)
    return server, servicer, port
