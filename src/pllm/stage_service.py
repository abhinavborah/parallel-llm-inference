"""The network service that hosts a stage worker (R3.4).

Each stage serves Stage.Steps: an ordered stream of StepRequest messages from the
previous hop, handled one at a time in arrival order within a stream. A stream that fails
at the next hop is ended with UNAVAILABLE even while the previous hop holds it open.

- Stage 0 receives token ids, runs its layers and forwards the hidden states to stage 1
  as a StepRequest carrying a float32 Tensor.
- The last stage receives hidden states, runs its layers and the LM head, takes the
  greedy token and its logit, and pushes a StepResult to the gateway's Results.Push.

Each request id has its own KV cache in a KVStore, with the position rules of R3.6: equal
appends, lower truncates then appends, higher is refused. A refused message is skipped,
not forwarded, and a Nack goes on the nacks queue: GAP for a position past the cache,
STALE_ATTEMPT for an attempt of a request that was already freed. The stream and the other
requests on it carry on. STALE_ATTEMPT thus covers two cases: the fencing of an older attempt
(R3.11, the fencing story) and a late step of an attempt that was freed, which is normal after
a deadline or a cancel. A reader of these Nacks treats one for a request that has already
ended as harmless; only one for a running request calls for recovery.
pllm.heartbeat sends the nacks queue to the gateway on the heartbeat stream (R3.10,
R3.12), hands each Free it receives to free(), and each route update to route() (R3.13).

route() points the first stage at a restarted second stage: a new channel to its address
and its epoch as the target epoch. A stream already open keeps the channel it started
with; it ends when that next hop is gone, and the previous hop's next stream uses the new
route. The channel is new even at the same address, because a channel that failed to
connect while the stage was down waits out gRPC's reconnect backoff (up to 120 s); each
channel keeps its own connections, since gRPC otherwise shares them between channels to
one address and the new channel would inherit the backoff.
Fencing by epoch is a separate story: the epoch fields are carried but not checked,
and two streams that carry the same request id at once are not guarded against each other.
The outbound queue is unbounded; bounded queues and backpressure come with the queue story.

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
from pllm.kvstore import Gap, KVStore, Stale
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
    target epoch fields of the forwarded message; they are 0 until registration and the
    first route update supply them.
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
        # Its own connections: by default gRPC shares them between channels to the same
        # address, so a new channel would inherit one still in reconnect backoff.
        self.options = [
            *wire.channel_options(worker.stage.config.hidden_size),
            ("grpc.use_local_subchannel_pool", 1),
        ]
        self.next_address = results_address if last else next_address
        self.channel = grpc.insecure_channel(self.next_address, self.options)
        self._route_lock = threading.Lock()
        self._inboxes: set[queue.Queue] = set()  # one per open Steps stream
        layers = worker.stage.layer_end - worker.stage.layer_start
        self.store = KVStore(worker.stage.config, layers, worker.new_cache)
        self.stage_index = 0 if worker.stage.first else 1
        self.nacks: queue.Queue = queue.Queue()  # Nacks waiting for the heartbeat stream

    def close(self) -> None:
        with self._route_lock:
            self.channel.close()

    @property
    def cached_requests(self) -> int:
        """The count a Beat reports."""
        return len(self.store)

    @property
    def inbox_depth(self) -> int:
        """Messages received and not yet handled, over every open stream; a Beat reports it."""
        with self._route_lock:
            return sum(inbox.qsize() for inbox in self._inboxes)

    def free(self, message: protocol_pb2.Free) -> bool:
        """Release a request's cache (R3.9) and refuse later messages of that attempt or an
        older one. Attempts start at 1, so attempt 0 (the field left unset) frees whatever is
        cached and remembers nothing. Freeing one that is not here does nothing."""
        return self.store.free(message.rid, message.attempt or None)

    def route(self, update: protocol_pb2.RouteUpdate) -> bool:
        """Send to the next stage at update.addr with target epoch update.epoch (R3.13).

        Only the first stage has a next stage; an update for any other hop is ignored, and
        so is one that changes nothing or names an older epoch than the current route;
        all return False. Streams opened from now on use the new route, on a new channel
        (see the module notes). If the address changed, the old channel is closed, which
        ends any stream still open on it; at the same address the old channel is left to
        the streams still using it, and is closed once they end and release it.
        """
        if self.last or update.next_stage != self.stage_index + 1 or not update.addr:
            log.warning("ignored route update for stage %d", update.next_stage)
            return False
        with self._route_lock:
            if update.epoch < self.next_epoch or (update.addr, update.epoch) == (
                self.next_address,
                self.next_epoch,
            ):
                return False
            old = self.channel if update.addr != self.next_address else None
            self.channel = grpc.insecure_channel(update.addr, self.options)
            self.next_address, self.next_epoch = update.addr, update.epoch
        if old is not None:
            old.close()
        log.info("next stage is now %s with epoch %d", update.addr, update.epoch)
        return True

    # ------------------------------------------------------------------ one hop

    def Steps(self, request_iterator, context):
        outbox: queue.Queue = queue.Queue()
        inbox: queue.Queue = queue.Queue()
        with self._route_lock:  # the route this stream keeps for its life
            channel, next_epoch = self.channel, self.next_epoch
            self._inboxes.add(inbox)
        try:
            if self.last:
                call = protocol_pb2_grpc.ResultsStub(channel).Push.future(_drain(outbox))
            else:
                call = protocol_pb2_grpc.StageStub(channel).Steps.future(_drain(outbox))
        except ValueError:  # the channel was closed by a route update a moment ago
            with self._route_lock:
                self._inboxes.discard(inbox)
            context.abort(grpc.StatusCode.UNAVAILABLE, "the next hop moved; send again")
        # A reader thread feeds the inbox, so this loop can notice a failed next hop while
        # the previous hop holds its stream open and waits (it would otherwise block in
        # the iterator until its own timeout).

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
                out = self._handle(request, context, next_epoch)
                if out is not None:  # None: refused with a Nack, nothing to pass on
                    outbox.put(out)
        finally:
            outbox.put(_DONE)
            with self._route_lock:
                self._inboxes.discard(inbox)
        try:
            call.result()
        except grpc.RpcError as err:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"next hop failed: {err.code().name}: {err.details()}",
            )
        return protocol_pb2.Empty()

    def _nack(self, request: protocol_pb2.StepRequest, reason: str) -> None:
        self.nacks.put(
            protocol_pb2.Nack(
                stage=self.stage_index,
                epoch=self.own_epoch,
                rid=request.rid,
                attempt=request.attempt,
                step=request.step,
                reason=reason,
            )
        )

    def _handle(self, request: protocol_pb2.StepRequest, context, next_epoch: int):
        inputs = self._inputs(request, context)
        if len(inputs) != request.length:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"length {request.length} but the payload carries {len(inputs)} positions",
            )
        try:
            cache = self.store.acquire(request.rid, request.start_pos, request.attempt)
        except Stale as stale:
            log.warning("STALE_ATTEMPT: %s", stale)
            self._nack(request, "STALE_ATTEMPT")
            return None
        except Gap as gap:
            log.warning("GAP: %s", gap)
            self._nack(request, "GAP")
            return None
        try:
            out = self.worker.step(inputs, request.start_pos, cache)
        except (ValueError, IndexError) as err:
            self.store.free_cache(request.rid, cache)  # the cache may be half written
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(err))
        except Exception as err:
            self.store.free_cache(request.rid, cache)
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
            target_epoch=next_epoch,
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
