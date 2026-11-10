"""A stage's side of Control.Heartbeat (R3.10, R3.12, R3.13).

After a stage registers it opens one bidirectional stream to the gateway and keeps it:

- Every interval seconds it sends a Beat with its stage, epoch, inbox depth, cached
  request count, and running totals of the time it has spent handling steps and the steps
  it has handled (durations on its own clock; the gateway stamps each Beat with its
  own). Each Nack the stage raises is sent as soon as it is on the nacks queue, between
  beats, so the gateway learns of a refusal without waiting for a timeout.
- From the gateway it receives Free messages, which release a request's cache on this
  stage, and route updates, which point the first stage at a restarted second stage.

If the gateway is not reachable, or the stream breaks, the stage opens a new stream after
retry seconds. If the gateway ends the stream with FAILED_PRECONDITION, a newer
registration of this stage has replaced this one, and the stage stops sending heartbeats
but keeps serving (its late outputs are for the fencing story to reject). A restarted
gateway also answers FAILED_PRECONDITION, since it holds no registrations; only workers
are restarted, so that case is not handled.

The stage's nacks queue is unbounded; while the gateway cannot be reached, Nacks wait in it.

A message taken from the nacks queue just before the stream breaks can be lost with it,
and so can a Free the gateway sent at that moment. A lost Free leaves that request's cache
on the stage until the process ends.

Nothing here depends on the operating system (R3.15).
"""

import logging
import queue
import threading
import time

import grpc

from pllm.proto import protocol_pb2, protocol_pb2_grpc

log = logging.getLogger("pllm.heartbeat")

HEARTBEAT_SECONDS = 0.5  # between beats
RETRY_SECONDS = 1.0  # before a new stream after one fails
POLL_SECONDS = 0.05  # how often the sender checks that it should stop


def beat(servicer) -> protocol_pb2.StageMsg:
    """The Beat for a stage servicer's current state."""
    busy_ns, steps = servicer.busy
    return protocol_pb2.StageMsg(
        beat=protocol_pb2.Beat(
            stage=servicer.stage_index,
            epoch=servicer.own_epoch,
            inbox_depth=servicer.inbox_depth,
            cached_requests=servicer.cached_requests,
            busy_ns=busy_ns,
            steps=steps,
        )
    )


class Heartbeat:
    """Keeps a stage's heartbeat stream to the gateway open until stop is set.

    servicer is the stage's StageServicer: its own_epoch, inbox_depth, cached_requests and
    nacks queue feed the stream, and its free and route methods take what comes back.
    """

    def __init__(
        self,
        gateway: str,
        servicer,
        interval: float = HEARTBEAT_SECONDS,
        retry: float = RETRY_SECONDS,
        options: list[tuple[str, int]] | None = None,
    ):
        if interval <= 0 or retry <= 0:
            raise ValueError("interval and retry must be positive")
        self.gateway = gateway
        self.servicer = servicer
        self.interval = interval
        self.retry = retry
        self.options = options
        self.streams = 0  # streams opened, for the logs and the tests
        self.superseded = False

    def run(self, stop: threading.Event) -> None:
        """Send heartbeats until stop is set or a newer registration replaces this stage."""
        with grpc.insecure_channel(self.gateway, options=self.options) as channel:
            stub = protocol_pb2_grpc.ControlStub(channel)
            while not stop.is_set():
                ended = threading.Event()
                self.streams += 1
                try:
                    for message in stub.Heartbeat(self._outgoing(stop, ended)):
                        self._apply(message)
                except grpc.RpcError as err:
                    if err.code() == grpc.StatusCode.FAILED_PRECONDITION:
                        self.superseded = True
                        log.error("heartbeat refused: %s; stopping heartbeats", err.details())
                        return
                    if not stop.is_set():
                        log.warning(
                            "heartbeat to %s failed (%s); retrying", self.gateway, err.code().name
                        )
                finally:
                    ended.set()
                stop.wait(self.retry)

    def start(self, stop: threading.Event) -> threading.Thread:
        thread = threading.Thread(target=self.run, args=(stop,), name="heartbeat", daemon=True)
        thread.start()
        return thread

    def _outgoing(self, stop: threading.Event, ended: threading.Event):
        while not (stop.is_set() or ended.is_set()):
            yield beat(self.servicer)
            due = time.monotonic() + self.interval
            while not (stop.is_set() or ended.is_set()):
                left = due - time.monotonic()
                if left <= 0:
                    break
                try:
                    nack = self.servicer.nacks.get(timeout=min(left, POLL_SECONDS))
                except queue.Empty:
                    continue
                yield protocol_pb2.StageMsg(nack=nack)

    def _apply(self, message: protocol_pb2.ControlMsg) -> None:
        kind = message.WhichOneof("msg")
        if kind == "free":
            self.servicer.free(message.free)
        elif kind == "route_update":
            self.servicer.route(message.route_update)
        else:
            log.warning("ignored an empty control message")
