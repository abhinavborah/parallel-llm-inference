"""A wireless link that stalls: keepalive and beats that jitter (R6.3, #124).

A link that goes silent with no TCP reset is injected by a proxy on loopback that stops
forwarding while both of its sockets stay open, as a Wi-Fi stall or a laptop that left the
network looks to each end. The real stage heartbeat client talks through it to a gRPC
server with the shared options. Beats that jitter just under the heartbeat timeout run on
the detector's fake clock, since real beats on a loaded runner are late by more than the
margin such a test needs.
"""

import queue
import random
import socket
import threading
import time
from concurrent import futures
from types import SimpleNamespace

import grpc
import pytest
from test_detection import detected, fake

from pllm import gateway as gateway_module
from pllm import wire
from pllm.heartbeat import Heartbeat
from pllm.proto import protocol_pb2_grpc

HIDDEN = 64
SLACK = 5.0  # seconds on top of a stated bound, for a loaded runner


class Blackhole:
    """A TCP proxy to target. cut() makes every connection open now silent: its bytes are
    read and dropped and nothing is closed, so neither end sees a reset. A connection made
    after the cut passes, as one made after a Wi-Fi stall ends would."""

    def __init__(self, target: int):
        self.target = target
        self.cuts: list[threading.Event] = []
        self.released = threading.Event()
        self.sockets: list[socket.socket] = []
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def cut(self) -> float:
        for each in self.cuts:
            each.set()
        return time.monotonic()

    def _accept(self) -> None:
        while True:
            try:
                down, _ = self.listener.accept()
            except OSError:
                return
            up = socket.create_connection(("127.0.0.1", self.target))
            self.sockets += [down, up]
            cut = threading.Event()
            self.cuts.append(cut)
            for a, b in ((down, up), (up, down)):
                threading.Thread(target=self._pump, args=(a, b, cut), daemon=True).start()

    def _pump(self, a: socket.socket, b: socket.socket, cut: threading.Event) -> None:
        while not self.released.is_set():
            try:
                data = a.recv(65536)
                if not data:
                    return
                if not cut.is_set():
                    b.sendall(data)
            except OSError:
                return

    def close(self) -> None:
        self.released.set()
        self.listener.close()
        for each in self.sockets:
            each.close()


class Control(protocol_pb2_grpc.ControlServicer):
    """The gateway's side of the heartbeat stream, reduced to reading beats and noting when
    each stream ends."""

    def __init__(self):
        self.ended: queue.Queue = queue.Queue()

    def Heartbeat(self, request_iterator, context):
        def read():
            try:
                for _ in request_iterator:
                    pass
            except grpc.RpcError:
                pass
            self.ended.put(time.monotonic())

        threading.Thread(target=read, daemon=True).start()
        while context.is_active():
            time.sleep(0.05)
        return
        yield


@pytest.fixture
def link():
    made = []

    def make(options):
        control = Control()
        server = grpc.server(futures.ThreadPoolExecutor(4), options=options)
        protocol_pb2_grpc.add_ControlServicer_to_server(control, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        proxy = Blackhole(port)
        made.append((server, proxy))
        return control, proxy

    yield make
    for server, proxy in made:
        proxy.close()
        server.stop(0).wait()


def stage():
    """What Heartbeat reads from a stage servicer."""
    return SimpleNamespace(
        stage_index=0,
        own_epoch=1,
        inbox_depth=0,
        cached_requests=0,
        busy=(0, 0),
        nacks=queue.Queue(),
        free=lambda _: None,
        route=lambda _: None,
    )


def beat_through(proxy, options, interval, stop):
    heart = Heartbeat(
        f"127.0.0.1:{proxy.port}", stage(), interval=interval, retry=0.05, options=options
    )
    heart.start(stop)
    return heart


def wait_until(condition, limit):
    """Seconds until condition holds, or None if it does not within limit."""
    start = time.monotonic()
    while time.monotonic() - start < limit:
        if condition():
            return time.monotonic() - start
        time.sleep(0.02)
    return None


def test_the_shared_options_set_keepalive_with_a_bound_below_the_heartbeat_cap():
    options = dict(wire.channel_options(HIDDEN))
    assert options["grpc.keepalive_time_ms"] == 2000
    assert options["grpc.keepalive_timeout_ms"] == 8000
    assert options["grpc.http2.ping_timeout_ms"] == 8000  # else an in-flight ping waits 60 s
    assert options["grpc.http2.max_pings_without_data"] == 0
    assert options["grpc.http2.max_ping_strikes"] == 0  # no too_many_pings on idle streams
    assert wire.KEEPALIVE_BOUND_SECONDS == 10.0
    assert gateway_module.HEARTBEAT_TIMEOUT_SECONDS < wire.KEEPALIVE_TIMEOUT_SECONDS


def test_an_idle_stream_lives_and_a_silent_link_breaks_within_the_bound(link):
    """The shipped values: about 12 s idle, then the cut, detected by about 10 s."""
    options = wire.channel_options(HIDDEN)
    control, proxy = link(options)
    stop = threading.Event()
    try:
        heart = beat_through(proxy, options, interval=600.0, stop=stop)  # one beat, then idle
        assert wait_until(lambda: heart.streams == 1, 5) is not None
        time.sleep(6 * wire.KEEPALIVE_SECONDS)  # pings on a stream with no data
        assert heart.streams == 1 and control.ended.empty(), "keepalive ended an idle stream"
        bound = wire.KEEPALIVE_BOUND_SECONDS + SLACK
        cut = proxy.cut()
        assert wait_until(lambda: heart.streams >= 2, bound) is not None, "stage side"
        assert control.ended.get(timeout=bound) - cut < bound, "server side"
    finally:
        stop.set()


def test_a_busy_stream_on_a_silent_link_breaks_within_a_short_bound(link, monkeypatch):
    """The same wiring with short values, beats flowing when the link goes silent."""
    monkeypatch.setattr(wire, "KEEPALIVE_SECONDS", 0.5)
    monkeypatch.setattr(wire, "KEEPALIVE_TIMEOUT_SECONDS", 0.5)
    options = wire.channel_options(HIDDEN)
    control, proxy = link(options)
    stop = threading.Event()
    try:
        heart = beat_through(proxy, options, interval=0.1, stop=stop)
        assert wait_until(lambda: heart.streams == 1, 5) is not None
        time.sleep(1.0)
        cut = proxy.cut()
        assert wait_until(lambda: heart.streams >= 2, 1.0 + SLACK) is not None, "stage side"
        assert control.ended.get(timeout=1.0 + SLACK) - cut < 1.0 + SLACK, "server side"
    finally:
        stop.set()


def test_without_keepalive_a_silent_link_is_not_detected(link):
    """The control: the same cut with gRPC's defaults goes unnoticed past the short bound."""
    options = [("grpc.max_receive_message_length", 1 << 20)]
    control, proxy = link(options)
    stop = threading.Event()
    try:
        heart = beat_through(proxy, options, interval=0.1, stop=stop)
        assert wait_until(lambda: heart.streams == 1, 5) is not None
        proxy.cut()
        time.sleep(1.0 + 2.0)
        assert heart.streams == 1 and control.ended.empty()
    finally:
        stop.set()


def test_beats_that_jitter_just_under_the_timeout_declare_nothing():
    timeout = 2.0
    detector, clock = fake(heartbeat_timeout=timeout)
    rng = random.Random(124)
    detector.beat(0, 1)
    for _ in range(1000):
        gap = rng.uniform(0.9 * timeout, timeout - 1e-6)
        clock.advance(gap)  # the oldest the beat gets before the next one
        detector.expire()
        assert detector.beat(0, 1)
    assert detected(detector) == []
    clock.advance(timeout)  # exactly the timeout: declared
    detector.expire()
    assert [e["path"] for e in detected(detector)] == ["heartbeat_timeout"]


def test_the_gateway_refuses_a_heartbeat_timeout_at_the_keepalive_timeout(monkeypatch):
    def accepted():
        raise AssertionError("the value was accepted")

    monkeypatch.setattr(gateway_module.config, "load", accepted)  # never starts a gateway
    model = sorted(gateway_module.PINNED)[0]
    for value in (wire.KEEPALIVE_TIMEOUT_SECONDS, wire.KEEPALIVE_TIMEOUT_SECONDS + 1):
        with pytest.raises(SystemExit):
            gateway_module.main([model, "--heartbeat-timeout", str(value)])
