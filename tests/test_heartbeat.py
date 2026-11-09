"""The heartbeat stream: beats, Nacks, frees and route updates (R3.10, R3.12, R3.13).

Each test runs a real gateway and real stage servers on loopback, with each stage's
heartbeat stream open as a worker opens it after registering.
"""

import queue
import socket
import threading
import time

import grpc
import pytest
from test_gateway import DIGEST, PROMPT, WAIT, System, code_of, wait_for
from test_model import tiny_reference

from pllm import control
from pllm.control import ExpectedModel, Registry
from pllm.gateway import Gateway, serve
from pllm.heartbeat import Heartbeat
from pllm.model import ModelConfig
from pllm.models import sha256_of
from pllm.proto import protocol_pb2, protocol_pb2_grpc
from pllm.reference import generate
from pllm.split import InProcessSplit
from pllm.stage_service import _DONE, _drain
from pllm.stage_service import serve as serve_stage
from pllm.worker import StageWorker, WorkerConfig, join_gateway

LAYERS = 4
FAST = 0.05  # heartbeat interval in these tests


class Beating(System):
    """A System whose stages registered over gRPC and keep heartbeat streams open."""

    def __init__(self, model_dir, k, interval=FAST, order=(1, 0), **kwargs):
        super().__init__(model_dir, k, register=False, **kwargs)
        self.model_dir, self.k = model_dir, k
        self.interval = interval
        self.stops: list[threading.Event] = []
        self.threads: list[threading.Thread] = []
        self.hearts: dict[int, Heartbeat] = {}
        self.extra = []  # servers started by a test
        for stage in order:
            self.join(stage, self.servicer1 if stage == 1 else self.servicer0)
        if set(order) == {0, 1}:
            self.wait_routes()

    def wait_routes(self, servicer0=None, servicer1=None):
        """Wait until each stage has the other's epoch from its route update, as the
        stages are up for requests only then (a request sent earlier is refused)."""
        s0, s1 = servicer0 or self.servicer0, servicer1 or self.servicer1
        wait_for(
            lambda: s0.next_epoch == s1.own_epoch and s1.upstream_epoch == s0.own_epoch,
            "both route updates",
        )

    def join(self, stage, servicer, port=None):
        """Register a stage through Control.Register and open its heartbeat stream."""
        port = port or self.ports[stage]
        start, end = self.layers[stage]
        with grpc.insecure_channel(self.address) as channel:
            servicer.own_epoch = control.register(
                channel, stage, (start, end), "tiny", DIGEST.hex(), f"127.0.0.1:{port}"
            )
        self.hearts[stage] = self.heart(servicer)
        return servicer.own_epoch

    def heart(self, servicer, **kwargs):
        heart = Heartbeat(
            self.address, servicer, interval=kwargs.pop("interval", self.interval), retry=FAST
        )
        stop = threading.Event()
        self.stops.append(stop)
        self.threads.append(heart.start(stop))
        return heart

    def stop_stage1(self):
        self.servicer1.close()
        self.server1.stop(0).wait()

    def restart_stage1(self, same_port=False):
        """A new stage 1 process, registered with a new epoch: on a new port, or on the
        one the old process used, as a worker restarted from the hosts file is."""
        n = self.layers[1][1]
        self.stop_stage1()
        bind = f"127.0.0.1:{self.ports[1]}" if same_port else "127.0.0.1:0"
        server, servicer, port = serve_stage(
            StageWorker(WorkerConfig(self.model_dir, self.k, n)),
            bind,
            results_address=self.address,
        )
        self.extra.append((server, servicer))
        epoch = self.join(1, servicer, port)
        return servicer, port, epoch

    def close(self):
        for stop in self.stops:
            stop.set()
        for thread in self.threads:
            thread.join(WAIT)
        for server, servicer in self.extra:
            server.stop(0).wait()
            servicer.close()
        super().close()


@pytest.fixture
def beating(tmp_path):
    made = []

    def make(k=2, **kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        made.append(Beating(tmp_path, k, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def count_beats(gateway):
    """Count every Beat the gateway records, per stage."""
    counts = {0: 0, 1: 0}
    real = gateway._record

    def spy(session, message):
        if message.WhichOneof("msg") == "beat":
            counts[session.stage] += 1
        real(session, message)

    gateway._record = spy
    return counts


def reference(model_dir, k, ids, n):
    return generate(InProcessSplit.load(model_dir, k), ids, n)


# -------------------------------------------------------------- beats


def test_each_stage_beats_its_stage_epoch_and_cached_count_at_the_interval(beating):
    run = beating(interval=0.1)
    counts = count_beats(run.gateway)
    wait_for(lambda: set(run.gateway.beats()) == {0, 1}, "a beat from both stages")
    time.sleep(1.0)
    for stage, servicer in ((0, run.servicer0), (1, run.servicer1)):
        assert 6 <= counts[stage] <= 14, counts  # about ten beats a second at 0.1 s
        latest = run.gateway.beats()[stage]
        assert latest["epoch"] == servicer.own_epoch == run.registry.stages[stage].epoch
        assert latest["cached_requests"] == 0 and latest["inbox_depth"] == 0
        assert latest["age_s"] < 0.5


def test_a_beat_reports_the_cached_count(beating):
    run = beating()
    run.servicer0.free = run.servicer1.free = lambda message: False  # keep the caches
    run.tokens(max_tokens=2)
    wait_for(
        lambda: all(b["cached_requests"] == 1 for b in run.gateway.beats().values()),
        "both stages to report one cached request",
    )


def test_a_beat_reports_the_inbox_depth(beating):
    run = beating(delay1=0.3)
    # four prefills straight to stage 0: it forwards them at once, stage 1 takes 0.3 s each
    with grpc.insecure_channel(f"127.0.0.1:{run.ports[0]}") as channel:
        outbox = queue.Queue()
        call = protocol_pb2_grpc.StageStub(channel).Steps.future(_drain(outbox))
        for rid in "abcd":
            outbox.put(
                protocol_pb2.StepRequest(
                    rid=rid,
                    attempt=1,
                    start_pos=0,
                    length=2,
                    target_epoch=run.servicer0.own_epoch,
                    tokens=protocol_pb2.TokenIds(ids=[5, 9]),
                )
            )
        wait_for(
            lambda: run.gateway.beats().get(1, {}).get("inbox_depth", 0) >= 2,
            "stage 1 to report a backlog",
        )
        outbox.put(_DONE)
        call.result(timeout=WAIT)
    wait_for(lambda: run.servicer1.inbox_depth == 0, "the backlog to drain")


# -------------------------------------------------------------- frees


def test_frees_travel_on_the_heartbeat_and_empty_both_caches(beating):
    run = beating()
    for _ in range(3):
        run.tokens(max_tokens=3)
        wait_for(
            lambda: run.servicer0.cached_requests == run.servicer1.cached_requests == 0,
            "both stages to free the request",
        )
    assert all(run.gateway.frees[stage].empty() for stage in (0, 1))


def test_a_step_still_at_stage0_when_its_free_reaches_stage1_is_refused_there(beating):
    run = beating(delay0=1.0)
    code, _ = code_of(run.call(timeout_ms=300))
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED
    # stage 1 had the Free before stage 0 finished the prefill and forwarded it
    wait_for(lambda: run.gateway.nack_count >= 1, "a Nack from stage 1")
    nack = run.gateway.nacks[0]
    assert (nack.stage, nack.reason, nack.attempt) == (1, "STALE_ATTEMPT", 1)
    assert run.servicer0.cached_requests == run.servicer1.cached_requests == 0


def test_a_request_past_its_deadline_is_freed_on_both_stages(beating):
    run = beating(delay1=1.0)
    code, _ = code_of(run.call(timeout_ms=300))
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED
    # stage 1 was still running the step when its Free arrived; nothing is left after it
    time.sleep(1.2)
    assert run.servicer0.cached_requests == run.servicer1.cached_requests == 0


def test_a_cancelled_request_is_freed_on_both_stages(beating):
    run = beating(delay1=0.5)
    call = run.call(max_tokens=50)
    next(call)
    call.cancel()
    wait_for(lambda: not run.gateway.requests, "the request to leave the table")
    time.sleep(1.0)
    assert run.servicer0.cached_requests == run.servicer1.cached_requests == 0


def test_a_step_that_arrives_after_its_free_does_not_cache_again(beating):
    run = beating(interval=5.0)  # a Nack must not wait for the next beat
    run.servicer0.free(protocol_pb2.Free(rid="late", attempt=1))
    began = time.monotonic()
    with grpc.insecure_channel(f"127.0.0.1:{run.ports[0]}") as channel:
        step = protocol_pb2.StepRequest(
            rid="late",
            attempt=1,
            start_pos=0,
            length=2,
            target_epoch=run.servicer0.own_epoch,
            tokens=protocol_pb2.TokenIds(ids=[5, 9]),
        )
        protocol_pb2_grpc.StageStub(channel).Steps(iter([step]), timeout=WAIT)
    wait_for(lambda: run.gateway.nack_count == 1, "the Nack at the gateway")
    assert time.monotonic() - began < 2.0
    nack = run.gateway.nacks[0]
    assert (nack.stage, nack.rid, nack.attempt, nack.reason) == (0, "late", 1, "STALE_ATTEMPT")
    assert nack.epoch == run.servicer0.own_epoch
    assert run.servicer0.cached_requests == run.servicer1.cached_requests == 0


def test_a_gap_nack_reaches_the_gateway_between_beats(beating):
    run = beating(interval=5.0)
    with grpc.insecure_channel(f"127.0.0.1:{run.ports[0]}") as channel:
        step = protocol_pb2.StepRequest(
            rid="lost",
            attempt=1,
            step=7,
            start_pos=5,
            length=1,
            target_epoch=run.servicer0.own_epoch,
            tokens=protocol_pb2.TokenIds(ids=[1]),
        )
        protocol_pb2_grpc.StageStub(channel).Steps(iter([step]), timeout=WAIT)
    wait_for(lambda: run.gateway.nack_count == 1, "the Nack at the gateway", timeout=2.0)
    assert (run.gateway.nacks[0].reason, run.gateway.nacks[0].step) == ("GAP", 7)


def test_frees_queued_before_a_stage_registers_again_are_dropped(beating):
    run = beating()
    run.stops[-1].set()  # stage 0's heartbeat stops, so its frees wait
    run.threads[-1].join(WAIT)
    run.gateway.frees[0].put(protocol_pb2.Free(rid="old", attempt=1))
    run.join(0, run.servicer0)
    assert run.gateway.frees[0].empty()


# -------------------------------------------------------------- route updates


def test_stage0_reaches_a_restarted_stage1_without_restarting(beating, tmp_path):
    run = beating()
    expected = reference(tmp_path, 2, [5, 9, 100, 3, 42, 7], 4)
    got = run.tokens(max_tokens=4)
    assert [(t.token, t.top_logit) for t in got] == list(zip(*expected, strict=True))
    server0, servicer0 = run.server0, run.servicer0
    new1, port1, epoch1 = run.restart_stage1()
    wait_for(
        lambda: (servicer0.next_address, servicer0.next_epoch) == (f"127.0.0.1:{port1}", epoch1),
        "stage 0 to take the route update",
    )
    seen = []
    real = new1._handle

    def spy(request, context, *rest):
        seen.append(request.target_epoch)
        return real(request, context, *rest)

    new1._handle = spy
    # the stream the gateway held to stage 0 ended with the old route
    wait_for(lambda: run.gateway._link.failure() is not None, "the old stream to end")
    got = run.tokens(max_tokens=4)
    assert [(t.token, t.top_logit) for t in got] == list(zip(*expected, strict=True))
    assert seen == [epoch1] * 4
    assert run.server0 is server0 and run.servicer0 is servicer0  # stage 0 did not restart


def test_stage0_reaches_stage1_restarted_on_the_same_port(beating, tmp_path):
    """A worker restarted from the hosts file comes back at the same address. Requests
    that fail while it is down put Stage 0's old channel into reconnect backoff; the
    route update gives Stage 0 a new channel, so the next request goes through at once."""
    run = beating()
    expected = reference(tmp_path, 2, [5, 9, 100, 3, 42, 7], 3)
    run.tokens(max_tokens=3)
    run.stop_stage1()
    for _ in range(3):
        code, _ = code_of(run.call(max_tokens=3))
        assert code == grpc.StatusCode.UNAVAILABLE
    servicer1, port1, epoch1 = run.restart_stage1(same_port=True)
    assert port1 == run.ports[1]
    wait_for(lambda: run.servicer0.next_epoch == epoch1, "the route update")
    began = time.monotonic()
    got = run.tokens(max_tokens=3)
    assert time.monotonic() - began < 2.0
    assert [(t.token, t.top_logit) for t in got] == list(zip(*expected, strict=True))


def test_stage0_gets_the_route_when_its_stream_opens(beating):
    run = beating()
    stage1 = run.registry.stages[1]
    wait_for(
        lambda: (
            (run.servicer0.next_address, run.servicer0.next_epoch) == (stage1.addr, stage1.epoch)
        ),
        "the route Stage 0 receives when its stream opens",
        timeout=2.0,
    )


def test_the_route_reaches_stage0_when_stage1_registers_after_it(beating):
    run = beating(order=(0,))
    wait_for(lambda: 0 in run.gateway.beats(), "stage 0's first beat")
    assert run.servicer0.next_epoch == 0
    epoch1 = run.join(1, run.servicer1)
    wait_for(lambda: run.servicer0.next_epoch == epoch1, "the route update")
    assert len(run.tokens(max_tokens=2)) == 2


def test_a_route_update_gives_a_new_channel_even_at_the_same_address(beating):
    run = beating()
    servicer = run.servicer0
    wait_for(lambda: servicer.next_epoch == run.registry.stages[1].epoch, "the first route")
    channel = servicer.channel
    update = protocol_pb2.RouteUpdate(next_stage=1, addr=servicer.next_address, epoch=99)
    assert servicer.route(update) is True and servicer.channel is not channel
    assert servicer.next_epoch == 99
    assert servicer.route(update) is False  # nothing changed
    older = protocol_pb2.RouteUpdate(next_stage=1, addr="127.0.0.1:1", epoch=98)
    assert servicer.route(older) is False  # arrived out of order
    assert (servicer.next_address, servicer.next_epoch) == (update.addr, 99)


def test_stage1_ignores_a_route_update(beating):
    run = beating()
    before = (run.servicer1.next_address, run.servicer1.next_epoch)
    update = protocol_pb2.RouteUpdate(next_stage=2, addr="127.0.0.1:1", epoch=5)
    assert run.servicer1.route(update) is False
    assert run.servicer0.route(protocol_pb2.RouteUpdate(next_stage=0, addr="x:1", epoch=5)) is False
    assert (run.servicer1.next_address, run.servicer1.next_epoch) == before


# -------------------------------------------------------------- the stream itself


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_stage_keeps_trying_until_the_gateway_is_up(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    port = free_port()
    run = System(tmp_path, 2)  # registered in process; its own gateway is not used here
    stop = threading.Event()
    heart = Heartbeat(f"127.0.0.1:{port}", run.servicer0, interval=FAST, retry=FAST)
    run.servicer0.own_epoch = run.registry.stages[0].epoch
    thread = heart.start(stop)
    try:
        wait_for(lambda: heart.streams >= 3, "a few failed attempts")
        server, _ = serve(run.gateway, f"127.0.0.1:{port}")
        try:
            wait_for(lambda: 0 in run.gateway.beats(), "a beat once the gateway is up")
        finally:
            stop.set()
            thread.join(WAIT)
            server.stop(0).wait()
    finally:
        stop.set()
        run.close()


def test_a_stage_that_registers_again_ends_the_old_stream(beating):
    run = beating()
    old = run.hearts[0]
    old_thread = run.threads[-1]
    old_epoch = run.servicer0.own_epoch
    wait_for(lambda: run.gateway.beats().get(0, {}).get("epoch") == old_epoch, "the old stream")
    run.join(0, run.servicer0)  # a new incarnation of stage 0
    old_thread.join(WAIT)
    assert not old_thread.is_alive() and old.superseded and old.streams == 1
    new_epoch = run.servicer0.own_epoch
    wait_for(lambda: run.gateway.beats()[0]["epoch"] == new_epoch, "beats from the new stream")


def test_a_registration_alone_ends_the_old_stream(beating):
    """The new process has registered but not opened its stream yet: the old stream must
    still end, so an old process that is only paused stops receiving frees and routes."""
    run = beating()
    old = run.hearts[1]
    old_thread = run.threads[0]
    epoch = run.servicer1.own_epoch
    wait_for(lambda: run.gateway.beats().get(1, {}).get("epoch") == epoch, "the old stream")
    start, end = run.layers[1]
    with grpc.insecure_channel(run.address) as channel:
        control.register(channel, 1, (start, end), "tiny", DIGEST.hex(), "127.0.0.1:9")
    old_thread.join(WAIT)
    assert not old_thread.is_alive() and old.superseded and old.streams == 1
    assert 1 not in run.gateway._sessions


def test_a_stream_from_an_old_epoch_is_refused(beating):
    run = beating()
    old_epoch = run.registry.stages[1].epoch
    run.join(1, run.servicer1)  # stage 1 registers again

    class Old:
        stage_index, own_epoch, inbox_depth, cached_requests = 1, old_epoch, 0, 0
        nacks = queue.Queue()

    heart = Heartbeat(run.address, Old(), interval=FAST, retry=FAST)
    stop = threading.Event()
    thread = heart.start(stop)
    thread.join(5.0)  # it returns once refused
    stop.set()
    thread.join(WAIT)
    assert heart.superseded and heart.streams == 1


def test_the_first_message_must_be_a_beat(beating):
    run = beating()
    nack = protocol_pb2.StageMsg(nack=protocol_pb2.Nack(stage=0, reason="GAP"))
    with grpc.insecure_channel(run.address) as channel:
        call = protocol_pb2_grpc.ControlStub(channel).Heartbeat(iter([nack]), timeout=WAIT)
        code, _ = code_of(call)
    assert code == grpc.StatusCode.INVALID_ARGUMENT


def test_stopping_closes_the_stream_at_the_gateway(beating):
    run = beating()
    wait_for(lambda: set(run.gateway._sessions) == {0, 1}, "both streams open")
    for stop in run.stops:
        stop.set()
    for thread in run.threads:
        thread.join(WAIT)
        assert not thread.is_alive()
    wait_for(lambda: not run.gateway._sessions, "the gateway to close both streams")


def test_beats_from_a_replaced_stream_are_not_recorded(beating):
    run = beating()
    wait_for(lambda: 1 in run.gateway._sessions, "stage 1's stream")
    session = run.gateway._sessions[1]
    run.gateway._sessions[1] = object()  # as if a newer stream had taken its place
    try:
        before = run.gateway.beats()[1]["age_s"]
        old = protocol_pb2.StageMsg(beat=protocol_pb2.Beat(stage=1, epoch=1, cached_requests=7))
        run.gateway._record(session, old)
        assert run.gateway.beats()[1]["cached_requests"] != 7
        assert run.gateway.beats()[1]["age_s"] >= before
    finally:
        run.gateway._sessions[1] = session


def test_a_prompt_with_the_heartbeat_running_equals_the_reference(beating, tmp_path):
    run = beating(k=3)
    expected = reference(tmp_path, 3, [int(w) for w in PROMPT.split()], 6)
    got = run.tokens(max_tokens=6)
    assert [(t.token, t.top_logit) for t in got] == list(zip(*expected, strict=True))


def test_a_worker_registers_then_keeps_its_heartbeat(tmp_path):
    """The path a serving worker takes after it starts: Register, then the stream."""
    tiny_reference(tmp_path, layers=LAYERS)
    digest = sha256_of(tmp_path / "model.safetensors")
    n = ModelConfig.from_file(tmp_path / "config.json").num_layers
    gateway = Gateway(
        Registry("tiny", ExpectedModel(bytes.fromhex(digest), n, 2)), None, (), 64, 2**20
    )
    server, port = serve(gateway, "127.0.0.1:0")
    stage_server, servicer, stage_port = serve_stage(
        StageWorker(WorkerConfig(tmp_path, 2, n)),
        "127.0.0.1:0",
        results_address=f"127.0.0.1:{port}",
    )
    stop = threading.Event()
    args = (f"127.0.0.1:{port}", 1, (2, n), "tiny", tmp_path, f"127.0.0.1:{stage_port}", servicer)
    thread = threading.Thread(target=join_gateway, args=(*args, stop), kwargs={"interval": FAST})
    thread.start()
    try:
        wait_for(lambda: 1 in gateway.beats(), "a beat after registering")
        assert gateway.beats()[1]["epoch"] == servicer.own_epoch == gateway.registry.stages[1].epoch
    finally:
        stop.set()
        thread.join(WAIT)
        stage_server.stop(0).wait()
        servicer.close()
        server.stop(0).wait()
    assert not thread.is_alive()


def test_the_default_pool_keeps_threads_for_both_heartbeats(beating):
    run = beating()
    assert run.gateway.max_running == 14  # 20 threads, 6 kept
    wait_for(lambda: set(run.gateway._sessions) == {0, 1}, "both heartbeat streams")


def test_requests_past_the_cap_are_refused_so_heartbeats_keep_a_thread(beating):
    run = beating(delay1=0.5)
    run.gateway.max_running = 1
    first = run.call(max_tokens=3)
    next(first)  # holds the only slot
    code, details = code_of(run.call(max_tokens=1))
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED and "try again" in details
    assert len(list(first)) == 2  # the first request is unaffected
    assert len(run.tokens(max_tokens=1)) == 1  # its slot is free again
