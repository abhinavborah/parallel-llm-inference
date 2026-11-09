"""Failure detection at the gateway (R6.3, R6.4).

The detector on a fake clock, then each path injected for real: a process that is killed or
suspended, and a slow step. The process is a small script that registers and keeps the real
heartbeat, so these need no model weights.
"""

import json
import logging
import queue
import socket as socket_module
import subprocess
import sys
import threading
import time

import grpc
import pytest
from test_gateway import DIGEST, HIDDEN, IDS, PROMPT, WAIT, Words, code_of, wait_for
from test_heartbeat import FAST, Beating, reference
from test_model import tiny_reference

from pllm import config, control, faults
from pllm.control import ExpectedModel, Registry
from pllm.detection import MAX_EVENTS, Detector
from pllm.gateway import Gateway, Session, serve
from pllm.model import ModelConfig
from pllm.models import sha256_of
from pllm.proto import protocol_pb2, protocol_pb2_grpc
from pllm.stage_service import serve as serve_stage
from pllm.supervisor import Supervisor
from pllm.worker import StageWorker, WorkerConfig, join_gateway

SECOND = 1_000_000_000
STAGE, EPOCH = 1, 7


class Clock:
    def __init__(self):
        self.ns = 1000

    def __call__(self):
        return self.ns

    def advance(self, seconds):
        self.ns += round(seconds * SECOND)


def fake(**kwargs):
    clock = Clock()
    return Detector(clock=clock, wall=lambda: 1234.5, **kwargs), clock


def detected(detector, **fields):
    return [
        event
        for event in detector.events()
        if event["event"] == "detected" and fields.items() <= event.items()
    ]


def test_a_registered_stage_that_never_beat_is_never_expired():
    detector, clock = fake()
    clock.advance(3600)
    detector.expire()
    assert detector.events() == []


def test_a_silent_stage_expires_at_the_timeout_and_not_before():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    beat_at = clock.ns
    clock.advance(1.999)
    detector.expire()
    assert detector.events() == [] and not detector.is_dead(EPOCH)
    clock.advance(0.001)
    detector.expire()
    assert detected(detector) == [
        {
            "event": "detected",
            "path": "heartbeat_timeout",
            "stage": STAGE,
            "epoch": EPOCH,
            "declared": True,
            "t_ns": clock.ns,
            "time": 1234.5,
            "last_beat_ns": beat_at,
            "since_last_beat_ns": 2 * SECOND,
            "rid": None,
            "step": None,
            "detail": "",
        }
    ]
    assert detector.is_dead(EPOCH)


def test_a_beat_keeps_a_stage_alive():
    detector, clock = fake(heartbeat_timeout=2.0)
    for _ in range(10):
        detector.beat(STAGE, EPOCH)
        clock.advance(1.5)
        detector.expire()
    assert detector.events() == []


def test_a_new_epoch_does_not_inherit_the_last_beat_of_the_old_one():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    detector.declare(STAGE, EPOCH, "stream_break")
    clock.advance(1.5)
    detector.beat(STAGE, EPOCH + 1)
    clock.advance(1.0)
    detector.expire()
    assert not detector.is_dead(EPOCH + 1)
    assert len(detected(detector)) == 1


def test_each_epoch_of_a_stage_has_its_own_last_beat():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    clock.advance(1.5)
    detector.beat(STAGE, EPOCH + 1)
    clock.advance(1.0)
    detector.expire()
    assert detector.is_dead(EPOCH) and not detector.is_dead(EPOCH + 1)


def test_a_dead_epoch_is_never_revived_by_a_late_beat():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    detector.declare(STAGE, EPOCH, "stream_break")
    assert detector.beat(STAGE, EPOCH) is False
    clock.advance(5)
    detector.expire()
    assert len(detected(detector)) == 1 and detector.is_dead(EPOCH)


def test_the_first_declaration_wins_from_many_threads():
    detector, _ = fake()
    detector.beat(STAGE, EPOCH)
    start = threading.Barrier(16)
    wins = []

    def declare(path):
        start.wait()
        wins.append(detector.declare(STAGE, EPOCH, path))

    threads = [threading.Thread(target=declare, args=(f"path{i}",)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(wins) == [False] * 15 + [True]
    assert len(detected(detector)) == 1


def test_a_step_deadline_declares_nothing_and_has_no_suspect_while_beats_are_fresh():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    clock.advance(1.0)  # exactly half: not yet suspect
    detector.step_deadline("r1", 4)
    record = detected(detector, path="step_deadline")[0]
    assert (record["rid"], record["step"], record["declared"]) == ("r1", 4, False)
    assert record["stage"] is None and record["epoch"] is None
    assert record["last_beat_ns"] is None and record["since_last_beat_ns"] is None
    assert not detector.is_dead(EPOCH)


def test_a_step_deadline_names_the_stage_with_the_oldest_beat_as_suspect():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(0, 3)
    clock.advance(0.5)
    detector.beat(1, 4)
    clock.advance(1.0)  # stage 0 last beat 1.5 s ago, stage 1 1.0 s ago
    detector.step_deadline("r1", 0)
    record = detected(detector, path="step_deadline")[0]
    assert (record["stage"], record["epoch"]) == (0, 3)
    assert record["since_last_beat_ns"] == 1.5 * SECOND
    assert not detector.is_dead(3)


def test_a_registration_reports_the_wait_since_the_declaration():
    detector, clock = fake()
    detector.declare(STAGE, EPOCH, "stream_break")
    clock.advance(3)
    detector.registered(STAGE, EPOCH, EPOCH + 1, clock.ns)
    assert detector.events()[-1] == {
        "event": "registered",
        "stage": STAGE,
        "old_epoch": EPOCH,
        "new_epoch": EPOCH + 1,
        "t_ns": clock.ns,
        "time": 1234.5,
        "waited_ns": 3 * SECOND,
        "route_queued_ns": clock.ns,
    }


def test_a_first_registration_has_no_wait_or_route():
    detector, _ = fake()
    detector.registered(STAGE, None, 1, None)
    record = detector.events()[-1]
    assert record["old_epoch"] is None and record["waited_ns"] is None
    assert record["route_queued_ns"] is None


def test_every_reported_time_comes_from_the_nanosecond_clock():
    detector, _ = fake()
    detector.beat(STAGE, EPOCH)
    detector.declare(STAGE, EPOCH, "stream_break")
    detector.registered(STAGE, EPOCH, EPOCH + 1, 5)
    for record in detector.events():
        for name, value in record.items():
            assert "mono" not in name
            if name.endswith("_ns"):
                assert value is None or isinstance(value, int)


def test_the_reporting_clock_is_the_gateway_perf_counter():
    assert Detector().now is time.perf_counter_ns


def test_events_are_json_bounded_and_logged_as_one_line(caplog):
    detector, _ = fake()
    with caplog.at_level(logging.WARNING, logger="pllm.detection"):
        detector.declare(STAGE, EPOCH, "stream_break", "UNAVAILABLE")
    assert json.loads(caplog.records[0].getMessage()) == detector.events()[0]
    assert "\n" not in caplog.records[0].getMessage()
    for epoch in range(MAX_EVENTS + 10):
        detector.step_deadline("r", epoch)
    assert len(detector.events()) == MAX_EVENTS


# -------------------------------------------------------------- real streams and processes

LAYERS = 4
K = 2
BEAT = 0.05

CHILD = """
import pathlib, queue, socket, sys, threading
import grpc
from pllm.heartbeat import Heartbeat
from pllm.proto import protocol_pb2, protocol_pb2_grpc

gateway, marker = sys.argv[1], pathlib.Path(sys.argv[2])
stage, start, end, port = map(int, sys.argv[3:7])
interval = float(sys.argv[7])


class Stand:
    stage_index, own_epoch, inbox_depth, cached_requests = stage, 0, 0, 0
    nacks = queue.Queue()

    def free(self, message):
        pass

    def route(self, message):
        pass


listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen()
servicer = Stand()
with grpc.insecure_channel(gateway) as channel:
    servicer.own_epoch = protocol_pb2_grpc.ControlStub(channel).Register(
        protocol_pb2.RegisterReq(
            stage=stage,
            layer_start=start,
            layer_end=end,
            model_id="tiny",
            weights_sha256=bytes(bytearray([1]) * 32),
            addr=f"127.0.0.1:{port}",
        ),
        timeout=30,
    ).epoch
heart = Heartbeat(gateway, servicer, interval=interval, retry=interval)
heart.run(threading.Event())
if heart.declared_dead:
    sys.exit(1)
marker.write_text("superseded" if heart.superseded else "ended")
threading.Event().wait()
"""


def free_port():
    with socket_module.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Plane:
    """A real gateway alone, with stage processes that register and beat but serve nothing."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.registry = Registry("tiny", ExpectedModel(DIGEST, LAYERS, K))
        self.gateway = Gateway(self.registry, Words(), (), HIDDEN, config.max_message_bytes(HIDDEN))
        self.server, port = serve(self.gateway, "127.0.0.1:0")
        self.address = f"127.0.0.1:{port}"
        self.detector = self.gateway.detector
        self.children = []
        self.script = tmp_path / "stage.py"
        self.script.write_text(CHILD)
        self.channel = grpc.insecure_channel(self.address)

    def command(self, stage, port):
        start, end = (0, K) if stage == 0 else (K, LAYERS)
        marker = self.tmp_path / f"marker{stage}"
        return [sys.executable, str(self.script), self.address, str(marker)] + [
            str(n) for n in (stage, start, end, port, BEAT)
        ]

    def child(self, stage):
        """Start a stage process and wait for its first beat. Returns it and its epoch."""
        proc = subprocess.Popen(self.command(stage, free_port()))
        self.children.append(proc)
        before = self.registry.stages.get(stage)
        wait_for(
            lambda: self.registry.stages.get(stage) is not before and stage in self.beating(),
            "the stage's first beat",
        )
        return proc, self.registry.stages[stage].epoch

    def beating(self):
        return {
            stage
            for stage, beat in self.gateway.beats().items()
            if beat["epoch"] == getattr(self.registry.stages.get(stage), "epoch", None)
        }

    def register(self, stage, port=1):
        """A replacement started by the test: it registers only."""
        layers = (0, K) if stage == 0 else (K, LAYERS)
        return control.register(
            self.channel, stage, layers, "tiny", DIGEST.hex(), f"127.0.0.1:{port}"
        )

    def stream(self, stage, epoch):
        """Open a heartbeat stream from this thread: one Beat, then held open."""
        release = threading.Event()

        def outgoing():
            yield protocol_pb2.StageMsg(beat=protocol_pb2.Beat(stage=stage, epoch=epoch))
            release.wait(WAIT)

        call = protocol_pb2_grpc.ControlStub(self.channel).Heartbeat(outgoing())
        self.streams = [*getattr(self, "streams", []), (call, release)]
        return call

    def detection(self, **fields):
        """The one detection matching fields, once it exists."""
        wait_for(lambda: detected(self.detector, **fields), f"a detection of {fields}")
        return detected(self.detector, **fields)[0]

    def close(self):
        for call, release in getattr(self, "streams", []):
            release.set()
            call.cancel()
        for proc in self.children:
            if proc.poll() is None:
                faults.kill(proc.pid)
            proc.wait(WAIT)
        self.channel.close()
        self.gateway.close()
        self.server.stop(0).wait()


@pytest.fixture
def plane(tmp_path):
    made = Plane(tmp_path)
    yield made
    made.close()


@pytest.fixture
def beating(tmp_path):
    made = []

    def make(**kwargs):
        tiny_reference(tmp_path, layers=LAYERS)
        made.append(Beating(tmp_path, K, interval=BEAT, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


@pytest.mark.parametrize("stage", [0, 1])
def test_a_killed_stage_is_detected_by_its_stream_break(plane, stage):
    proc, epoch = plane.child(stage)
    before = plane.detector.now()
    faults.kill(proc.pid)
    proc.wait(WAIT)
    record = plane.detection(path="stream_break", epoch=epoch)
    assert record["stage"] == stage and record["declared"] is True
    assert record["t_ns"] >= before and record["t_ns"] >= record["last_beat_ns"]
    assert record["since_last_beat_ns"] == record["t_ns"] - record["last_beat_ns"]
    assert record["detail"] in ("broken", "closed")  # a kill can read as a clean end
    assert plane.detector.is_dead(epoch)
    time.sleep(0.2)
    assert len(detected(plane.detector, epoch=epoch)) == 1


def test_a_suspended_stage_is_detected_by_its_heartbeat_timeout(plane):
    proc, epoch = plane.child(1)
    faults.suspend(proc.pid)  # its sockets stay open, so no stream breaks
    time.sleep(1.0)
    assert detected(plane.detector, epoch=epoch) == []
    record = plane.detection(path="heartbeat_timeout", epoch=epoch)
    assert record["stage"] == 1 and record["declared"] is True
    assert record["since_last_beat_ns"] >= plane.detector.heartbeat_timeout_ns
    assert detected(plane.detector, path="stream_break") == []


def test_a_slow_step_is_recorded_without_declaring_the_stage_and_the_request_completes(beating):
    run = beating(delay1=0.5)
    run.gateway.detector.step_timeout_ns = round(0.2 * SECOND)
    tokens = run.tokens(max_tokens=2)
    assert len(tokens) == 2
    records = detected(run.gateway.detector, path="step_deadline")
    assert sorted(r["step"] for r in records) == [0, 1]  # once per step
    assert all(r["stage"] is None and r["declared"] is False for r in records)
    assert all(r["rid"] == tokens[0].rid for r in records)
    assert detected(run.gateway.detector, declared=True) == []


def test_a_step_held_by_a_suspended_stage_names_it_then_its_timeout_declares_it(plane):
    plane.detector.heartbeat_timeout_ns = 4 * SECOND
    plane.detector.step_timeout_ns = 3 * SECOND
    proc, epoch = plane.child(0)
    plane.register(1)  # no heartbeat, so never a suspect
    faults.suspend(proc.pid)  # still accepts the connection
    stub = protocol_pb2_grpc.FrontendStub(plane.channel)
    req = protocol_pb2.GenerateReq(prompt=PROMPT, max_tokens=1, timeout_ms=6000)
    code, _ = code_of(stub.Generate(req, timeout=WAIT))
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED
    step = plane.detection(path="step_deadline")
    assert (step["stage"], step["epoch"], step["declared"]) == (0, epoch, False)
    declared = plane.detection(path="heartbeat_timeout", epoch=epoch)
    assert declared["t_ns"] > step["t_ns"]


def stop_stage(run, stage):
    run.stops[0 if stage == 1 else 1].set()  # stages join in the order 1, 0
    server, servicer = (run.server0, run.servicer0) if stage == 0 else (run.server1, run.servicer1)
    server.stop(0).wait()
    servicer.close()


def start_replacement(run, stage):
    target = (
        {"results_address": run.address} if stage else {"next_address": f"127.0.0.1:{run.ports[1]}"}
    )
    start, end = run.layers[stage]
    server, servicer, port = serve_stage(
        StageWorker(WorkerConfig(run.model_dir, start, end)), "127.0.0.1:0", **target
    )
    run.extra.append((server, servicer))
    return servicer, run.join(stage, servicer, port)


@pytest.mark.parametrize("stage", [0, 1])
def test_a_dead_epoch_waits_for_a_new_registration_and_a_route_update(beating, tmp_path, stage):
    run = beating()
    detector = run.gateway.detector
    expected = reference(tmp_path, K, IDS, 3)
    old = run.registry.stages[stage].epoch
    stop_stage(run, stage)
    wait_for(lambda: detector.is_dead(old), "the epoch to be declared dead")
    code, details = code_of(run.call(max_tokens=3))
    assert code == grpc.StatusCode.UNAVAILABLE and f"epoch {old} is dead" in details
    time.sleep(0.2)  # it waits: nothing else happens without a Register
    assert run.registry.stages[stage].epoch == old
    assert [e["event"] for e in detector.events()] == ["registered"] * 2 + ["detected"]

    servicer, new = start_replacement(run, stage)
    assert new > old
    record = detector.events()[-1]
    assert record["event"] == "registered"
    assert (record["stage"], record["old_epoch"], record["new_epoch"]) == (stage, old, new)
    assert record["waited_ns"] > 0.2 * SECOND
    assert isinstance(record["route_queued_ns"], int)
    assert len(detected(detector, epoch=old)) == 1  # no second record from the Register
    run.wait_routes(**{f"servicer{stage}": servicer})

    got = run.tokens(max_tokens=3)
    assert [(t.token, t.top_logit) for t in got] == list(zip(*expected, strict=True))


def test_a_register_before_detection_writes_one_record_and_a_later_break_none(plane):
    proc, epoch = plane.child(1)
    new = plane.register(1)
    record = plane.detection(epoch=epoch)
    assert record["path"] == "replaced" and record["declared"] is True
    faults.kill(proc.pid)
    proc.wait(WAIT)
    time.sleep(0.2)
    assert len(detected(plane.detector, epoch=epoch)) == 1
    assert not plane.detector.is_dead(new)


def test_a_stream_replaced_by_a_new_stream_of_the_same_epoch_declares_nothing(plane):
    epoch = plane.register(1)
    first = plane.stream(1, epoch)
    wait_for(lambda: 1 in plane.gateway._sessions, "the first stream")
    session = plane.gateway._sessions[1]
    plane.stream(1, epoch)
    wait_for(lambda: plane.gateway._sessions[1] is not session, "the second stream")
    code, _ = code_of(first)
    assert code == grpc.StatusCode.FAILED_PRECONDITION
    wait_for(session.closed.is_set, "the first stream to close")
    time.sleep(0.2)
    assert detected(plane.detector) == [] and not plane.detector.is_dead(epoch)


def test_a_break_is_declared_even_when_the_stream_is_already_removed(plane):
    """On a kill the generator's cleanup can run before the reader sees the break."""
    epoch = plane.register(1)
    session = Session(1, epoch)
    assert 1 not in plane.gateway._sessions
    plane.gateway._stream_ended(session, "broken")
    assert plane.detector.is_dead(epoch)


def test_close_stops_the_detector_thread(plane):
    watcher = plane.gateway._watcher
    assert watcher.is_alive()
    plane.gateway.close()
    assert not watcher.is_alive()


# -------------------------------------------------------------- a stage declared dead that lives


def test_a_replacement_takes_precedence_over_the_declared_death_of_a_stream(plane):
    session = Session(1, 7)
    assert plane.gateway._refusal(session) is None
    session.fenced.set()
    assert plane.gateway._refusal(session)[0] == grpc.StatusCode.ABORTED
    session.superseded.set()  # a late Beat and a Register landed together
    assert plane.gateway._refusal(session)[0] == grpc.StatusCode.FAILED_PRECONDITION


def test_a_stage_resumed_with_no_replacement_is_told_and_exits(plane):
    proc, epoch = plane.child(1)
    faults.suspend(proc.pid)
    plane.detection(path="heartbeat_timeout", epoch=epoch)
    assert proc.poll() is None  # not told at the declaration: it is suspended
    faults.resume(proc.pid)
    assert proc.wait(WAIT) == 1
    assert plane.registry.stages[1].epoch == epoch  # the epoch is not revived
    assert len(detected(plane.detector, epoch=epoch)) == 1


def test_a_resumed_stage_is_replaced_by_its_supervisor_with_no_manual_step(plane, tmp_path):
    port = free_port()
    supervisor = Supervisor(
        plane.command(1, port),
        ready_address=("127.0.0.1", port),
        poll_seconds=0.02,
        log_path=tmp_path / "events.jsonl",
    )
    supervisor.start()
    try:
        wait_for(lambda: 1 in plane.beating(), "the first beat")
        epoch = plane.registry.stages[1].epoch
        supervisor.apply("suspend")
        plane.detection(path="heartbeat_timeout", epoch=epoch)
        supervisor.apply("resume")
        wait_for(lambda: plane.registry.stages[1].epoch > epoch, "the new registration")
        new = plane.registry.stages[1].epoch
        wait_for(lambda: plane.gateway.beats()[1]["epoch"] == new, "a beat from the new epoch")
        wait_for(lambda: plane.detector.events()[-1]["event"] == "registered", "the record")
        record = plane.detector.events()[-1]
        assert (record["old_epoch"], record["new_epoch"]) == (epoch, new)
        assert not plane.detector.is_dead(new)
    finally:
        supervisor.stop()


def test_a_stage_resumed_after_a_replacement_stops_beating_and_keeps_running(plane, tmp_path):
    proc, epoch = plane.child(1)
    faults.suspend(proc.pid)
    plane.detection(path="heartbeat_timeout", epoch=epoch)
    time.sleep(0.3)  # long enough for a status sent at the declaration to be on the wire
    new = plane.register(1)
    faults.resume(proc.pid)
    marker = tmp_path / "marker1"
    wait_for(marker.exists, "the old process to read the refusal")
    assert marker.read_text() == "superseded"
    assert proc.poll() is None  # it keeps serving, for the fencing story to reject
    beat = plane.gateway.beats()[1]
    assert beat["epoch"] == epoch and beat["age_s"] >= 2.0  # its late beats did not count
    assert plane.registry.stages[1].epoch == new


def test_a_stream_opened_by_a_dead_epoch_is_refused_at_its_first_beat(plane):
    epoch = plane.register(0)
    plane.detector.declare(0, epoch, "stream_break", "broken")
    code, details = code_of(plane.stream(0, epoch))
    assert code == grpc.StatusCode.ABORTED and "declared dead" in details
    assert plane.gateway.beats() == {} and plane.detector.is_dead(epoch)


def test_a_worker_whose_epoch_is_declared_dead_ends_with_an_error(tmp_path):
    """The path a serving worker takes: join_gateway raises and main exits with code 1."""
    tiny_reference(tmp_path, layers=LAYERS)
    digest = bytes.fromhex(sha256_of(tmp_path / "model.safetensors"))
    n = ModelConfig.from_file(tmp_path / "config.json").num_layers
    gateway = Gateway(Registry("tiny", ExpectedModel(digest, n, K)), None, (), HIDDEN, 2**20)
    server, port = serve(gateway, "127.0.0.1:0")

    class Stand:
        stage_index, own_epoch, inbox_depth, cached_requests = 1, 0, 0, 0
        nacks = queue.Queue()

    raised = []

    def join():
        try:
            join_gateway(
                f"127.0.0.1:{port}",
                1,
                (K, n),
                "tiny",
                tmp_path,
                "127.0.0.1:1",
                Stand(),
                threading.Event(),
                interval=FAST,
            )
        except RuntimeError as err:
            raised.append(err)

    thread = threading.Thread(target=join, daemon=True)
    thread.start()
    try:
        wait_for(lambda: 1 in gateway.beats(), "a beat after registering")
        gateway.detector.declare(1, gateway.registry.stages[1].epoch, "stream_break")
        thread.join(WAIT)
        assert not thread.is_alive() and "declared stage 1" in str(raised[0])
    finally:
        gateway.close()
        server.stop(0).wait()
