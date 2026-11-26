"""Recovery time in four parts (R6.12): detection, respawn, weight load and replay prefill,
with the tokens recomputed. A new stage reports its weight load time with Register."""

import subprocess
import sys

import grpc
import pytest
from test_detection import DIGEST, Plane, start_replacement
from test_fail_cleanly import Stages
from test_gateway import IDS, WAIT, code_of, wait_for
from test_heartbeat import Beating, reference
from test_model import tiny_reference
from test_replay import BEAT, CHILD, LAYERS, K, replace, take

from pllm import control, faults
from pllm import gateway as gateway_module
from pllm.control import ExpectedModel
from pllm.gateway import recovery_parts
from pllm.models import sha256_of
from pllm.proto import protocol_pb2, protocol_pb2_grpc


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


def registered(plane, epoch):
    return [e for e in plane.detector.events() if e.get("new_epoch") == epoch][0]


# ------------------------------------------------------------------ the weight load time


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, None),
        ((), None),
        (((control.LOAD_KEY, "123456789"),), 123456789),
        (((control.LOAD_KEY, b"42"),), 42),
        (((control.LOAD_KEY, "-5"),), None),
        (((control.LOAD_KEY, "1.5"),), None),
        (((control.LOAD_KEY, ""),), None),
        (((control.LOAD_KEY, "٣"),), None),  # a digit, but not an ASCII one
        ((("other", "7"),), None),
    ],
)
def test_the_load_time_is_a_plain_integer_or_none(metadata, expected):
    assert control.load_ns_of(metadata) == expected


def test_a_stage_reports_its_load_time_with_register(plane):
    epoch = control.register(
        plane.channel, 1, (K, LAYERS), "tiny", DIGEST.hex(), "127.0.0.1:1", load_ns=987654321
    )
    assert registered(plane, epoch)["load_ns"] == 987654321


@pytest.mark.parametrize("value", [None, "soon", "-1"])
def test_a_register_with_no_or_a_bad_load_time_still_registers(plane, value):
    """An older worker sends no load time; a bad value is dropped, never logged."""
    req = protocol_pb2.RegisterReq(
        stage=1, layer_start=K, layer_end=LAYERS, model_id="tiny",
        weights_sha256=DIGEST, addr="127.0.0.1:1",
    )  # fmt: skip
    metadata = None if value is None else ((control.LOAD_KEY, value),)
    stub = protocol_pb2_grpc.ControlStub(plane.channel)
    epoch = stub.Register(req, timeout=10, metadata=metadata).epoch
    assert registered(plane, epoch)["load_ns"] is None


# ------------------------------------------------------------------ the split itself

S = 1_000_000_000


def death(t_ns, last_beat_ns, path="stream_break"):
    return {
        "stage": 1,
        "epoch": 3,
        "path": path,
        "t_ns": t_ns,
        "last_beat_ns": last_beat_ns,
        "since_last_beat_ns": t_ns - last_beat_ns,
    }


def registration(t_ns, load_ns):
    return {"stage": 1, "old_epoch": 3, "new_epoch": 4, "t_ns": t_ns, "load_ns": load_ns}


def test_a_kill_splits_into_detection_respawn_and_load():
    """Last beat at 10 s, declared at 10.2 s, registered at 12 s after a 1.5 s load."""
    parts = recovery_parts(death(10_200_000_000, 10 * S), registration(12 * S, 1_500_000_000))
    assert parts == {
        "stage": 1,
        "old_epoch": 3,
        "new_epoch": 4,
        "path": "stream_break",
        "detection_ns": 200_000_000,
        "respawn_ns": 300_000_000,
        "weight_load_ns": 1_500_000_000,
        "overlap": False,
    }


def test_a_restart_that_overlaps_detection_has_no_negative_respawn():
    """The supervisor restarted the worker while the gateway still waited for a heartbeat
    timeout: only 0.5 s from declaration to registration, after a 0.9 s load."""
    parts = recovery_parts(
        death(12 * S, 10 * S, "heartbeat_timeout"), registration(12_500_000_000, 900_000_000)
    )
    assert parts["respawn_ns"] == 0 and parts["overlap"] is True
    assert parts["detection_ns"] == 2 * S and parts["weight_load_ns"] == 900_000_000


def test_a_replaced_stage_is_all_overlap():
    """The new incarnation registered first; its Register made the declaration."""
    parts = recovery_parts(
        death(12 * S, 10 * S, "replaced"), registration(12 * S + 1000, 800_000_000)
    )
    assert parts["path"] == "replaced" and parts["respawn_ns"] == 0 and parts["overlap"] is True


def test_with_no_load_time_or_no_registration_the_split_is_none():
    no_load = recovery_parts(death(11 * S, 10 * S), registration(13 * S, None))
    assert no_load["new_epoch"] == 4 and no_load["detection_ns"] == S
    assert no_load["respawn_ns"] is None and no_load["weight_load_ns"] is None
    assert no_load["overlap"] is None
    waiting = recovery_parts(death(11 * S, 10 * S), None)
    assert waiting["new_epoch"] is None and waiting["respawn_ns"] is None
    assert recovery_parts(None, None)["stage"] is None


# ------------------------------------------------------------------ the recovery line


def recoveries(run):
    """Capture every metrics line of type recovery the gateway writes from now on."""
    lines = []
    real = run.gateway.recorder.event

    def event(record):
        if record["event"] == "recovery":
            lines.append(record)
        real(record)

    run.gateway.recorder.event = event
    return lines


def replayed_sizes(run):
    """The tokens each replay starts with, as the gateway's _recover sees them."""
    sizes = {}
    real = run.gateway._recover

    def recover(request, attempt, context):
        sizes[request.rid] = len(request.tokens)
        return real(request, attempt, context)

    run.gateway._recover = recover
    return sizes


@pytest.mark.parametrize("stage", [0, 1])
def test_a_killed_stage_process_logs_its_recovery_in_four_parts(beating, tmp_path, stage):
    """A real stage process is killed mid-decode and a new process registers with its load
    time: the request's recovery line has every part and the tokens it recomputed."""
    run = beating(order=(1 - stage,), delay1=0.05)
    neighbour = "" if stage == 1 else f"127.0.0.1:{run.ports[1]}"
    args = [str(tmp_path), str(stage), str(K), str(LAYERS), run.address, neighbour, str(BEAT)]
    real = bytes.fromhex(sha256_of(tmp_path / "model.safetensors"))
    run.registry.expected = ExpectedModel(real, LAYERS, K)
    lines, sizes = recoveries(run), replayed_sizes(run)

    def spawn():
        return subprocess.Popen([sys.executable, "-c", CHILD, *args])

    children = [spawn()]
    try:
        first = run.registry.stages
        wait_for(lambda: stage in first and stage in run.gateway.beats(), "the child's beat")
        if stage == 1:
            wait_for(lambda: run.servicer0.next_epoch == first[1].epoch, "the route update")
        else:
            wait_for(lambda: run.servicer1.upstream_epoch == first[0].epoch, "the route update")
        stream = run.call(max_tokens=16)
        tokens = take(stream, 5)
        old = run.registry.stages[stage].epoch
        faults.kill(children[0].pid)
        children[0].wait(WAIT)
        wait_for(lambda: run.gateway.detector.is_dead(old), "the stage to be declared dead")
        children.append(spawn())
        tokens += list(stream)
    finally:
        for child in children:
            if child.poll() is None:
                faults.kill(child.pid)
            child.wait(WAIT)
    assert [t.token for t in tokens] == reference(tmp_path, K, IDS, 16)[0]
    [line] = lines
    rid = tokens[0].rid
    assert (line["rid"], line["recovered"], line["outcome"]) == (rid, True, None)
    assert (line["stage"], line["old_epoch"]) == (stage, old)
    assert line["new_epoch"] == run.registry.stages[stage].epoch
    for key in ("detection_ns", "weight_load_ns", "replay_prefill_ns", "total_ns"):
        assert line[key] > 0, key
    assert line["respawn_ns"] >= 0 and isinstance(line["overlap"], bool)
    # both are on the gateway clock and the prefill comes after the declaration
    assert line["total_ns"] >= line["detection_ns"] + line["replay_prefill_ns"]
    assert line["recomputed_tokens"] == sizes[rid] >= 5
    assert line["divergences"] == 0


def test_each_request_bumped_by_one_death_logs_its_own_recovery(beating, tmp_path):
    """An idle request holding KV recovers too (R6.5): one line per request, the same death.
    Stages in this process report no load time, so the split stops at detection."""
    run = beating(credits=1, delay1=0.05)
    lines, sizes = recoveries(run), replayed_sizes(run)
    streams = [run.call(max_tokens=12), run.call(max_tokens=12)]
    got = [take(stream, 3) for stream in streams]
    old = replace(run, 1)
    servicer, _ = start_replacement(run, 1)
    run.wait_routes(servicer1=servicer)
    for stream, tokens in zip(streams, got, strict=True):
        tokens += list(stream)
    assert sorted(line["rid"] for line in lines) == sorted(t[0].rid for t in got)
    for line in lines:
        assert line["recovered"] and (line["stage"], line["old_epoch"]) == (1, old)
        assert line["new_epoch"] == run.registry.stages[1].epoch
        assert line["detection_ns"] == lines[0]["detection_ns"] > 0
        assert line["weight_load_ns"] is None and line["respawn_ns"] is None
        assert line["replay_prefill_ns"] > 0 and line["total_ns"] > 0
        assert line["recomputed_tokens"] == sizes[line["rid"]]


def test_a_recovery_that_never_gets_a_stage_is_logged_as_not_recovered(beating):
    run = beating(delay1=0.05)
    lines = recoveries(run)
    stream = run.call(max_tokens=50, timeout_ms=1500)
    take(stream, 2)
    old = replace(run, 1)
    code, _ = code_of(stream)
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED
    wait_for(lambda: lines, "the recovery line")
    [line] = lines
    assert (line["recovered"], line["outcome"], line["old_epoch"]) == (False, "failed", old)
    assert line["new_epoch"] is None and line["total_ns"] is None
    assert line["replay_prefill_ns"] is None and line["recomputed_tokens"] == 0


def test_the_death_a_request_gives_up_at_is_logged_as_not_recovered(beating):
    """At MAX_DEATHS the request fails before any replay starts; that death still has its
    line, after one recovered line per earlier death."""
    run = beating(delay1=0.05)
    lines = recoveries(run)
    stages = Stages(run)
    stream = run.call(max_tokens=400, timeout_ms=60_000)
    next(stream)
    epochs = []
    for _ in range(gateway_module.MAX_DEATHS - 1):
        epochs.append(run.registry.stages[1].epoch)
        stages.kill_and_replace(1)
        next(stream)
    epochs.append(run.registry.stages[1].epoch)
    stages.kill_and_replace(1)
    code, _ = code_of(stream)
    assert code == grpc.StatusCode.UNAVAILABLE
    wait_for(lambda: len(lines) == gateway_module.MAX_DEATHS, "a line per death")
    assert [line["old_epoch"] for line in lines] == epochs
    assert [line["recovered"] for line in lines] == [True] * (len(epochs) - 1) + [False]
    assert lines[-1]["outcome"] == "failed" and lines[-1]["total_ns"] is None
