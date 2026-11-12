"""Gateway metrics: the log, one clock, bytes per hop and the Stage 0 report (R7.1, R7.2).

The integration tests run a real gateway and two real stage servers on loopback.
"""

import json
import signal
import subprocess
import sys
import threading
import time
from concurrent import futures

import grpc
import pytest
from test_gateway import HIDDEN, IDS, LAYERS, PROMPT, WAIT, System, code_of, wait_for
from test_model import tiny_reference

from pllm import gateway as gateway_module
from pllm import metrics
from pllm.metrics import Counters, Recorder, parse_counters
from pllm.proto import protocol_pb2 as pb
from pllm.proto import protocol_pb2_grpc
from pllm.reference import generate
from pllm.split import InProcessSplit

STOP_LIMIT = 30  # seconds; well inside the 40 s a launcher allows before it kills


def activations(step=1, replay=False, length=1):
    return pb.StepRequest(
        rid="r",
        attempt=1,
        step=step,
        length=length,
        replay=replay,
        activations=pb.Tensor(dtype="float32", shape=[length, 4], data=b"\0" * (16 * length)),
    )


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def of_type(rows, kind, **match):
    return [r for r in rows if r["type"] == kind and all(r[k] == v for k, v in match.items())]


def end_run(run):
    run.gateway.close()
    run.gateway.recorder.finish()


@pytest.fixture
def metered(tmp_path):
    """A running system with a metrics file; returns the system and the file's path."""
    made = []
    tiny_reference(tmp_path, layers=LAYERS)

    def make(k=2, clock=None, **kwargs):
        run = System(tmp_path, k, **kwargs)
        path = tmp_path / f"run{len(made)}.jsonl"
        run.gateway.recorder = Recorder(path, {"model": "tiny"}, clock=clock)
        made.append(run)
        return run, path

    yield make
    for run in made:
        run.gateway.recorder.finish()
        run.close()


# -------------------------------------------------------------- lines, phases, sizes


def test_each_call_is_one_valid_line_even_from_many_threads(tmp_path):
    path = tmp_path / "m.jsonl"
    recorder = Recorder(path)

    def write():
        for _ in range(1000):
            recorder.hop(metrics.STAGE1_GATEWAY, pb.StepResult(rid="r", attempt=1, step=1))

    threads = [threading.Thread(target=write) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    recorder.finish()
    rows = lines(path)
    assert len(of_type(rows, "hop")) == 8000
    assert of_type(rows, "bytes")[0]["messages"] == 8000


@pytest.mark.parametrize(
    "message, phase",
    [
        (pb.StepRequest(step=0, length=6), "prefill"),
        (pb.StepRequest(step=0, length=1), "prefill"),  # a one-token prompt
        (pb.StepRequest(step=3, start_pos=8, length=1), "decode"),
        (pb.StepRequest(step=0, length=6, replay=True), "replay"),
        (pb.StepRequest(step=2, start_pos=8, length=1, replay=True), "replay"),
        (pb.StepResult(step=0), "prefill"),
        (pb.StepResult(step=1), "decode"),
    ],
)
def test_the_phase_comes_from_the_step_and_the_replay_flag(message, phase):
    assert metrics.phase_of(message) == phase


@pytest.mark.parametrize(
    "message, kind, payload, positions",
    [
        (pb.StepRequest(step=0, length=2, tokens=pb.TokenIds(ids=[300, 7])), "tokens", 0, 2),
        (activations(length=3), "activations", 48, 3),
    ],
)
def test_sizes_are_the_serialized_sizes(message, kind, payload, positions):
    assert metrics.measure(message)[0] == f"StepRequest.{kind}"
    assert metrics.measure(message)[2:] == (
        len(message.SerializeToString()),
        payload,
        positions,
    )


def test_a_result_has_its_serialized_size_and_no_payload():
    result = pb.StepResult(rid="r", attempt=1, step=2, token=9, top_logit=1.5)
    assert metrics.measure(result) == (
        metrics.RESULT,
        "decode",
        len(result.SerializeToString()),
        0,
        0,
    )


def good(*extra):
    return json.dumps([[metrics.ACTIVATIONS, "decode", 1, 2, 3, 4, *extra]])


def test_a_good_report_is_read():
    assert parse_counters(((metrics.TRAILER_KEY, good()),)) == json.loads(good())
    assert parse_counters(((metrics.TRAILER_KEY, "[]"),)) == []


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        (),
        (("other", good()),),
        ((metrics.TRAILER_KEY, "not json"),),
        ((metrics.TRAILER_KEY, "{}"),),
        ((metrics.TRAILER_KEY, good(1234567)),),  # a seventh value: a smuggled timestamp
        ((metrics.TRAILER_KEY, good().replace("decode", "t_ns")),),
        ((metrics.TRAILER_KEY, good().replace(metrics.ACTIVATIONS, "time")),),
        ((metrics.TRAILER_KEY, good().replace("1, 2", "-1, 2")),),
        ((metrics.TRAILER_KEY, good().replace("1, 2", "1.5, 2")),),
        ((metrics.TRAILER_KEY, good().replace("1, 2", "true, 2")),),
        ((metrics.TRAILER_KEY, json.dumps(json.loads(good()) * 2)),),  # one phase twice
        ((metrics.TRAILER_KEY, json.dumps(["x"])),),
        ((metrics.TRAILER_KEY, good()), (metrics.TRAILER_KEY, good())),
    ],
)
def test_a_bad_report_is_refused(metadata):
    assert parse_counters(metadata) is None


def test_a_recorder_with_no_file_keeps_nothing():
    recorder = Recorder()
    recorder.hop(metrics.STAGE1_GATEWAY, pb.StepResult(step=1))
    recorder.link_ended(None)
    recorder.finish()
    assert recorder.links == 0 and not recorder._counts[metrics.STAGE1_GATEWAY].buckets


def test_a_second_run_cannot_write_into_an_earlier_file(tmp_path):
    (tmp_path / "m.jsonl").write_text("earlier\n")
    with pytest.raises(FileExistsError):
        Recorder(tmp_path / "m.jsonl")


def test_lines_after_the_end_of_the_run_are_ignored(tmp_path):
    path = tmp_path / "m.jsonl"
    recorder = Recorder(path)
    recorder.finish()
    recorder.finish()
    recorder.request("failed", None, rid="late")
    recorder.hop(metrics.STAGE1_GATEWAY, pb.StepResult(step=1))
    assert lines(path)[-1]["type"] == "run_end" and not of_type(lines(path), "request")


def test_counters_add_up_by_message_type_and_phase():
    counters = Counters()
    counters.count(activations(step=0, length=2))
    counters.count(activations(step=1))
    counters.count(activations(step=2))
    rows = {row[1]: row for row in counters.rows()}
    assert rows["prefill"][2:] == [1, activations(step=0, length=2).ByteSize(), 32, 2]
    assert rows["decode"][2] == 2 and rows["decode"][4:] == [32, 2]


# -------------------------------------------------------------- one clock, whole system


def test_every_time_in_the_log_comes_from_the_one_clock(metered):
    given = []

    def clock():
        given.append(1000 + 7 * len(given))
        return given[-1]

    run, path = metered(clock=clock)
    run.tokens(max_tokens=3)
    end_run(run)
    stamps = [v for row in lines(path) for k, v in row.items() if k.endswith("_ns")]
    assert len(stamps) > 8 and set(stamps) <= set(given)


def test_one_line_per_request_and_per_hop_message_and_output_unchanged(tmp_path, metered):
    run, path = metered()
    tokens = run.tokens(max_tokens=4)
    end_run(run)
    assert ([t.token for t in tokens], [t.top_logit for t in tokens]) == generate(
        InProcessSplit.load(tmp_path, 2), IDS, 4
    )
    rows = lines(path)
    assert rows[0]["type"] == "run" and rows[0]["clock"] == "perf_counter_ns"
    assert rows[-1]["type"] == "run_end"
    request = of_type(rows, "request")
    assert len(request) == 1
    r = request[0]
    assert (r["outcome"], r["code"]) == ("max_tokens", None)
    assert (r["prompt_tokens"], r["max_tokens"], r["output_tokens"]) == (len(IDS), 4, 4)
    assert r["arrived_ns"] <= r["first_token_ns"] <= r["last_token_ns"] <= r["t_ns"]
    for hop, msg in (
        (metrics.GATEWAY_STAGE0, "StepRequest.tokens"),
        (metrics.STAGE1_GATEWAY, metrics.RESULT),
    ):
        sent = of_type(rows, "hop", hop=hop)
        assert [s["step"] for s in sent] == [0, 1, 2, 3]
        assert [s["phase"] for s in sent] == ["prefill", "decode", "decode", "decode"]
        assert all(s["msg"] == msg and s["rid"] == r["rid"] and s["accepted"] for s in sent)
    assert PROMPT not in json.dumps(rows)


def test_each_step_is_sent_before_its_result_comes_back(metered):
    run, path = metered()
    run.tokens(max_tokens=3)
    end_run(run)
    rows = lines(path)
    sent = of_type(rows, "hop", hop=metrics.GATEWAY_STAGE0)
    back = of_type(rows, "hop", hop=metrics.STAGE1_GATEWAY)
    assert all(s["t_ns"] < b["t_ns"] for s, b in zip(sent, back, strict=True))


def test_a_clean_run_totals_every_hop_prefill_apart_from_decode(metered):
    run, path = metered()
    run.tokens(max_tokens=4)
    end_run(run)
    rows = lines(path)
    for hop in (metrics.GATEWAY_STAGE0, metrics.STAGE1_GATEWAY):
        hops = of_type(rows, "hop", hop=hop)
        for phase in ("prefill", "decode"):
            total = of_type(rows, "bytes", hop=hop, phase=phase)[0]
            assert total["source"] == "gateway"
            assert total["bytes"] == sum(h["bytes"] for h in hops if h["phase"] == phase)
            assert total["messages"] == sum(h["phase"] == phase for h in hops)
    prefill = of_type(rows, "bytes", hop=metrics.STAGE0_STAGE1, phase="prefill")[0]
    decode = of_type(rows, "bytes", hop=metrics.STAGE0_STAGE1, phase="decode")[0]
    assert (prefill["source"], prefill["msg"]) == ("stage0", metrics.ACTIVATIONS)
    assert (prefill["messages"], prefill["payload_bytes"], prefill["positions"]) == (
        1,
        HIDDEN * 4 * len(IDS),
        len(IDS),
    )
    assert (decode["messages"], decode["payload_bytes"], decode["positions"]) == (
        3,
        HIDDEN * 4 * 3,
        3,
    )
    assert prefill["bytes"] > prefill["payload_bytes"]  # the protobuf fields around it
    end = of_type(rows, "run_end")[0]
    assert (end["links"], end["links_unreported"], end["stage0_stage1_complete"]) == (1, 0, True)


def test_a_run_with_no_requests_is_complete_and_empty(metered):
    run, path = metered()
    end_run(run)
    rows = lines(path)
    assert not of_type(rows, "bytes") and of_type(rows, "run_end")[0]["links"] == 0


def test_close_twice_counts_each_stream_once_and_the_gateway_stays_usable(tmp_path, metered):
    run, path = metered()
    run.tokens(max_tokens=2)
    run.gateway.close()
    run.gateway.close()
    run.gateway.recorder.finish()
    assert of_type(lines(path), "run_end")[0]["links"] == 1
    second = tmp_path / "second.jsonl"
    run.gateway.recorder = Recorder(second)
    run.tokens(max_tokens=2, prompt="5 9")
    end_run(run)
    assert len(of_type(lines(second), "request")) == 1
    assert of_type(lines(second), "run_end")[0]["stage0_stage1_complete"] is True


def stage0_totals(rows):
    return sum(r["messages"] for r in of_type(rows, "bytes", hop=metrics.STAGE0_STAGE1))


def test_a_reopened_stream_to_the_same_stage_0_is_not_counted_twice(metered):
    run, path = metered()
    run.tokens(max_tokens=3)
    old = run.gateway._link
    run.register(0)  # the same servicer, a new epoch: the gateway moves to a new stream
    run.tokens(max_tokens=2)
    assert run.gateway._link is not old
    end_run(run)
    rows = lines(path)
    assert stage0_totals(rows) == 5
    end = of_type(rows, "run_end")[0]
    assert (end["links"], end["links_unreported"]) == (2, 0)


# -------------------------------------------------------------- each way a request ends


def test_a_deadline_writes_one_line_with_its_status(metered):
    run, path = metered(delay1=2.0)
    code_of(run.call(timeout_ms=300))
    end_run(run)
    request = of_type(lines(path), "request")
    assert [(r["outcome"], r["code"]) for r in request] == [("failed", "DEADLINE_EXCEEDED")]


def test_a_client_cancel_writes_one_line(metered):
    run, path = metered(delay1=0.3)
    stream = run.call(max_tokens=50)
    next(stream)
    stream.cancel()
    wait_for(lambda: run.gateway.table() == [], "the request to leave the table")
    end_run(run)
    request = of_type(lines(path), "request")
    assert [(r["outcome"], r["code"]) for r in request] == [("cancelled by the client", None)]
    assert request[0]["output_tokens"] >= 1


def test_a_client_leaving_while_a_token_is_on_its_way_is_a_cancel(metered):
    run, path = metered()

    class Context:
        def is_active(self):
            return True

        def code(self):
            return None

    req = pb.GenerateReq(prompt=PROMPT, max_tokens=5, timeout_ms=WAIT * 1000)
    stream = run.gateway.Generate(req, Context())
    next(stream)
    stream.close()
    end_run(run)
    assert [r["outcome"] for r in of_type(lines(path), "request")] == ["cancelled by the client"]


def test_a_nack_writes_one_failed_line(metered):
    run, path = metered(delay1=0.3)
    stream = run.call(max_tokens=20)
    first = next(stream)
    run.gateway.requests[first.rid].results.put(
        pb.Nack(stage=1, rid=first.rid, attempt=1, step=1, reason="STALE_ATTEMPT")
    )
    code, _ = code_of(stream)
    assert code == grpc.StatusCode.UNAVAILABLE
    end_run(run)
    request = of_type(lines(path), "request")
    assert [(r["outcome"], r["code"]) for r in request] == [("failed", "UNAVAILABLE")]


@pytest.mark.parametrize(
    "kwargs, status",
    [({"max_tokens": 0}, "INVALID_ARGUMENT"), ({"prompt": ""}, "INVALID_ARGUMENT")],
)
def test_a_refused_request_writes_a_rejected_line(metered, kwargs, status):
    run, path = metered()
    code_of(run.call(**kwargs))
    end_run(run)
    request = of_type(lines(path), "request")
    assert [(r["outcome"], r["code"]) for r in request] == [("rejected", status)]
    assert request[0]["rid"] is None and request[0]["arrived_ns"] <= request[0]["t_ns"]


def test_a_request_with_no_stages_registered_writes_a_rejected_line(metered):
    run, path = metered(register=False)
    code_of(run.call())
    end_run(run)
    request = of_type(lines(path), "request")
    assert [(r["outcome"], r["code"]) for r in request] == [("rejected", "UNAVAILABLE")]
    assert request[0]["prompt_tokens"] == len(IDS)


def test_a_request_over_the_cap_writes_a_rejected_line(metered):
    run, path = metered(delay1=0.3, max_workers=gateway_module.RESERVED_THREADS + 1, max_queued=0)
    stream = run.call(max_tokens=3)
    next(stream)
    code, _ = code_of(run.call(max_tokens=1))
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert len(list(stream)) == 2
    end_run(run)
    outcomes = sorted((r["outcome"], r["code"]) for r in of_type(lines(path), "request"))
    assert outcomes == [("max_tokens", None), ("rejected", "RESOURCE_EXHAUSTED")]


def test_a_request_that_leaves_the_admission_queue_writes_one_line(metered):
    run, path = metered(delay1=0.1, max_workers=gateway_module.RESERVED_THREADS + 1, max_queued=2)
    first = run.call(max_tokens=12, timeout_ms=20_000)
    next(first)  # holds the only running slot
    code, _ = code_of(run.call(max_tokens=2, timeout_ms=200))
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED
    queued = run.call(max_tokens=2, timeout_ms=20_000)
    give_up = time.monotonic() + 2
    while not run.gateway._waiting and time.monotonic() < give_up:
        time.sleep(0.01)
    queued.cancel()
    while run.gateway._waiting and time.monotonic() < give_up:
        time.sleep(0.01)
    assert len(list(first)) == 11
    end_run(run)
    assert sorted((r["outcome"], r["code"]) for r in of_type(lines(path), "request")) == [
        ("cancelled by the client", None),
        ("max_tokens", None),
        ("rejected", "DEADLINE_EXCEEDED"),
    ]


# -------------------------------------------------------------- fault injection


def test_a_stale_result_is_logged_as_not_accepted_and_counted(metered):
    run, path = metered()
    run.gateway.Push(iter([pb.StepResult(rid="nobody", attempt=1, step=1, token=1)]), None)
    end_run(run)
    rows = lines(path)
    assert [h["accepted"] for h in of_type(rows, "hop")] == [False]
    assert of_type(rows, "bytes", hop=metrics.STAGE1_GATEWAY)[0]["messages"] == 1


def assert_unreported(rows, links=1):
    end = of_type(rows, "run_end")[0]
    assert (end["links"], end["links_unreported"]) == (links, 1)
    assert end["stage0_stage1_complete"] is False


def test_stage_0_stopped_during_a_run_is_reported_as_incomplete(metered):
    run, path = metered(delay1=0.3)
    stream = run.call(max_tokens=20)
    next(stream)
    run.server0.stop(0).wait()
    code, _ = code_of(stream)
    assert code == grpc.StatusCode.UNAVAILABLE
    end_run(run)
    rows = lines(path)
    assert_unreported(rows)
    assert of_type(rows, "bytes", hop=metrics.GATEWAY_STAGE0)  # what the gateway saw is kept
    assert not of_type(rows, "bytes", hop=metrics.STAGE0_STAGE1)
    assert of_type(rows, "request")[0]["code"] == "UNAVAILABLE"


def test_stage_1_stopped_during_a_run_still_gives_stage_0_counters(metered):
    run, path = metered(delay1=0.3)
    stream = run.call(max_tokens=20)
    next(stream)
    run.server1.stop(0).wait()
    code, _ = code_of(stream)
    assert code == grpc.StatusCode.UNAVAILABLE
    end_run(run)
    rows = lines(path)
    assert stage0_totals(rows) >= 1
    assert of_type(rows, "run_end")[0]["stage0_stage1_complete"] is True


class Stub(protocol_pb2_grpc.StageServicer):
    """A Stage 0 that takes messages, answers nothing and ends as it is told."""

    def __init__(self, trailer=None, hold=None):
        self.trailer, self.hold = trailer, hold

    def Steps(self, request_iterator, context):
        for _ in request_iterator:
            pass
        if self.hold is not None:
            self.hold.wait(WAIT)
        if self.trailer is not None:
            context.set_trailing_metadata(((metrics.TRAILER_KEY, self.trailer),))
        return pb.Empty()


@pytest.fixture
def stubbed(metered):
    """A system whose Stage 0 is a Stub, with one request already sent to it."""
    servers = []

    def make(stub):
        run, path = metered()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        protocol_pb2_grpc.add_StageServicer_to_server(stub, server)
        run.real_port0 = run.ports[0]
        run.ports[0] = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        run.register(0)
        code_of(run.call(timeout_ms=100))
        return run, path

    yield make
    for server in servers:
        server.stop(0).wait()


def test_a_stage_that_sends_a_time_gets_no_time_into_the_log(stubbed):
    run, path = stubbed(Stub(trailer=good(987654321)))
    end_run(run)
    rows = lines(path)
    assert_unreported(rows)
    assert "987654321" not in path.read_text(encoding="utf-8")
    assert not of_type(rows, "bytes", hop=metrics.STAGE0_STAGE1)


def test_a_stage_that_sends_no_report_is_counted_as_unreported(stubbed):
    run, path = stubbed(Stub())
    end_run(run)
    assert_unreported(lines(path))


def test_a_hung_stage_0_cannot_hold_up_close(stubbed, monkeypatch):
    monkeypatch.setattr(gateway_module, "CLOSE_WAIT_SECONDS", 0.5)
    hold = threading.Event()
    run, path = stubbed(Stub(hold=hold))
    began = time.monotonic()
    try:
        end_run(run)
        assert time.monotonic() - began < 3
    finally:
        hold.set()
    assert_unreported(lines(path))


def test_a_request_on_a_replaced_stream_to_a_hung_stage_0_fails_at_once(stubbed):
    hold = threading.Event()
    run, path = stubbed(Stub(hold=hold))
    try:
        stream = run.call(max_tokens=2, timeout_ms=WAIT * 1000)
        wait_for(lambda: run.gateway.table(), "the request to run")
        run.ports[0] = run.real_port0
        run.register(0)
        began = time.monotonic()
        assert len(run.tokens(max_tokens=2)) == 2  # this one moves the gateway to the new stream
        code, details = code_of(stream)
        assert code == grpc.StatusCode.UNAVAILABLE and "replaced" in details
        assert time.monotonic() - began < WAIT / 2  # not left to wait for its deadline
    finally:
        hold.set()
    end_run(run)
    assert of_type(lines(path), "run_end")[0]["links"] == 2


def test_a_full_disk_never_fails_a_request(metered):
    run, path = metered()

    class Full:
        def write(self, _):
            raise OSError("no space left")

        def close(self):
            pass

    run.gateway.recorder._file = Full()
    assert len(run.tokens(max_tokens=3)) == 3
    assert run.gateway.recorder.write_errors >= 7  # 3 sent, 3 received, 1 request


def test_a_request_running_at_shutdown_gets_its_line_before_the_end_of_the_run(metered):
    run, path = metered(delay1=0.2)
    stream = run.call(max_tokens=50)
    next(stream)
    done = threading.Event()
    done.set()
    gateway_module.run_until_stopped(run.gateway, run.server, done)
    with pytest.raises(grpc.RpcError):
        list(stream)
    rows = lines(path)
    assert [r["outcome"] for r in of_type(rows, "request")] == ["failed"]
    assert rows[-1]["type"] == "run_end"
    assert [r["type"] for r in rows].index("request") < len(rows) - 1


CHILD = """
import sys
from pllm.control import ExpectedModel, Registry
from pllm.gateway import Gateway, run_until_stopped, serve, stop_event
from pllm.metrics import Recorder

class Words:
    def encode(self, text):
        return [int(word) for word in text.split()]
    def decode(self, ids):
        return ""

gateway = Gateway(Registry("tiny", ExpectedModel(b"1" * 32, 4, 2)), Words(), (), 64, 10**7)
gateway.recorder = Recorder(sys.argv[1])
done = stop_event()
server, port = serve(gateway, "127.0.0.1:0")
print(port, flush=True)
run_until_stopped(gateway, server, done)
"""


def test_the_launchers_stop_signal_ends_the_gateway_cleanly(tmp_path):
    """The launcher sends SIGINT on POSIX and CTRL_BREAK on Windows, and expects exit 0."""
    path = tmp_path / "m.jsonl"
    windows = sys.platform == "win32"
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(path)],
        stdout=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0,
    )
    try:
        assert child.stdout.readline().strip().isdigit()
        began = time.monotonic()
        child.send_signal(signal.CTRL_BREAK_EVENT if windows else signal.SIGINT)
        assert child.wait(STOP_LIMIT) == 0
        assert time.monotonic() - began < STOP_LIMIT
    finally:
        child.kill()
        child.communicate()
    assert lines(path)[-1]["type"] == "run_end"
