"""The gateway: Generate, the decode loop, the request table and freeing (R4.1 to R4.3, R4.6).

Each test runs a real gateway and two real stage servers on loopback. Equality with the
single-process reference is on one host, fp32, batch 1, so tokens and top logits are
compared exactly.
"""

import json
import threading
import time

import grpc
import pytest
from test_model import tiny_reference

from pllm import config, gateway
from pllm.control import ExpectedModel, Registry
from pllm.gateway import Gateway, eos_ids, serve
from pllm.model import ModelConfig
from pllm.models import MODELS_DIR
from pllm.proto import protocol_pb2, protocol_pb2_grpc
from pllm.reference import generate
from pllm.split import InProcessSplit
from pllm.stage_service import serve as serve_stage
from pllm.worker import StageWorker, WorkerConfig

LAYERS = 4
HIDDEN = 64  # the tiny model's hidden size
WAIT = 30  # seconds before a hung test fails
DIGEST = b"\x01" * 32
PROMPT = "5 9 100 3 42 7"
IDS = [5, 9, 100, 3, 42, 7]


class Words:
    """A stand-in tokenizer: token ids written as numbers."""

    def encode(self, text):
        return [int(word) for word in text.split()]

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


class System:
    """A gateway, stage 1 pushing to it and stage 0 forwarding to stage 1, all registered."""

    def __init__(
        self,
        model_dir,
        k,
        eos=(),
        max_bytes=None,
        delay1=0.0,
        register=True,
        tokenizer=None,
        delay0=0.0,
        max_workers=gateway.MAX_RUNNING + gateway.RESERVED_THREADS,
    ):
        n = ModelConfig.from_file(model_dir / "config.json").num_layers
        limit = max_bytes or config.max_message_bytes(HIDDEN)
        self.registry = Registry("tiny", ExpectedModel(DIGEST, n, k))
        self.gateway = Gateway(self.registry, tokenizer or Words(), eos, HIDDEN, limit)
        self.server, port = serve(self.gateway, "127.0.0.1:0", max_workers)
        self.address = f"127.0.0.1:{port}"
        self.server1, self.servicer1, port1 = serve_stage(
            StageWorker(WorkerConfig(model_dir, k, n, delay1)),
            "127.0.0.1:0",
            results_address=self.address,
        )
        self.server0, self.servicer0, port0 = serve_stage(
            StageWorker(WorkerConfig(model_dir, 0, k, delay0)),
            "127.0.0.1:0",
            next_address=f"127.0.0.1:{port1}",
        )
        self.ports = {0: port0, 1: port1}
        self.layers = {0: (0, k), 1: (k, n)}
        self.channel = grpc.insecure_channel(self.address)
        self.stub = protocol_pb2_grpc.FrontendStub(self.channel)
        if register:
            for stage in (0, 1):
                self.register(stage)

    def register(self, stage):
        """Register a stage and give both stages the epochs a worker and its heartbeat
        would: its own epoch, and the route update its neighbour receives."""
        start, end = self.layers[stage]
        entry = self.registry.register(
            protocol_pb2.RegisterReq(
                stage=stage,
                layer_start=start,
                layer_end=end,
                model_id="tiny",
                weights_sha256=DIGEST,
                addr=f"127.0.0.1:{self.ports[stage]}",
            )
        )
        servicers = (self.servicer0, self.servicer1)
        servicers[stage].own_epoch = entry.epoch
        servicers[1 - stage].route(
            protocol_pb2.RouteUpdate(next_stage=stage, addr=entry.addr, epoch=entry.epoch)
        )
        return entry

    def call(self, prompt=PROMPT, max_tokens=8, timeout_ms=WAIT * 1000):
        req = protocol_pb2.GenerateReq(prompt=prompt, max_tokens=max_tokens, timeout_ms=timeout_ms)
        return self.stub.Generate(req, timeout=WAIT)

    def tokens(self, **kwargs):
        return list(self.call(**kwargs))

    def deliver_frees(self):
        """What the heartbeat story will do: hand each queued Free to its stage."""
        for stage, servicer in ((0, self.servicer0), (1, self.servicer1)):
            for message in self.gateway.frees_for(stage):
                servicer.free(message)

    def close(self):
        self.channel.close()
        self.gateway.close()
        for server in (self.server0, self.server1, self.server):
            server.stop(0).wait()
        self.servicer0.close()
        self.servicer1.close()


@pytest.fixture
def system(tmp_path):
    made = []

    def make(k=2, tie=True, qwen3=False, **kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, tie, qwen3, layers=LAYERS)
        made.append(System(tmp_path, k, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def wait_for(predicate, what, timeout=WAIT):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def code_of(call):
    with pytest.raises(grpc.RpcError) as caught:
        list(call)
    return caught.value.code(), caught.value.details()


# -------------------------------------------------------------- tokens, order, equality


@pytest.mark.parametrize("qwen3", [False, True])
@pytest.mark.parametrize("tie", [True, False])
def test_generate_gives_the_single_process_result_in_order(tmp_path, system, tie, qwen3):
    for k in (1, 3):
        run = system(k=k, tie=tie, qwen3=qwen3)
        tokens = run.tokens(max_tokens=8)
        expected = generate(InProcessSplit.load(tmp_path, k), IDS, 8)
        assert ([t.token for t in tokens], [t.top_logit for t in tokens]) == expected, f"k={k}"
        assert [t.step for t in tokens] == list(range(8))
        assert len({t.rid for t in tokens}) == 1 and tokens[0].rid
        assert [t.text for t in tokens] == [f"<{t.token}>" for t in tokens]


def test_the_prompt_goes_as_one_message_at_position_zero_then_one_token_per_step(system):
    run = system()
    sent = []
    real = run.servicer0._handle

    def spy(request, context, *rest):
        sent.append((request.attempt, request.step, request.start_pos, list(request.tokens.ids)))
        return real(request, context, *rest)

    run.servicer0._handle = spy
    tokens = [t.token for t in run.tokens(max_tokens=3)]
    assert sent == [
        (1, 0, 0, IDS),
        (1, 1, len(IDS), [tokens[0]]),
        (1, 2, len(IDS) + 1, [tokens[1]]),
    ]


def test_a_request_stops_at_max_tokens(system):
    assert len(system().tokens(max_tokens=3)) == 3


def test_a_request_stops_at_end_of_sequence(tmp_path, system):
    tiny_reference(tmp_path, layers=LAYERS)
    ids = [60, 61, 62, 63]  # the tiny model's greedy output changes token part way through
    expected, _ = generate(InProcessSplit.load(tmp_path, 2), ids, 10)
    # the first step whose token has not appeared before, so the stop is unambiguous
    stop = next(i for i in range(1, 10) if expected[i] not in expected[:i])
    tokens = system(eos=[expected[stop]]).tokens(prompt="60 61 62 63", max_tokens=10)
    assert [t.token for t in tokens] == expected[: stop + 1] and stop + 1 < 10


# -------------------------------------------------------------- request table and frees


def test_the_request_table_shows_a_running_request(system):
    run = system(delay1=0.5)
    rows, stream = [], run.call(max_tokens=4, timeout_ms=20_000)
    first = next(stream)
    wait_for(lambda: run.gateway.table() and run.gateway.table()[0]["tokens"], "a row")
    rows = run.gateway.table()
    rest = list(stream)
    assert len(rows) == 1
    row = rows[0]
    assert row["rid"] == first.rid and row["attempt"] == 1
    assert row["tokens"][0] == first.token
    assert row["expected_step"] == len(row["tokens"])
    assert 0 < row["deadline_s"] <= 20
    assert len(rest) == 3 and run.gateway.table() == []


def test_an_ended_request_is_freed_on_both_stages(system):
    run = system()
    rid = run.tokens(max_tokens=4)[0].rid
    assert run.servicer0.cached_requests == run.servicer1.cached_requests == 1
    frees = {stage: run.gateway.frees_for(stage) for stage in (0, 1)}
    assert all([(f.rid, f.attempt) for f in frees[s]] == [(rid, 1)] for s in (0, 1))
    for stage, servicer in ((0, run.servicer0), (1, run.servicer1)):
        servicer.free(frees[stage][0])
    assert run.servicer0.cached_requests == run.servicer1.cached_requests == 0


def test_requests_one_after_another_each_free_their_state(system):
    run = system()
    for _ in range(3):
        run.tokens(max_tokens=2)
        run.deliver_frees()
    assert run.servicer0.cached_requests == run.servicer1.cached_requests == 0


# -------------------------------------------------------------- admission


def test_a_prompt_over_the_receive_limit_is_rejected_before_stage_0(system):
    run = system(max_bytes=4 * HIDDEN * 4 + config.FRAMING_MARGIN)  # room for 4 tokens
    code, details = code_of(run.call(prompt="1 2 3 4 5"))
    assert code == grpc.StatusCode.INVALID_ARGUMENT
    assert "5 tokens" in details and "receive limit" in details
    assert run.gateway._link is None and run.servicer0.cached_requests == 0
    assert run.gateway.table() == []
    assert len(run.tokens(prompt="1 2 3 4", max_tokens=1)) == 1  # at the limit it fits


def test_the_limit_check_uses_the_configured_limit_by_default(system):
    run = system()
    assert run.gateway.max_bytes == config.max_message_bytes(HIDDEN)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_tokens": 0}, "max_tokens"),
        ({"timeout_ms": 0}, "timeout_ms"),
        ({"prompt": ""}, "no tokens"),
    ],
)
def test_a_bad_request_is_rejected(system, kwargs, match):
    code, details = code_of(system().call(**kwargs))
    assert code == grpc.StatusCode.INVALID_ARGUMENT and match in details


# -------------------------------------------------------------- failure handling


@pytest.mark.parametrize("registered", [(), (0,), (1,)])
def test_a_request_without_both_stages_registered_is_unavailable(system, registered):
    run = system(register=False)
    for stage in registered:
        run.register(stage)
    code, details = code_of(run.call())
    assert code == grpc.StatusCode.UNAVAILABLE and "not registered" in details


def test_a_request_fails_cleanly_when_stage_1_is_down(system):
    run = system()
    run.server1.stop(0).wait()
    code, details = code_of(run.call())
    assert code == grpc.StatusCode.UNAVAILABLE and "stage 0 stream failed" in details
    assert run.gateway.table() == []
    assert [len(run.gateway.frees_for(s)) for s in (0, 1)] == [1, 1]


def test_a_request_fails_cleanly_when_stage_0_is_down(system):
    run = system()
    run.server0.stop(0).wait()
    code, _ = code_of(run.call())
    assert code == grpc.StatusCode.UNAVAILABLE
    assert run.gateway.table() == []


def test_a_request_past_its_deadline_fails_and_is_freed(system):
    run = system(delay1=2.0)
    began = time.monotonic()
    code, details = code_of(run.call(timeout_ms=300))
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED and "deadline" in details
    assert time.monotonic() - began < 1.5  # judged by the gateway, not by a slow stage
    assert run.gateway.table() == []
    assert [len(run.gateway.frees_for(s)) for s in (0, 1)] == [1, 1]


def test_a_request_the_client_cancels_ends_and_is_freed(system):
    run = system(delay1=0.3)
    stream = run.call(max_tokens=50)
    next(stream)
    stream.cancel()
    wait_for(lambda: run.gateway.table() == [], "the request to leave the table")
    wait_for(lambda: run.gateway.frees[1].qsize() == 1, "a free for stage 1")
    assert len(run.gateway.frees_for(0)) == 1


def test_after_a_failed_stage_0_stream_the_next_request_opens_a_new_one(system):
    run = system()
    run.tokens(max_tokens=1)
    first = run.gateway._link
    first.call.cancel()  # the stream to stage 0 ends
    wait_for(lambda: first.failure() is not None, "the stream to end")
    assert len(run.tokens(max_tokens=2)) == 2
    assert run.gateway._link is not first


def test_a_re_registered_stage_0_gets_a_new_stream_with_its_new_epoch(system):
    run = system()
    run.tokens(max_tokens=1)
    old = run.gateway._link
    entry = run.register(0)
    assert len(run.tokens(max_tokens=1)) == 1
    assert run.gateway._link is not old and run.gateway._link.epoch == entry.epoch


@pytest.mark.parametrize("stage", [0, 1])
def test_a_stage_that_dies_during_a_request_fails_it_cleanly(system, stage):
    run = system(delay1=0.3)
    stream = run.call(max_tokens=20)
    next(stream)
    (run.server0 if stage == 0 else run.server1).stop(0).wait()
    code, details = code_of(stream)
    assert code == grpc.StatusCode.UNAVAILABLE and "stage 0 stream failed" in details
    assert run.gateway.table() == []
    assert [len(run.gateway.frees_for(s)) for s in (0, 1)] == [1, 1]


def test_a_result_for_the_right_request_but_another_step_or_attempt_is_dropped(system):
    alone = [t.token for t in system().tokens(max_tokens=4)]
    run = system(delay1=0.3)
    stream = run.call(max_tokens=4)
    first = next(stream)
    strays = [
        protocol_pb2.StepResult(rid=first.rid, attempt=2, step=1, token=99),
        protocol_pb2.StepResult(rid=first.rid, attempt=1, step=3, token=99),
    ]
    run.gateway.Push(iter(strays), None)
    assert [first.token] + [t.token for t in stream] == alone
    assert run.gateway.dropped == 2


def test_a_stale_result_already_queued_for_a_request_is_not_taken_as_the_next_step(system):
    alone = [t.token for t in system().tokens(max_tokens=4)]
    run = system(delay1=0.3)
    stream = run.call(max_tokens=4)
    first = next(stream)
    request = run.gateway.requests[first.rid]
    request.results.put(protocol_pb2.StepResult(rid=first.rid, attempt=1, step=0, token=99))
    assert [first.token] + [t.token for t in stream] == alone
    assert run.gateway.dropped == 1


class Bytes:
    """A stand-in byte-level tokenizer: 1 and 2 are the two bytes of an e with an acute
    accent, every other id is a single ASCII letter."""

    # 63 and 87 are what the tiny model gives for "60 61 62 63", so its stream splits one
    PIECES = {1: bytes([0xC3]), 2: bytes([0xA9]), 63: bytes([0xC3]), 87: bytes([0xA9])}

    def encode(self, text):
        return [int(word) for word in text.split()]

    def decode(self, ids):
        raw = b"".join(self.PIECES.get(i, bytes([97 + i % 26])) for i in ids)
        return raw.decode("utf-8", errors="replace")


def test_a_character_split_across_tokens_is_sent_once_it_is_complete(system):
    gate = system(tokenizer=Bytes()).gateway
    assert gate._text([1], final=False) is None
    assert gate._text([1, 2], final=False) == chr(0xE9)
    assert gate._text([1], final=True) == chr(0xFFFD)
    assert gate._text([1, 1, 1, 1], final=False) is not None  # never held back for ever


def test_the_streamed_text_adds_up_to_the_text_of_all_tokens(system):
    run = system(tokenizer=Bytes())
    tokens = run.tokens(prompt="60 61 62 63", max_tokens=10)
    ids = [t.token for t in tokens]
    assert 87 in ids and "" in [t.text for t in tokens]  # a character was held back
    assert "".join(t.text for t in tokens) == Bytes().decode(ids)


def test_a_result_that_matches_no_waiting_step_is_dropped_and_counted(system):
    run = system()
    stray = protocol_pb2.StepResult(rid="nobody", attempt=1, step=0, token=1)
    run.gateway.Push(iter([stray]), None)
    assert run.gateway.dropped == 1


def test_the_stage_0_message_carries_its_epoch_and_the_gateway_as_sender(system):
    run = system()
    seen = []
    real = run.servicer0._handle

    def spy(request, context, *rest):
        seen.append((request.target_epoch, request.sender_epoch))
        return real(request, context, *rest)

    run.servicer0._handle = spy
    run.tokens(max_tokens=1)
    assert seen == [(run.registry.stages[0].epoch, 0)]


# -------------------------------------------------------------- the real model files


def test_eos_ids_reads_an_int_a_list_or_nothing(tmp_path):
    for value, want in ((2, [2]), ([2, 7], [2, 7]), (None, [])):
        (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": value}))
        assert eos_ids(tmp_path) == want


SMOLLM2_DIR = MODELS_DIR / "smollm2-135m"


@pytest.mark.skipif(
    not (SMOLLM2_DIR / "tokenizer.json").exists(), reason="SmolLM2-135M is not downloaded"
)
def test_the_smollm2_tokenizer_round_trips_a_prompt_and_has_an_end_token():
    tokenizer = gateway.Tokenizer("smollm2-135m")
    ids = tokenizer.encode("The capital of France is")
    assert ids and tokenizer.decode(ids) == "The capital of France is"
    assert eos_ids(SMOLLM2_DIR) == [2]


def test_generate_runs_alongside_other_requests_without_mixing_them(system):
    """Not the many-in-flight story (#26): two clients at once each get their own tokens."""
    run = system()
    alone = [t.token for t in run.tokens(max_tokens=4)]
    out = {}

    def one(name):
        out[name] = [t.token for t in run.tokens(max_tokens=4)]

    threads = [threading.Thread(target=one, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    assert out == {"a": alone, "b": alone}


# -------------------------------------------------------------- server threads


def outcomes(run, clients, **kwargs):
    """Start clients at once; return the token count or the status code of each."""
    out = [None] * clients
    gate = threading.Barrier(clients)

    def one(i):
        gate.wait()
        try:
            out[i] = len(run.tokens(**kwargs))
        except grpc.RpcError as err:
            out[i] = err.code()

    threads = [threading.Thread(target=one, args=(i,)) for i in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    return out


def test_a_request_over_the_running_limit_is_refused_and_the_others_finish(system):
    run = system(delay1=0.3, max_workers=gateway.RESERVED_THREADS + 3)  # 3 requests at once
    assert run.gateway.max_running == 3
    streams = [run.call(max_tokens=3, timeout_ms=20_000) for _ in range(3)]
    firsts = [next(stream) for stream in streams]
    wait_for(lambda: len(run.gateway.table()) == 3, "three running requests")
    code, details = code_of(run.call(max_tokens=3))
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED and "3 requests" in details
    assert [len([first, *stream]) for first, stream in zip(firsts, streams, strict=True)] == [
        3,
        3,
        3,
    ]
    assert len(run.tokens(max_tokens=3)) == 3  # the slots are given back


def test_a_full_thread_pool_refuses_requests_instead_of_stalling_them(system):
    """The reported fault: as many clients as server threads used to leave none for the
    results stream, and every request then failed at its deadline."""
    run = system(delay1=0.2, max_workers=gateway.RESERVED_THREADS + 3)
    out = outcomes(run, 5, max_tokens=3, timeout_ms=10_000)
    assert grpc.StatusCode.DEADLINE_EXCEEDED not in out
    assert out.count(3) >= 3
    assert all(x == 3 or x == grpc.StatusCode.RESOURCE_EXHAUSTED for x in out)


def test_serve_needs_threads_beyond_the_reserved_ones(system):
    run = system()
    with pytest.raises(ValueError, match="max_workers"):
        serve(run.gateway, "127.0.0.1:0", gateway.RESERVED_THREADS)
