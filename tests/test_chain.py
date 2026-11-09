"""The chain over gRPC: gateway client to stage 0 to stage 1 to the gateway's results (R3.4).

The client here stands in for the gateway. Equality with the in-process split is on one
host, fp32, batch 1, one thread count, so tokens and top logits are compared exactly.
"""

import gc
import queue
from concurrent import futures

import grpc
import pytest
import torch
from test_model import tiny_reference
from transformers import Qwen3Config, Qwen3ForCausalLM

from pllm import config, prompts, wire
from pllm.model import ModelConfig
from pllm.models import MODELS_DIR, PINNED, verify
from pllm.proto import protocol_pb2, protocol_pb2_grpc
from pllm.reference import generate
from pllm.split import InProcessSplit
from pllm.stage_service import _DONE, _drain, serve
from pllm.worker import StageWorker, WorkerConfig

LAYERS = 5
HIDDEN = 64  # the tiny model's hidden size
SMOLLM2_DIR = MODELS_DIR / "smollm2-135m"
WAIT = 30  # seconds before a hung chain fails the test


# ---------------------------------------------------------------- the wire format


def test_a_tensor_round_trips_bit_for_bit():
    values = torch.tensor([[0.0, -0.0, 1e-45, 3.4e38], [float("inf"), -float("inf"), 0.1, -7.25]])
    back = wire.tensor_from_proto(wire.tensor_to_proto(values))
    assert back.dtype == torch.float32 and back.shape == values.shape
    assert back.view(torch.int32).tolist() == values.view(torch.int32).tolist()
    nan = torch.tensor([[float("nan")]])
    assert torch.isnan(wire.tensor_from_proto(wire.tensor_to_proto(nan))).all()


def test_the_bytes_are_raw_little_endian_fp32_with_dtype_and_shape():
    tensor = torch.arange(6, dtype=torch.float32).view(2, 3) / 7
    message = wire.tensor_to_proto(tensor)
    assert message.dtype == "float32" and list(message.shape) == [2, 3]
    assert message.data == tensor.numpy().astype("<f4").tobytes()
    assert len(message.data) == 4 * tensor.numel()  # four bytes a value, no text encoding


def test_a_slice_is_sent_as_its_own_values():
    base = torch.randn(8, 4)
    part = base[2:5]
    assert torch.equal(wire.tensor_from_proto(wire.tensor_to_proto(part)), part)
    assert torch.equal(wire.tensor_from_proto(wire.tensor_to_proto(base.t())), base.t())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.int64])
def test_only_fp32_is_sent(dtype):
    with pytest.raises(wire.BadTensor, match="only float32"):
        wire.tensor_to_proto(torch.zeros(2, 2, dtype=dtype))


@pytest.mark.parametrize(
    ("message", "match"),
    [
        (dict(dtype="float64", shape=[1], data=bytes(8)), "not 'float32'"),
        (dict(dtype="float32", shape=[], data=b""), "positive dimensions"),
        (dict(dtype="float32", shape=[2, 0], data=b""), "positive dimensions"),
        (dict(dtype="float32", shape=[2, 3], data=bytes(23)), "needs 24 bytes, got 23"),
        (dict(dtype="float32", shape=[2, 3], data=bytes(28)), "needs 24 bytes, got 28"),
    ],
)
def test_a_tensor_that_does_not_describe_its_bytes_is_rejected(message, match):
    with pytest.raises(wire.BadTensor, match=match):
        wire.tensor_from_proto(protocol_pb2.Tensor(**message))


# ------------------------------------------------------------ the receive limit


def test_the_limit_follows_the_formula_from_the_decision():
    # max_batch 1 x 512 tokens x 576 hidden x 4 bytes, plus the framing margin
    assert config.max_message_bytes(576) == 512 * 576 * 4 + config.FRAMING_MARGIN
    # a 512-token prefill of the largest stretch model (hidden 2048) is exactly gRPC's 4 MiB
    # default before framing, so over it once the message fields are added
    assert 512 * 2048 * 4 == 4 * 1024 * 1024
    assert config.max_message_bytes(2048) > 512 * 2048 * 4


@pytest.mark.parametrize(
    "table",
    ["", "[limits]\nmax_batch = 1\n", "[limits]\nmax_batch = 0\nmax_prompt_tokens = 512\n"]
    + ["[limits]\nmax_batch = true\nmax_prompt_tokens = 512\n"],
)
def test_a_missing_or_bad_limits_table_is_rejected(tmp_path, table):
    path = tmp_path / "hosts.toml"
    path.write_text(table)
    with pytest.raises(ValueError, match=r"\[limits\] needs a positive whole number"):
        config.max_message_bytes(576, path)


class Sink(protocol_pb2_grpc.StageServicer):
    """A Stage that only decodes what it is sent, so the size limit is tested alone."""

    def Steps(self, request_iterator, context):
        for request in request_iterator:
            wire.tensor_from_proto(request.activations)
        return protocol_pb2.Empty()


def send_prefill(options_server, options_channel, hidden=2048, tokens=512):
    server = grpc.server(futures.ThreadPoolExecutor(2), options=options_server)
    protocol_pb2_grpc.add_StageServicer_to_server(Sink(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    request = protocol_pb2.StepRequest(
        rid="r", length=tokens, activations=wire.tensor_to_proto(torch.zeros(tokens, hidden))
    )
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}", options_channel) as channel:
            protocol_pb2_grpc.StageStub(channel).Steps(iter([request]), timeout=WAIT)
    finally:
        server.stop(0)


def test_a_512_token_prefill_over_the_default_limit_needs_the_configured_limit():
    ok = wire.channel_options(2048)
    send_prefill(ok, ok)  # 4.2 MB, accepted with the limit from config
    with pytest.raises(grpc.RpcError) as err:  # the control: the default limit rejects it
        send_prefill([], [])
    assert err.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    with pytest.raises(grpc.RpcError) as err:  # the server's limit alone is not enough
        send_prefill([], ok)
    assert err.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


# -------------------------------------------------------------- the chain itself


class Collector(protocol_pb2_grpc.ResultsServicer):
    """The gateway's Results service: puts every result on a queue."""

    def __init__(self):
        self.results: queue.Queue = queue.Queue()

    def Push(self, request_iterator, context):
        for result in request_iterator:
            self.results.put(result)
        return protocol_pb2.Empty()


class Chain:
    """Two stage servers and a results server on free local ports, and a client that plays
    the gateway: it sends a request's steps on one stream and waits for each result."""

    def __init__(self, model_dir, k):
        n = ModelConfig.from_file(model_dir / "config.json").num_layers
        hidden = ModelConfig.from_file(model_dir / "config.json").hidden_size
        options = wire.channel_options(hidden)
        self.collector = Collector()
        self.results_server = grpc.server(futures.ThreadPoolExecutor(2), options=options)
        protocol_pb2_grpc.add_ResultsServicer_to_server(self.collector, self.results_server)
        results_port = self.results_server.add_insecure_port("127.0.0.1:0")
        self.results_server.start()
        worker1 = StageWorker(WorkerConfig(model_dir, k, n))
        self.server1, self.servicer1, self.port1 = serve(
            worker1, "127.0.0.1:0", results_address=f"127.0.0.1:{results_port}"
        )
        worker0 = StageWorker(WorkerConfig(model_dir, 0, k))
        self.server0, self.servicer0, self.port0 = serve(
            worker0, "127.0.0.1:0", next_address=f"127.0.0.1:{self.port1}"
        )
        self.channel = grpc.insecure_channel(f"127.0.0.1:{self.port0}", options)
        self.stage1_is_gone = False
        self.outbox: queue.Queue = queue.Queue()
        self.call = protocol_pb2_grpc.StageStub(self.channel).Steps.future(_drain(self.outbox))

    def result(self):
        try:
            return self.collector.results.get(timeout=WAIT)
        except queue.Empty:
            raise AssertionError(f"no result in {WAIT} s; stream: {self.call.done()}") from None

    def generate_many(self, jobs, new_tokens):
        """Prefill every prompt in jobs (request id to token ids), then decode greedily,
        sending one step of each request in turn so they interleave on the same stream.
        Returns each request's tokens and top logits."""
        send = {rid: list(ids) for rid, ids in jobs.items()}
        pos = dict.fromkeys(jobs, 0)
        out = {rid: ([], []) for rid in jobs}
        for step in range(new_tokens):
            for rid, ids in send.items():
                self.outbox.put(
                    protocol_pb2.StepRequest(
                        rid=rid,
                        attempt=1,
                        step=step,
                        start_pos=pos[rid],
                        length=len(ids),
                        tokens=protocol_pb2.TokenIds(ids=ids),
                    )
                )
            for _ in jobs:
                result = self.result()
                assert (result.attempt, result.step) == (1, step)
                out[result.rid][0].append(result.token)
                out[result.rid][1].append(result.top_logit)
            for rid in jobs:
                pos[rid] += len(send[rid])
                send[rid] = [out[rid][0][-1]]
        return out

    def generate(self, ids, new_tokens, rid="r"):
        """Prefill ids, then decode greedily; return the tokens and their top logits."""
        return self.generate_many({rid: ids}, new_tokens)[rid]

    def close(self):
        self.outbox.put(_DONE)
        if self.stage1_is_gone:
            with pytest.raises(grpc.RpcError):
                self.call.result(timeout=WAIT)
        else:
            self.call.result(timeout=WAIT)
        self.channel.close()
        for server in (self.server0, self.server1, self.results_server):
            server.stop(0).wait()
        self.servicer0.close()
        self.servicer1.close()


@pytest.fixture
def make_chain():
    chains = []

    def make(model_dir, k):
        chains.append(Chain(model_dir, k))
        return chains[-1]

    yield make
    for chain in chains:
        chain.close()


@pytest.mark.parametrize("qwen3", [False, True])
@pytest.mark.parametrize("tie", [True, False])
def test_the_chain_gives_the_in_process_split_result(tmp_path, make_chain, tie, qwen3):
    tiny_reference(tmp_path, tie, qwen3, layers=LAYERS)
    ids = [5, 9, 100, 3, 42, 7]
    for k in (1, 3, 4):
        expected = generate(InProcessSplit.load(tmp_path, k), ids, 8)
        assert make_chain(tmp_path, k).generate(ids, 8) == expected, f"k={k}"


def test_a_512_token_prompt_is_accepted(tmp_path, make_chain):
    tiny_reference(tmp_path, layers=LAYERS)
    ids = torch.randint(0, 128, (512,), generator=torch.Generator().manual_seed(4)).tolist()
    expected = generate(InProcessSplit.load(tmp_path, 2), ids, 4)
    assert make_chain(tmp_path, 2).generate(ids, 4) == expected


def test_each_request_id_has_its_own_cache(tmp_path, make_chain):
    tiny_reference(tmp_path, layers=LAYERS)
    chain = make_chain(tmp_path, 2)
    prompts_ = {"a": [5, 9, 100, 3], "b": [1, 2, 3, 4, 5, 6, 7], "c": [64, 8]}
    split = InProcessSplit.load(tmp_path, 2)
    alone = {rid: generate(split, ids, 6) for rid, ids in prompts_.items()}
    assert len({tuple(tokens) for tokens, _ in alone.values()}) == 3  # the prompts differ
    # steps of the three requests alternate on one stream, each with its own cache
    assert chain.generate_many(prompts_, 6) == alone
    assert set(chain.servicer0.caches) == set(chain.servicer1.caches) == set(prompts_)


WIDE = 2048  # a hidden size whose 512-token prefill is 4 MiB, over gRPC's default limit


def wide_model(path):
    """A small Qwen3 with hidden size 2048 but narrow layers, so only the messages are big."""
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=WIDE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    Qwen3ForCausalLM(cfg).to(torch.float32).eval().save_pretrained(path)


def test_a_512_token_prompt_over_the_default_limit_goes_through_the_chain(tmp_path, make_chain):
    wide_model(tmp_path)
    ids = torch.randint(0, 128, (512,), generator=torch.Generator().manual_seed(6)).tolist()
    expected = generate(InProcessSplit.load(tmp_path, 1), ids, 3)
    assert make_chain(tmp_path, 1).generate(ids, 3) == expected


def test_without_the_configured_limit_the_same_prompt_is_refused(tmp_path, make_chain, monkeypatch):
    wide_model(tmp_path)
    monkeypatch.setattr(wire, "channel_options", lambda hidden_size: [])  # gRPC's defaults
    chain = make_chain(tmp_path, 1)
    ids = torch.randint(0, 128, (512,), generator=torch.Generator().manual_seed(6)).tolist()
    request = protocol_pb2.StepRequest(rid="x", length=512, tokens=protocol_pb2.TokenIds(ids=ids))
    # stage 0 forwards 4 MiB of activations; stage 1 refuses them at the default limit
    code, details = one_message(chain.port0, request)
    assert code == grpc.StatusCode.UNAVAILABLE and "RESOURCE_EXHAUSTED" in details


def one_message(port, request):
    """Send one request straight to a stage and return the error code it ends with."""
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        with pytest.raises(grpc.RpcError) as err:
            protocol_pb2_grpc.StageStub(channel).Steps(iter([request]), timeout=WAIT)
    return err.value.code(), err.value.details()


def test_a_stage_rejects_what_it_cannot_run(tmp_path, make_chain):
    tiny_reference(tmp_path, layers=LAYERS)
    chain = make_chain(tmp_path, 2)
    tokens = protocol_pb2.TokenIds(ids=[1, 2, 3])
    good = wire.tensor_to_proto(torch.zeros(3, HIDDEN))
    base = dict(rid="x", attempt=1, step=0, start_pos=0, length=3)
    cases = [
        # stage 0 takes tokens, stage 1 takes activations
        (
            chain.port0,
            dict(base, activations=good),
            grpc.StatusCode.INVALID_ARGUMENT,
            "takes tokens",
        ),
        (chain.port1, dict(base, tokens=tokens), grpc.StatusCode.INVALID_ARGUMENT, "takes activ"),
        (chain.port1, dict(base), grpc.StatusCode.INVALID_ARGUMENT, "no payload"),
        # the length field must match the positions carried
        (
            chain.port0,
            dict(base, length=4, tokens=tokens),
            grpc.StatusCode.INVALID_ARGUMENT,
            "length 4",
        ),
        # a malformed tensor, and one of the wrong width
        (
            chain.port1,
            dict(
                base, activations=protocol_pb2.Tensor(dtype="float32", shape=[3, HIDDEN], data=b"")
            ),
            grpc.StatusCode.INVALID_ARGUMENT,
            "needs",
        ),
        (
            chain.port1,
            dict(base, activations=wire.tensor_to_proto(torch.zeros(3, HIDDEN + 1))),
            grpc.StatusCode.INVALID_ARGUMENT,
            "hidden states",
        ),
        # a token id outside the vocabulary
        (
            chain.port0,
            dict(base, length=1, tokens=protocol_pb2.TokenIds(ids=[9999])),
            grpc.StatusCode.INVALID_ARGUMENT,
            "outside the embedding",
        ),
        # a request this stage has not seen cannot start past position 0
        (
            chain.port0,
            dict(base, start_pos=5, tokens=tokens),
            grpc.StatusCode.FAILED_PRECONDITION,
            "unknown",
        ),
    ]
    for port, fields, code, text in cases:
        got, details = one_message(port, protocol_pb2.StepRequest(**fields))
        assert got == code and text in details, (text, got, details)


def test_a_failed_next_hop_ends_the_stream_instead_of_hanging(tmp_path, make_chain):
    tiny_reference(tmp_path, layers=LAYERS)
    chain = make_chain(tmp_path, 2)
    chain.server1.stop(0).wait()
    request = protocol_pb2.StepRequest(
        rid="x", length=3, tokens=protocol_pb2.TokenIds(ids=[1, 2, 3])
    )
    code, details = one_message(chain.port0, request)
    assert code == grpc.StatusCode.UNAVAILABLE and "next hop failed" in details
    chain.stage1_is_gone = True  # the client's own idle stream ends with the same error


def test_a_failed_next_hop_is_reported_on_a_stream_the_client_holds_open(tmp_path, make_chain):
    tiny_reference(tmp_path, layers=LAYERS)
    chain = make_chain(tmp_path, 2)
    chain.server1.stop(0).wait()
    chain.outbox.put(
        protocol_pb2.StepRequest(rid="x", length=3, tokens=protocol_pb2.TokenIds(ids=[1, 2, 3]))
    )
    # the stream is not closed by the client; stage 0 must end it by itself
    with pytest.raises(grpc.RpcError) as err:
        chain.call.result(timeout=WAIT)
    assert err.value.code() == grpc.StatusCode.UNAVAILABLE
    chain.stage1_is_gone = True


def test_a_failed_step_drops_the_cache_of_that_request(tmp_path, make_chain):
    tiny_reference(tmp_path, layers=LAYERS)
    chain = make_chain(tmp_path, 2)
    first = protocol_pb2.StepRequest(
        rid="x", step=0, length=3, tokens=protocol_pb2.TokenIds(ids=[1, 2, 3])
    )
    bad = protocol_pb2.StepRequest(
        rid="x", step=1, start_pos=3, length=1, tokens=protocol_pb2.TokenIds(ids=[9999])
    )
    with grpc.insecure_channel(f"127.0.0.1:{chain.port0}") as channel:
        with pytest.raises(grpc.RpcError) as err:
            protocol_pb2_grpc.StageStub(channel).Steps(iter([first, bad]), timeout=WAIT)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "x" not in chain.servicer0.caches


def test_an_unexpected_error_in_a_stage_is_internal_and_drops_the_cache(
    tmp_path, make_chain, monkeypatch
):
    tiny_reference(tmp_path, layers=LAYERS)
    chain = make_chain(tmp_path, 2)

    def broken(inputs, start_pos, cache):
        raise RuntimeError("secret detail that stays in the log")

    monkeypatch.setattr(chain.servicer0.worker, "step", broken)
    request = protocol_pb2.StepRequest(
        rid="x", length=3, tokens=protocol_pb2.TokenIds(ids=[1, 2, 3])
    )
    code, details = one_message(chain.port0, request)
    assert code == grpc.StatusCode.INTERNAL and details == "stage failed: RuntimeError"
    assert "x" not in chain.servicer0.caches


def test_every_step_runs_with_the_process_thread_count(tmp_path, make_chain):
    tiny_reference(tmp_path, layers=LAYERS)
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        chain = make_chain(tmp_path, 2)
        seen = []
        real = chain.servicer0.worker.step

        def spy(inputs, start_pos, cache):
            seen.append(torch.get_num_threads())
            return real(inputs, start_pos, cache)

        chain.servicer0.worker.step = spy
        chain.generate([5, 9, 100, 3], 4)
    finally:
        torch.set_num_threads(before)
    assert seen == [2, 2, 2, 2]


def test_the_last_stage_needs_a_results_address_and_the_others_a_next_hop(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    with pytest.raises(ValueError, match="results address"):
        serve(StageWorker(WorkerConfig(tmp_path, 2, LAYERS)), "127.0.0.1:0")
    with pytest.raises(ValueError, match="next stage's address"):
        serve(StageWorker(WorkerConfig(tmp_path, 0, 2)), "127.0.0.1:0")


# ------------------------------------------------------------- the real model


@pytest.fixture
def four_threads():
    """A fixed thread count for the equality runs (R8.9)."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(before)


@pytest.mark.skipif(
    not (SMOLLM2_DIR / "model.safetensors").exists(), reason="model weights not downloaded"
)
@pytest.mark.parametrize("length", [16, 512])
def test_smollm2_through_the_chain_equals_the_in_process_split(make_chain, four_threads, length):
    for filename, expected in PINNED["smollm2-135m"].files.items():
        verify(SMOLLM2_DIR / filename, expected)
    ids = prompts.prompts("smollm2-135m", length)[0]
    k = config.split_point("smollm2-135m")
    expected = generate(InProcessSplit.load(SMOLLM2_DIR, k), ids, 4)
    gc.collect()  # the whole model is freed before the stages load, to stay inside Host B memory
    assert make_chain(SMOLLM2_DIR, k).generate(ids, 4) == expected
