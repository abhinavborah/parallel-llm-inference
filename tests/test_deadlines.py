"""Deadlines on every hop and no request that hangs (R5.4, R5.6; decision #32).

The gateway judges deadlines on its own monotonic clock. Each step carries budget_ms, the
time left, and a stage turns it into a deadline on its own monotonic clock on arrival.
Equality with a fault-free run is on one host, fp32, batch 1.
"""

import subprocess
import sys
import threading
import time

import grpc
import pytest
from test_gateway import WAIT, System, Words, wait_for
from test_model import tiny_reference

from pllm import faults
from pllm.control import ExpectedModel, Registry
from pllm.faults import StepDelay
from pllm.gateway import Gateway, serve
from pllm.model import ModelConfig
from pllm.proto import protocol_pb2, protocol_pb2_grpc
from pllm.stage_service import budget_ms, due
from pllm.stage_service import serve as serve_stage
from pllm.worker import StageWorker, WorkerConfig, join_gateway, sha256_of

LAYERS = 4
K = 2  # split point: stage 0 runs layers 0 and 1
PROMPT = "5 9 100 3 42 7"
FAST = 0.1  # heartbeat interval in these tests
TOLERANCE = 0.5  # seconds a request may end after its deadline, on loopback (R5.6)


@pytest.fixture
def system(tmp_path):
    made = []

    def make(**kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        made.append(System(tmp_path, K, **kwargs))
        return made[-1]

    yield make
    for each in made:
        each.close()


def spy(servicer, seen):
    """Record (rid, step, budget_ms, local deadline) of each message the stage handles."""
    real = servicer._handle

    def handle(request, context, next_epoch, deadline=None):
        seen.append((request.rid, request.step, request.budget_ms, deadline))
        return real(request, context, next_epoch, deadline)

    servicer._handle = handle


# -------------------------------------------------------------- the budget arithmetic


def test_the_budget_is_the_time_left_rounded_up_and_never_zero():
    assert budget_ms(None, 5.0) == 0
    assert budget_ms(5.25, 5.0) == 250
    assert budget_ms(5.0001, 5.0) == 1
    assert budget_ms(5.0, 5.0) == 1  # an exhausted budget still reads as a deadline
    assert budget_ms(4.0, 5.0) == 1
    assert budget_ms(2.0**32, 0.0) == 2**32 - 1  # never more than the uint32 field holds


def test_a_budget_becomes_a_deadline_on_the_receivers_clock():
    assert due(0, 100.0) is None  # no deadline
    assert due(250, 100.0) == pytest.approx(100.25)


# -------------------------------------------------------------- carried on every hop


def test_every_hop_carries_the_time_left_and_each_stage_sets_its_own_deadline(system):
    run = system(delay0=0.2)  # Stage 0 spends 0.2 s on each step
    at0, at1 = [], []
    spy(run.servicer0, at0)
    spy(run.servicer1, at1)
    timeout_ms = 20_000
    sent = time.monotonic()
    tokens = run.tokens(max_tokens=4, timeout_ms=timeout_ms)
    assert [t.step for t in tokens] == [0, 1, 2, 3]
    assert [step for _, step, _, _ in at0] == [0, 1, 2, 3]
    assert [step for _, step, _, _ in at1] == [0, 1, 2, 3]
    budgets0 = [budget for _, _, budget, _ in at0]
    budgets1 = [budget for _, _, budget, _ in at1]
    assert all(0 < budget <= timeout_ms for budget in budgets0)
    assert budgets0 == sorted(budgets0, reverse=True)  # the time left only shrinks
    # Stage 1 gets what is left of Stage 0's budget: less by the time Stage 0 spent
    assert all(0 < b1 <= b0 - 150 for b0, b1 in zip(budgets0, budgets1, strict=True))
    for *_, deadline in at0 + at1:
        assert deadline is not None  # set on arrival, within the budget of the call
        assert sent < deadline <= sent + timeout_ms / 1000 + 1.0


def test_a_step_whose_deadline_passed_while_it_waited_is_dropped_not_run(system):
    """Stage 0 is busy with a slow step while a step with a short budget waits behind it:
    the waiting step expires and is dropped before it runs, and is never forwarded."""
    run = system(delay0=0.4)
    at1 = []
    spy(run.servicer1, at1)
    epoch0 = run.registry.stages[0].epoch

    def step(rid, budget):
        return protocol_pb2.StepRequest(
            rid=rid,
            attempt=1,
            step=0,
            start_pos=0,
            length=2,
            target_epoch=epoch0,
            sender_epoch=0,
            tokens=protocol_pb2.TokenIds(ids=[5, 9]),
            budget_ms=budget,
        )

    with grpc.insecure_channel(f"127.0.0.1:{run.ports[0]}") as channel:
        protocol_pb2_grpc.StageStub(channel).Steps(
            iter([step("slow", 0), step("short", 100), step("long", 20_000)]), timeout=WAIT
        )
    wait_for(lambda: len(at1) == 2, "Stage 1 to get the two steps that did not expire")
    assert [rid for rid, *_ in at1] == ["slow", "long"]
    assert run.servicer0.expired == 1 and run.servicer1.expired == 0
    assert run.servicer0.store.attempt("short") is None  # nothing was cached for it


def test_a_stepped_wall_clock_changes_no_deadline(system, monkeypatch):
    """Deadlines use monotonic clocks only: the wall clock jumping an hour forward and back
    on every step changes neither the tokens nor what expires."""
    expected = [t.token for t in system().tokens(max_tokens=4)]
    run = system()
    real_time = time.time
    jumps = iter([3600, -3600] * 100)
    offset = [0]
    monkeypatch.setattr(time, "time", lambda: real_time() + offset[0])
    at0 = []
    spy(run.servicer0, at0)
    inner = run.servicer0._handle

    def stepping(*args):
        offset[0] = next(jumps)
        return inner(*args)

    run.servicer0._handle = stepping
    tokens = run.tokens(max_tokens=4, timeout_ms=20_000)
    assert [t.token for t in tokens] == expected
    assert run.servicer0.expired == run.servicer1.expired == 0
    assert all(0 < budget <= 20_000 for _, _, budget, _ in at0)
    # and a request with a short deadline still fails at it, not an hour early or late
    run.servicer1.worker.delay = StepDelay(0.05)  # 200 steps take 10 s, far past 300 ms
    began = time.monotonic()
    with pytest.raises(grpc.RpcError) as err:
        run.tokens(max_tokens=200, timeout_ms=300)
    assert err.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    assert time.monotonic() - began < 0.3 + TOLERANCE


# -------------------------------------------------------------- no request hangs

CHILD = """
import sys, threading
from pathlib import Path
from pllm.stage_service import serve
from pllm.worker import StageWorker, WorkerConfig, join_gateway
model_dir, stage, k, n, gateway, neighbour = sys.argv[1:7]
stage, k, n = int(stage), int(k), int(n)
layers = (0, k) if stage == 0 else (k, n)
where = {"next_address": neighbour} if stage == 0 else {"results_address": gateway}
server, servicer, port = serve(StageWorker(WorkerConfig(Path(model_dir), *layers)),
                               "127.0.0.1:0", **where)
print(port, flush=True)
join_gateway(gateway, stage, layers, "tiny", Path(model_dir), f"127.0.0.1:{port}", servicer,
             threading.Event(), interval=float(sys.argv[7]))
"""


class Split:
    """A real gateway; one stage in a child process (so it can be suspended), the other in
    this process. Both register and keep their heartbeat streams, as a worker does."""

    def __init__(self, model_dir, suspended):
        n = ModelConfig.from_file(model_dir / "config.json").num_layers
        digest = bytes.fromhex(sha256_of(model_dir / "model.safetensors"))
        self.gateway = Gateway(
            Registry("tiny", ExpectedModel(digest, n, K)), Words(), (), 64, 2**20
        )
        self.server, port = serve(self.gateway, "127.0.0.1:0")
        self.address = f"127.0.0.1:{port}"
        self.stop = threading.Event()
        self.threads, self.servers, self.child = [], [], None
        self.channel = grpc.insecure_channel(self.address)
        self.stub = protocol_pb2_grpc.FrontendStub(self.channel)
        layers = (0, K) if suspended == 1 else (K, n)  # the stage kept in this process
        try:
            if suspended == 0:  # the child forwards to this process's Stage 1: start it first
                stage_port = self._local(model_dir, 1, layers, results_address=self.address)
                self.child = self._spawn(model_dir, 0, n, f"127.0.0.1:{stage_port}")
                self._port()  # the child is serving
            else:
                self.child = self._spawn(model_dir, 1, n, "")
                child_port = self._port()
                self._local(model_dir, 0, layers, next_address=f"127.0.0.1:{child_port}")
        except BaseException:
            self.close()  # a child that failed to start leaves nothing running
            raise

    def _port(self):
        """The port the child prints once it serves, or an error after WAIT seconds."""
        got = []
        reader = threading.Thread(target=lambda: got.append(self.child.stdout.readline()))
        reader.daemon = True
        reader.start()
        reader.join(WAIT)
        if not got or not got[0].strip():
            raise RuntimeError("the stage process did not start serving")
        return int(got[0])

    def _spawn(self, model_dir, stage, n, neighbour):
        args = [str(model_dir), str(stage), str(K), str(n), self.address, neighbour, str(FAST)]
        return subprocess.Popen(
            [sys.executable, "-c", CHILD, *args], stdout=subprocess.PIPE, text=True
        )

    def _local(self, model_dir, stage, layers, **where):
        server, servicer, port = serve_stage(
            StageWorker(WorkerConfig(model_dir, *layers)), "127.0.0.1:0", **where
        )
        self.servers.append((server, servicer))
        args = (self.address, stage, layers, "tiny", model_dir, f"127.0.0.1:{port}", servicer)
        thread = threading.Thread(
            target=join_gateway, args=(*args, self.stop), kwargs={"interval": FAST}, daemon=True
        )
        thread.start()
        self.threads.append(thread)
        return port

    def call(self, max_tokens, timeout_ms):
        req = protocol_pb2.GenerateReq(prompt=PROMPT, max_tokens=max_tokens, timeout_ms=timeout_ms)
        return [t.token for t in self.stub.Generate(req, timeout=WAIT)]

    def ready(self):
        """The tokens of a first request, once both stages are registered and routed."""
        ends = time.monotonic() + WAIT
        while True:
            try:
                return self.call(4, 5_000)
            except grpc.RpcError:
                if time.monotonic() > ends:
                    raise
                time.sleep(0.1)

    def cached(self):
        """Requests with a cache at the stage in this process."""
        return sum(servicer.cached_requests for _, servicer in self.servers)

    def close(self):
        if self.child is not None:
            faults.resume(self.child.pid)
            faults.kill(self.child.pid)
            self.child.wait(WAIT)
        self.stop.set()
        for thread in self.threads:
            thread.join(WAIT)
        self.channel.close()
        self.gateway.close()
        self.server.stop(0).wait()
        for server, servicer in self.servers:
            server.stop(0).wait()
            servicer.close()


@pytest.mark.parametrize("suspended", [0, 1], ids=["stage0", "stage1"])
def test_with_a_suspended_worker_every_request_ends_by_its_deadline(tmp_path, suspended):
    """A real stage process is suspended with requests in flight and more arriving: each
    request fails with DEADLINE_EXCEEDED within its deadline plus the tolerance, and none
    hangs. After a resume the pipeline serves again with the same tokens."""
    tiny_reference(tmp_path, layers=LAYERS)
    split = Split(tmp_path, suspended)
    try:
        before = split.ready()
        faults.suspend(split.child.pid)
        timeout_ms = 500  # the child stays suspended about 0.6 s in all
        outcomes = {}

        def one(i):
            began = time.monotonic()
            try:
                split.call(50, timeout_ms)
                outcomes[i] = ("completed", time.monotonic() - began)
            except grpc.RpcError as err:
                outcomes[i] = (err.code(), time.monotonic() - began)

        threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for i, thread in enumerate(threads):
            thread.start()
            if i == 1:
                time.sleep(0.1)  # two in flight first, then two more arrive
        for thread in threads:
            thread.join(WAIT)
        faults.resume(split.child.pid)
        assert len(outcomes) == 4
        for code, took in outcomes.values():
            assert code == grpc.StatusCode.DEADLINE_EXCEEDED
            assert timeout_ms / 1000 - 0.05 <= took <= timeout_ms / 1000 + TOLERANCE
        assert not split.gateway.requests
        assert split.ready() == before
        # every request's cache is freed at the stage that kept running
        wait_for(lambda: split.cached() == 0, "the failed requests' caches to be freed")
    finally:
        split.close()
