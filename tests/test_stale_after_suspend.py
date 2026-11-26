"""A stage suspended past the heartbeat timeout, then resumed after its replacement serves,
is harmless (R6.8).

The stage runs as a real process. It is suspended inside a step, so it holds an output it
has not sent yet: the process marks a file as each step begins, and is suspended only after
a new mark, so a slow runner cannot suspend it before the step has reached it. Its epoch is
declared dead by the heartbeat timeout, a new process takes its place, and only then is the
old one resumed. It is told it was superseded and keeps serving, as a worker does, so it
sends that late output:

- an old Stage 1 pushes its result to the gateway, whose result filter drops it;
- an old Stage 0 forwards its hidden states to Stage 1 with its old sender epoch, and Stage 1
  refuses them with BAD_SENDER. The Nack names an old attempt, so it fails nothing.

The client sees each step once and the fault-free tokens. Equality is on one host, fp32,
batch 1.
"""

import subprocess
import sys

import pytest
from test_gateway import IDS, WAIT, wait_for
from test_heartbeat import Beating, reference
from test_model import tiny_reference
from test_replay import take

from pllm import faults, metrics
from pllm.control import ExpectedModel
from pllm.models import sha256_of

LAYERS = 4
K = 2
BEAT = 0.05
TIMEOUT = 1.0  # heartbeat timeout: 20 beats, so a slow runner does not declare a live stage
OLD_DELAY = 0.4  # each step of the old process, so the suspend lands inside one
N = 32  # tokens, enough that the request still runs when the old process resumes

CHILD = """
import sys, threading
from pathlib import Path
from pllm.stage_service import serve
from pllm.worker import StageWorker, WorkerConfig, join_gateway
model_dir, stage, k, n, gateway, neighbour, interval, delay, marks = sys.argv[1:10]
stage, k, n = int(stage), int(k), int(n)
layers = (0, k) if stage == 0 else (k, n)
where = {"next_address": neighbour} if stage == 0 else {"results_address": gateway}
config = WorkerConfig(Path(model_dir), *layers, float(delay))
worker = StageWorker(config)
sleep = worker.delay


def delay_after_a_mark():  # a step has reached this process and its delay begins
    with open(marks, "a") as file:
        file.write(".")
    sleep()


worker.delay = delay_after_a_mark
server, servicer, port = serve(worker, "127.0.0.1:0", **where)
join_gateway(gateway, stage, layers, "tiny", Path(model_dir), f"127.0.0.1:{port}", servicer,
             threading.Event(), interval=float(interval))
server.wait_for_termination()  # superseded: no more beats, but it serves on, as main does
"""


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


def outputs_of(run, stage):
    """Record (epoch, attempt, step, rejected) of each output of the stage under test, where
    it lands: the results the gateway receives from Stage 1, or the steps Stage 1 receives
    from Stage 0 (the epoch is then the sender epoch)."""
    seen = []
    if stage == 1:
        real = run.gateway.recorder.hop

        def hop(name, message, accepted=True):
            if name == metrics.STAGE1_GATEWAY:
                seen.append((message.epoch, message.attempt, message.step, not accepted))
            real(name, message, accepted=accepted)

        run.gateway.recorder.hop = hop
    else:
        real = run.servicer1._handle

        def handle(request, context, next_epoch, deadline=None):
            out = real(request, context, next_epoch, deadline)
            seen.append((request.sender_epoch, request.attempt, request.step, out is None))
            return out

        run.servicer1._handle = handle
    return seen


@pytest.mark.parametrize("stage", [0, 1], ids=["stage0", "stage1"])
def test_a_resumed_old_stage_has_every_late_output_rejected(beating, tmp_path, stage):
    run = beating(order=(1 - stage,), delay0=0.05, delay1=0.05)  # the other stage stays here
    run.gateway.detector.heartbeat_timeout_ns = round(TIMEOUT * 1e9)
    # the children register the real weights' digest, not the stand-in the harness used
    real = bytes.fromhex(sha256_of(tmp_path / "model.safetensors"))
    run.registry.expected = ExpectedModel(real, LAYERS, K)
    neighbour = "" if stage == 1 else f"127.0.0.1:{run.ports[1]}"
    outputs = outputs_of(run, stage)

    def spawn(delay, marks):
        args = [str(tmp_path), str(stage), str(K), str(LAYERS), run.address, neighbour]
        rest = [str(BEAT), str(delay), str(marks)]
        return subprocess.Popen([sys.executable, "-c", CHILD, *args, *rest])

    old_marks = tmp_path / "old.marks"

    def begun():  # steps the old process has begun
        return old_marks.stat().st_size if old_marks.exists() else 0

    def routed(epoch):  # the neighbour in this process has the stage's route update
        if stage == 1:
            return run.servicer0.next_epoch == epoch
        return run.servicer1.upstream_epoch == epoch

    children, suspended = [spawn(OLD_DELAY, old_marks)], None
    old_proc = children[0]
    try:
        stages = run.registry.stages
        wait_for(lambda: stage in stages and stage in run.gateway.beats(), "the old beat")
        old = stages[stage].epoch
        wait_for(lambda: routed(old), "the route update to the old process")
        expected = reference(tmp_path, K, IDS, N)[0]
        stream = run.call(max_tokens=N)
        tokens = take(stream, 2)
        before = begun()
        wait_for(lambda: begun() > before, "the old process to begin its next step")
        faults.suspend(old_proc.pid)  # within its delay: the output is not sent yet
        suspended = old_proc

        wait_for(lambda: run.gateway.detector.is_dead(old), "the old epoch to be declared dead")
        events = run.gateway.detector.events()
        [death] = [e for e in events if e["event"] == "detected" and e["epoch"] == old]
        assert death["path"] == "heartbeat_timeout" and death["declared"]
        declared = len(outputs)  # the old process sends nothing from here until it resumes

        children.append(spawn(0.0, tmp_path / "new.marks"))
        wait_for(lambda: stages[stage].epoch > old, "the replacement's registration")
        new = stages[stage].epoch
        wait_for(lambda: routed(new), "the route update to the replacement")
        wait_for(
            lambda: any(e == new and not rejected for e, *_, rejected in outputs),
            "the replacement to serve",
        )
        tokens += take(stream, 1)

        faults.resume(old_proc.pid)
        suspended = None

        def late():
            return [o for o in outputs[declared:] if o[0] == old]

        wait_for(late, "a late output from the resumed old process")
        assert old_proc.poll() is None  # alive and revived, yet harmless
        if stage == 0:  # Stage 1's refusal reaches the gateway and fails nothing
            attempt = late()[0][1]
            wait_for(
                lambda: any(
                    (n.stage, n.reason, n.attempt) == (1, "BAD_SENDER", attempt)
                    for n in list(run.gateway.nacks)
                ),
                "Stage 1's BAD_SENDER Nack at the gateway",
            )
            assert run.gateway.discards["stale_nack"] >= 1
        tokens += list(stream)

        assert all(rejected for *_, rejected in late())  # every late output rejected
        assert [t.step for t in tokens] == list(range(N))  # each step once: no duplicate
        assert [t.token for t in tokens] == expected
        assert run.gateway.replays >= 1 and run.gateway.divergences == 0
        assert stages[stage].epoch == new
    finally:
        if suspended is not None:
            faults.resume(suspended.pid)
        for child in children:
            if child.poll() is None:
                faults.kill(child.pid)
            child.wait(WAIT)
