"""Both stages failing at once (R6.14).

Both stage processes are killed together and new ones start together, as the two hosts'
supervisors would start them. Both deaths are declared, every request is bumped at each
declaration, and the replay starts once both stages have registered again. With full replay
a single failure already replays through both stages, so the extra cost of a double failure
is the wait for the slower of two parallel restarts, plus contention while both start.
"""

import subprocess
import sys
import time

import grpc
import pytest
from test_gateway import IDS, WAIT, wait_for
from test_heartbeat import Beating, reference
from test_model import tiny_reference
from test_replay import BEAT, LAYERS, K, take

from pllm import faults
from pllm.control import ExpectedModel
from pllm.models import sha256_of

pytestmark = pytest.mark.fault

STEP = 0.02  # seconds each child stage sleeps per step, so a request is mid-decode at a kill
N = 16

# test_replay.CHILD with a step delay; Stage 0 starts with a placeholder neighbour that the
# route update replaces, since the Stage 1 child's port is not known yet
CHILD = """
import sys, threading
from pathlib import Path
from pllm.stage_service import serve
from pllm.worker import StageWorker, WorkerConfig, join_gateway
model_dir, stage, k, n, gateway, beat, step = sys.argv[1:8]
stage, k, n = int(stage), int(k), int(n)
layers = (0, k) if stage == 0 else (k, n)
where = {"next_address": "127.0.0.1:1"} if stage == 0 else {"results_address": gateway}
worker = StageWorker(WorkerConfig(Path(model_dir), *layers, step_delay=float(step)))
server, servicer, port = serve(worker, "127.0.0.1:0", **where)
join_gateway(gateway, stage, layers, "tiny", Path(model_dir), f"127.0.0.1:{port}", servicer,
             threading.Event(), interval=float(beat))
"""


class Children:
    """Both stages as real child processes of one gateway in this process."""

    def __init__(self, run, model_dir):
        self.run, self.model_dir = run, model_dir
        self.all: list[subprocess.Popen] = []
        self.current = {stage: self.spawn(stage) for stage in (0, 1)}
        self.wait_up()

    def spawn(self, stage):
        args = [str(self.model_dir), str(stage), str(K), str(LAYERS), self.run.address]
        child = subprocess.Popen([sys.executable, "-c", CHILD, *args, str(BEAT), str(STEP)])
        self.all.append(child)
        return child

    def wait_up(self):
        """Wait until both stages are registered and alive, then until a request completes:
        one sent before the route updates land is refused by fencing."""
        wait_for(lambda: self.run.gateway._down() is None, "both stages up")
        until = time.monotonic() + WAIT
        while True:
            try:
                list(self.run.call(max_tokens=1))
                return
            except grpc.RpcError:
                assert time.monotonic() < until, "no request completed"
                time.sleep(BEAT)

    def kill(self, stages):
        """Kill the stages together, wait for each death to be declared, then start their
        replacements together. Returns the old epochs."""
        gateway = self.run.gateway
        old = {stage: self.run.registry.stages[stage].epoch for stage in stages}
        for stage in stages:
            faults.kill(self.current[stage].pid)
        for stage in stages:
            self.current[stage].wait(WAIT)
        for stage in stages:
            wait_for(lambda s=stage: gateway.detector.is_dead(old[s]), "the death declared")
        for stage in stages:
            self.current[stage] = self.spawn(stage)
        return old

    def close(self):
        for child in self.all:
            if child.poll() is None:
                faults.kill(child.pid)
            child.wait(WAIT)


@pytest.fixture
def pipeline(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    run = Beating(tmp_path, K, interval=BEAT, order=())  # no stage in this process
    # the children register the real weights' digest, not the stand-in the harness used
    real = bytes.fromhex(sha256_of(tmp_path / "model.safetensors"))
    run.registry.expected = ExpectedModel(real, LAYERS, K)
    events = []
    write = run.gateway.recorder.event

    def event(record):
        events.append(record)
        write(record)

    run.gateway.recorder.event = event
    children = None
    try:
        children = Children(run, tmp_path)
        yield run, children, events
    finally:
        if children is not None:
            children.close()
        run.close()


def recover(run, children, events, stages, expected):
    """One fresh request, the stages killed mid-decode: check the client gets the fault-free
    tokens, each step once, and return the request's recovery line and the old epochs."""
    children.wait_up()
    before = len(events)
    stream = run.call(max_tokens=N)
    tokens = take(stream, 3)
    old = children.kill(stages)
    tokens += list(stream)
    assert [t.token for t in tokens] == expected
    assert [t.step for t in tokens] == list(range(N))
    rid = tokens[0].rid

    def lines():
        return [e for e in events[before:] if e["event"] == "recovery" and e["rid"] == rid]

    wait_for(lines, "the recovery line")
    [line] = lines()
    assert line["recovered"] and line["divergences"] == 0
    return line, old


def test_both_stages_killed_together_recover_and_the_request_finishes_unchanged(pipeline, tmp_path):
    """The replay waits for both new registrations, so the recovery lasts at least as long
    as each stage's own restart: what a single failure of that stage costs at least."""
    run, children, events = pipeline
    expected = reference(tmp_path, K, IDS, N)[0]
    line, old = recover(run, children, events, (0, 1), expected)
    assert run.gateway.divergences == 0
    for stage in (0, 1):
        assert run.registry.stages[stage].epoch > old[stage]
    assert line["attempt"] >= 3  # bumped at each declaration; a fencing refusal may add one

    def one(kind, stage, key, epoch):
        return next(e for e in events if e["event"] == kind and e.get(key) == epoch
                    and e["stage"] == stage and e.get("declared", True))  # fmt: skip

    deaths = {s: one("detected", s, "epoch", old[s]) for s in (0, 1)}
    registrations = {s: one("registered", s, "old_epoch", old[s]) for s in (0, 1)}
    assert line["t_ns"] >= max(r["t_ns"] for r in registrations.values())
    first_beat = min(d["last_beat_ns"] for d in deaths.values())
    for stage in (0, 1):
        restart = registrations[stage]["t_ns"] - deaths[stage]["last_beat_ns"]
        assert line["t_ns"] - first_beat >= restart > 0
    # the line names one of the two deaths: the one the gateway handled last, usually the
    # later declared, though the two declarations can reach it in either order
    assert line["old_epoch"] == old[line["stage"]]
