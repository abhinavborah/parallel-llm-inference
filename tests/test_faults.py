import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

from pllm import faults

# Child that appends one byte to a file every 10 ms, so progress is visible from outside.
TICKER = """
import sys, time
with open(sys.argv[1], "ab", buffering=0) as f:
    while True:
        f.write(b".")
        time.sleep(0.01)
"""

# Parent that runs the ticker as its child, so faults must reach descendants.
NESTED = """
import subprocess, sys
subprocess.run([sys.executable, "-c", sys.argv[1], sys.argv[2]])
"""


def _wait_for_growth(path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(0.01)
    raise AssertionError("child never started writing")


@pytest.fixture(params=["direct", "nested"])
def ticker(request, tmp_path):
    path = tmp_path / "ticks"
    if request.param == "direct":
        args = [sys.executable, "-c", TICKER, str(path)]
    else:
        args = [sys.executable, "-c", NESTED, TICKER, str(path)]
    proc = subprocess.Popen(args)
    try:
        _wait_for_growth(path)
        yield proc, path
    finally:
        if proc.poll() is None:
            faults.kill(proc.pid)
            proc.wait()


def test_kill_ends_the_process_and_its_descendants(ticker):
    proc, path = ticker
    descendants = psutil.Process(proc.pid).children(recursive=True)
    faults.kill(proc.pid)
    code = proc.wait(timeout=5)
    if sys.platform == "win32":
        assert code != 0
    else:
        assert code == -signal.SIGKILL
    psutil.wait_procs(descendants, timeout=5)
    assert not any(p.is_running() for p in descendants)

    time.sleep(0.1)
    stopped = os.path.getsize(path)
    time.sleep(0.2)
    assert os.path.getsize(path) == stopped


def test_suspend_stops_progress_and_resume_continues(ticker):
    proc, path = ticker
    faults.suspend(proc.pid)
    time.sleep(0.1)  # let any write in flight land
    frozen = os.path.getsize(path)
    time.sleep(0.3)
    assert os.path.getsize(path) == frozen
    assert proc.poll() is None

    faults.resume(proc.pid)
    time.sleep(0.3)
    assert os.path.getsize(path) > frozen


def test_suspend_and_resume_skip_a_process_that_already_exited(tmp_path, monkeypatch):
    path = tmp_path / "ticks"
    live = subprocess.Popen([sys.executable, "-c", TICKER, str(path)])
    gone = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _wait_for_growth(path)
        gone_proc = psutil.Process(gone.pid)
        gone.kill()
        gone.wait(timeout=5)
        # The exited process comes first, as if it left after the tree snapshot.
        live_tree = faults._tree(live.pid)
        monkeypatch.setattr(faults, "_tree", lambda pid: [gone_proc, *live_tree])

        faults.suspend(live.pid)
        time.sleep(0.1)
        frozen = os.path.getsize(path)
        time.sleep(0.3)
        assert os.path.getsize(path) == frozen

        faults.resume(live.pid)
        time.sleep(0.3)
        assert os.path.getsize(path) > frozen
    finally:
        for proc in (live, gone):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class RefusingProcess:
    """Refuses suspend and resume with access denied, as Windows does for a process
    that has terminated while a handle to it is still open."""

    def __init__(self, exited):
        self.exited = exited
        self.calls = []

    def suspend(self):
        self.calls.append("suspend")
        raise psutil.AccessDenied(pid=1)

    def resume(self):
        self.calls.append("resume")
        raise psutil.AccessDenied(pid=1)

    def wait(self, timeout=None):
        if not self.exited:
            raise psutil.TimeoutExpired(timeout, pid=1)
        return 1


class RecordingProcess:
    def __init__(self):
        self.calls = []

    def suspend(self):
        self.calls.append("suspend")

    def resume(self):
        self.calls.append("resume")


@pytest.mark.parametrize("action", ["suspend", "resume"])
def test_access_denied_from_an_exited_process_is_skipped(monkeypatch, action):
    refusing, live = RefusingProcess(exited=True), RecordingProcess()
    monkeypatch.setattr(faults, "_tree", lambda pid: [refusing, live])
    getattr(faults, action)(1)
    assert live.calls == [action]


@pytest.mark.parametrize("action", ["suspend", "resume"])
def test_access_denied_from_a_running_process_is_raised_after_the_rest(monkeypatch, action):
    live = RecordingProcess()
    monkeypatch.setattr(faults, "_tree", lambda pid: [RefusingProcess(exited=False), live])
    with pytest.raises(psutil.AccessDenied):
        getattr(faults, action)(1)
    assert live.calls == [action]


def test_step_delay_sleeps_for_the_configured_time():
    # perf_counter, not monotonic: on Windows with Python 3.12, monotonic ticks in 15.6 ms steps.
    delay = faults.StepDelay(0.05)
    start = time.perf_counter()
    delay()
    assert time.perf_counter() - start >= 0.05


def test_zero_step_delay_does_not_sleep():
    start = time.perf_counter()
    faults.StepDelay()()
    assert time.perf_counter() - start < 0.01


def test_negative_step_delay_is_rejected():
    with pytest.raises(ValueError):
        faults.StepDelay(-1)
