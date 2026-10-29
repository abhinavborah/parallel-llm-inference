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
