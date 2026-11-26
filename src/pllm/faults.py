"""Fault injection: kill, suspend, resume, a per-step sleep and a per-message hop delay.

Process control goes through psutil so the same calls work on macOS and Windows.
Each supervisor applies these to the worker it owns.

Every call acts on the process and all its descendants. On Windows the venv
python.exe is a launcher that runs the real interpreter as a child, and
suspending only the launcher leaves the worker running.
"""

import time
from dataclasses import dataclass

import psutil


def _tree(pid: int) -> list[psutil.Process]:
    root = psutil.Process(pid)
    return [root, *root.children(recursive=True)]


def kill(pid: int) -> None:
    """Kill a process tree with no chance to clean up.

    psutil sends SIGKILL on macOS and Linux and calls TerminateProcess on Windows.
    """
    _each(pid, "kill")


def suspend(pid: int) -> None:
    """Stop a process tree in place; it keeps its memory and sockets but runs no code."""
    _each(pid, "suspend")


def resume(pid: int) -> None:
    """Continue a process tree stopped by suspend."""
    _each(pid, "resume")


def _each(pid: int, action: str) -> None:
    # A descendant can exit after the tree snapshot. Skip it and carry on, so one
    # missing process never leaves the rest of the tree half suspended or half resumed.
    # A real refusal is raised only after every other process has been handled.
    refused = []
    for proc in _tree(pid):
        try:
            getattr(proc, action)()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as err:
            # Windows can refuse to suspend or resume a process that has terminated but
            # whose handle is still open, with access denied instead of no such process.
            # A process that is still running and refuses is a real failure.
            if not _exited(proc):
                refused.append(err)
    if refused:
        raise refused[0]


def _exited(proc: psutil.Process) -> bool:
    # A short wait, not zero: Windows can still list a PID for a moment after the
    # process has ended. On macOS and Linux this path is not reached for our own
    # workers, since signalling an exited child succeeds.
    try:
        proc.wait(timeout=0.1)
    except psutil.TimeoutExpired:
        return False
    except psutil.NoSuchProcess:
        pass
    return True


@dataclass(frozen=True)
class StepDelay:
    """Sleep injected before each step to simulate a slow worker. Zero disables it."""

    seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError(f"step delay must be non-negative, got {self.seconds}")

    def __call__(self) -> None:
        if self.seconds > 0:
            time.sleep(self.seconds)


@dataclass(frozen=True)
class HopDelay:
    """Delay added to each message of one hop (R7.7, decision #48). Zero adds none.

    Pipelined, not blocking like StepDelay: each message is released seconds after it
    entered the hop, so messages that enter together leave together and the stream's
    throughput is not capped at one message per delay. release() is asked once per
    message, so a later fault can choose per message (a longer delay, or a drop).
    """

    seconds: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.seconds < float("inf"):
            raise ValueError(f"hop delay must be finite and non-negative, got {self.seconds}")

    def release(self, message, now: float) -> float:
        """When a message that entered the hop at now (monotonic seconds) may leave it."""
        return now + self.seconds


def sleep_until(release: float, give_up=None, poll: float = 0.05) -> bool:
    """Sleep until the monotonic time release; False if give_up() turned true first.
    The wait is cut into poll steps so a stream that ends is noticed while it waits."""
    while (left := release - time.monotonic()) > 0:
        if give_up is not None and give_up():
            return False
        time.sleep(min(left, poll))
    return True
