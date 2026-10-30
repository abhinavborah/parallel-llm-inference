"""Fault injection for stage workers: kill, suspend, resume and a per-step sleep.

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
    _each(pid, psutil.Process.kill)


def suspend(pid: int) -> None:
    """Stop a process tree in place; it keeps its memory and sockets but runs no code."""
    _each(pid, psutil.Process.suspend)


def resume(pid: int) -> None:
    """Continue a process tree stopped by suspend."""
    _each(pid, psutil.Process.resume)


def _each(pid: int, action) -> None:
    # A descendant can exit after the tree snapshot. Skip it and carry on, so one
    # missing process never leaves the rest of the tree half suspended or half resumed.
    for proc in _tree(pid):
        try:
            action(proc)
        except psutil.NoSuchProcess:
            pass


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
