"""Fault injection: kill, suspend, resume, a per-step sleep and a per-message hop delay.

Process control goes through psutil so the same calls work on macOS and Windows.
Each supervisor applies these to the worker it owns.

Every call acts on the process and all its descendants. On Windows the venv
python.exe is a launcher that runs the real interpreter as a child, and
suspending only the launcher leaves the worker running.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from pllm import config

log = logging.getLogger("pllm.faults")


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
    log: deque | None = field(default=None, compare=False, repr=False)  # where held() notes

    def __post_init__(self) -> None:
        if not 0 <= self.seconds < float("inf"):
            raise ValueError(f"hop delay must be finite and non-negative, got {self.seconds}")

    def release(self, message, now: float) -> float:
        """When a message that entered the hop at now (monotonic seconds) may leave it."""
        return now + self.seconds

    def held(self, message, entered: float, left: float) -> None:
        """Note how long a message was really held, on this process's clock: a late wake-up
        makes it longer than seconds, and pllm.x4 reports it next to the requested delay.
        A log entry is (rid, attempt, step, requested seconds, held seconds, entered)."""
        if self.log is not None:
            entry = (message.rid, message.attempt, message.step, self.seconds, left - entered)
            self.log.append((*entry, entered))


def sleep_until(release: float, give_up=None, poll: float = 0.05) -> bool:
    """Sleep until the monotonic time release; False if give_up() turned true first.
    The wait is cut into poll steps so a stream that ends is noticed while it waits."""
    while (left := release - time.monotonic()) > 0:
        if give_up is not None and give_up():
            return False
        time.sleep(min(left, poll))
    return True


HOP_DELAY_FILE = config.REPO_ROOT / "runs" / "hop_delay_ms"
CONTROL_SECONDS = 0.2  # how often a process with the control on reads the file


CONTROLLED = ("gateway", "stage1")  # the two processes that delay a hop, both on Host A


def ack_path(path: Path, name: str) -> Path:
    """Where the process called name writes the delay it applied."""
    return path.with_name(f"{path.name}.{name}")


def ack_paths(path: Path) -> list[Path]:
    return [ack_path(path, name) for name in CONTROLLED]


def held_path(path: Path, name: str) -> Path:
    """Where the process called name appends one JSON line per message it held."""
    return path.with_name(f"{path.name}.{name}.held.jsonl")


def read_delay_ms(path: Path) -> float | None:
    """The delay in the control file, in ms: 0 if there is no file, None if it cannot be
    read or is not a finite non-negative number."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0.0
    except OSError:
        return None  # Windows: being replaced right now
    try:
        value = float(text)
    except ValueError:
        return None
    return value if 0 <= value < float("inf") else None


def write_atomic(path: Path, text: str, tries: int = 20) -> None:
    """Replace path with text in one step, so a reader never sees half a file. On Windows
    the replace fails while another process has the file open, so it is retried."""
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    for attempt in range(tries):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == tries - 1:
                temp.unlink(missing_ok=True)
                raise
            time.sleep(0.01)


class HopDelayControl:
    """Sets a hop delay at run time from a file on this laptop, for the X4 sweep (R7.11).

    Only a process started with the control flag runs one, so a file left behind never
    delays a normal run. Every interval it reads the file (a number of ms; no file is 0);
    on a change it calls apply(HopDelay) and then writes the value it applied to its ack
    file, which pllm.x4 waits for. A file it cannot read or parse leaves the delay as it
    is and is logged once. Every delay it applies notes each message's real hold in one log,
    which each interval appends to its held file (held_path) for pllm.x4 to read.
    """

    def __init__(self, name: str, apply, path: Path = HOP_DELAY_FILE, interval=CONTROL_SECONDS):
        self.name, self.apply, self.path, self.interval = name, apply, Path(path), interval
        self.applied_ms: float | None = None
        self.log: deque = deque()  # appended by the hop's threads, emptied by flush()
        self._warned = False  # log a bad file once, not every poll

    def poll(self) -> None:
        """Read the file once and apply a change."""
        value = read_delay_ms(self.path)
        if value is None:
            if not self._warned:
                log.warning("hop delay file %s is not a number of ms; delay kept", self.path)
                self._warned = True
            return
        self._warned = False
        if value == self.applied_ms:
            return
        self.apply(HopDelay(value / 1000, self.log))
        self.applied_ms = value
        log.info("%s: hop delay now %g ms", self.name, value)
        write_atomic(ack_path(self.path, self.name), f"{value!r}\n")

    def flush(self) -> None:
        """Append the holds noted since the last flush to the held file, one line each."""
        lines = []
        while self.log:
            rid, attempt, step, seconds, held, _ = self.log.popleft()
            row = {"rid": rid, "attempt": attempt, "step": step, "ms": seconds * 1e3}
            lines.append(json.dumps({**row, "held_ms": held * 1e3}) + "\n")
        if lines:
            with held_path(self.path, self.name).open("a", encoding="utf-8") as out:
                out.writelines(lines)

    def start(self, stop: threading.Event) -> threading.Thread:
        """Apply the file now, then poll it on a daemon thread until stop is set."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.poll()

        def run():
            while not stop.wait(self.interval):
                try:
                    self.poll()
                    self.flush()
                except OSError as err:  # an ack not written this time is written next time
                    log.warning("hop delay control: %s", err)

        thread = threading.Thread(target=run, daemon=True, name=f"hop-delay-{self.name}")
        thread.start()
        return thread
