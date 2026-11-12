"""The supervisor of one laptop's worker: it spawns the worker, respawns it when it exits,
and applies kill, suspend and resume requests through the fault injector (R6.2, R6.1).

The supervisor runs the worker as a child process and checks it every poll_seconds. When the
worker exits for any reason, including a kill from the fault injector, it starts a new one
with no manual step. A worker that keeps dying within STABLE_SECONDS of starting is
restarted after a growing pause (BACKOFF_SECONDS, then the last value), so a broken worker
does not spin the CPU; the pause never grows past the last value and the supervisor never
gives up. A suspended worker is still running, so it is not restarted: noticing a stalled
worker is the gateway's job (heartbeat timeout, R6.3).

Events go to a JSON-lines file, one object per line, with `time` (wall clock of this laptop,
seconds since the epoch) and `mono` (monotonic seconds):

- spawned: pid, restart (0 for the first start)
- exited: pid, code, lived (seconds the process ran)
- ready: pid, seconds_since_spawn, and for a respawn respawn_seconds, measured from the
  moment the exit was noticed to the moment the worker accepted a connection on its port
- fault: action (kill, suspend or resume) and pid
- backoff: seconds
- spawn_failed: error (the command could not be started)
- stopped

A worker started here exits by itself if this supervisor disappears without stopping it (a
force kill, a crash): it is given the supervisor's pid in PLLM_SUPERVISOR_PID.

The respawn time includes whatever the worker does before it listens: loading its layers.
Registration with the gateway follows listening and is logged by the worker.

Fault requests arrive over HTTP on a small control port (serve_control): POST /kill,
/suspend or /resume, and GET /status. There is no authentication; it is meant for the
course's LAN. Nothing here depends on the operating system except what psutil and
subprocess already hide (R3.15).

Usage (worker of stage 1 of SmolLM2-135M, as set in the hosts file):
uv run python -m pllm.supervisor smollm2-135m --stage 1 --threads 6
"""

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

from pllm import config, faults
from pllm.models import PINNED

log = logging.getLogger("pllm.supervisor")

STABLE_SECONDS = 5.0  # a worker that lived this long did not crash on start
BACKOFF_SECONDS = (0.0, 0.5, 1.0, 2.0, 5.0)  # pause before restart 1, 2, 3 ... of a crash loop
READY_TIMEOUT_SECONDS = 180.0  # how long a new worker may take to start listening
REQUEST_TIMEOUT_SECONDS = 10.0
PARENT_ENV = "PLLM_SUPERVISOR_PID"  # the worker's guard against an orphaned run
FAULT_ACTIONS = ("kill", "suspend", "resume")


class NoWorker(RuntimeError):
    """A fault request when no worker process is running."""


class Supervisor:
    def __init__(
        self,
        command: Sequence[str],
        *,
        ready_address: tuple[str, int] | None = None,
        log_path: str | Path | None = None,
        cwd: str | Path | None = None,
        poll_seconds: float = 0.05,
        stable_seconds: float = STABLE_SECONDS,
        backoff_seconds: Sequence[float] = BACKOFF_SECONDS,
        ready_timeout: float = READY_TIMEOUT_SECONDS,
    ):
        if not command:
            raise ValueError("command is empty")
        if not backoff_seconds or any(b < 0 for b in backoff_seconds):
            raise ValueError("backoff_seconds needs non-negative values")
        self.command = list(command)
        self.ready_address = ready_address
        self.cwd = cwd
        self.poll_seconds = poll_seconds
        self.stable_seconds = stable_seconds
        self.backoff_seconds = tuple(backoff_seconds)
        self.ready_timeout = ready_timeout
        self.restarts = 0
        self.suspended = False
        self.last_respawn_seconds: float | None = None
        self._ready_at: float | None = None  # when the current worker started listening
        self._injected_kill = False  # the next exit was asked for through apply("kill")
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()  # set while the current worker is listening
        self._log_lock = threading.Lock()
        self._log_file = None
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("already started")
        self._thread = threading.Thread(target=self._run, name="supervisor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop supervising and kill the worker's process tree."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                faults.kill(proc.pid)
            except psutil.Error:
                pass
            proc.wait(timeout)
        self._event("stopped")
        with self._log_lock:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    @property
    def pid(self) -> int | None:
        with self._lock:
            proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    def wait_ready(self, timeout: float, restarts: int = 0) -> bool:
        """True once a worker that is restart number `restarts` or later is listening."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.restarts >= restarts and self._ready.is_set() and self.pid is not None:
                return True
            time.sleep(self.poll_seconds)
        return False

    # ------------------------------------------------------------- fault requests

    def apply(self, action: str) -> int:
        """Apply kill, suspend or resume to the worker through the fault injector; return
        the pid it acted on. Raises NoWorker when none is running."""
        if action not in FAULT_ACTIONS:
            raise ValueError(f"action must be one of {FAULT_ACTIONS}, got {action!r}")
        pid = self.pid
        if pid is None:
            raise NoWorker("no worker process is running")
        if action == "kill":
            self._injected_kill = True  # this exit is asked for, not a crash
        try:
            getattr(faults, action)(pid)
        except psutil.NoSuchProcess as err:
            raise NoWorker("the worker exited before the request reached it") from err
        except psutil.Error:
            self._injected_kill = False
            raise
        if action == "suspend":
            self.suspended = True
        elif action == "resume":
            self.suspended = False
        self._event("fault", action=action, pid=pid)
        return pid

    def status(self) -> dict:
        pid = self.pid
        if pid is None:
            state = "stopped" if self._stop.is_set() else "starting"
        else:
            state = "suspended" if self.suspended else "running"
        return {
            "state": state,
            "pid": pid,
            "restarts": self.restarts,
            "ready": self._ready.is_set(),
            "last_respawn_seconds": self.last_respawn_seconds,
            # this laptop's memory, for the dashboard (R9.4)
            "free_ram_bytes": psutil.virtual_memory().available,
        }

    # ------------------------------------------------------------------ the loop

    def _run(self) -> None:
        crashes = 0  # restarts in a row of workers that did not last stable_seconds
        detected: float | None = None  # when the last exit was noticed
        while not self._stop.is_set():
            self._ready = threading.Event()
            self._ready_at = None
            self._injected_kill = False
            self.suspended = False
            spawned = time.monotonic()
            try:
                proc = subprocess.Popen(self.command, cwd=self.cwd, env=self._child_env())
            except OSError as err:
                self._event("spawn_failed", error=str(err))
                log.error("cannot start %s: %s", self.command, err)
                crashes += 1
                if self._pause(crashes):
                    return
                continue
            with self._lock:
                self._proc = proc
            self._event("spawned", pid=proc.pid, restart=self.restarts)
            self._watch_start(proc, spawned, detected)
            while proc.poll() is None and not self._stop.is_set():
                time.sleep(self.poll_seconds)
            if self._stop.is_set():
                return
            detected = time.monotonic()
            lived = detected - spawned
            self._event("exited", pid=proc.pid, code=proc.returncode, lived=round(lived, 3))
            log.warning(
                "worker %d exited with code %s after %.1f s", proc.pid, proc.returncode, lived
            )
            # A worker is stable if a fault request ended it, or if it had been listening for
            # stable_seconds. One that dies while loading, or soon after it starts listening
            # (for example on a rejected registration), is a crash, however long it loaded.
            ready_at = self._ready_at
            stable = self._injected_kill or (
                ready_at is not None and detected - ready_at >= self.stable_seconds
            )
            crashes = 0 if stable else crashes + 1
            self.restarts += 1
            if crashes and self._pause(crashes):
                return

    @staticmethod
    def _child_env() -> dict[str, str]:
        """The worker learns its supervisor's pid, so it can exit if the supervisor is killed
        and cannot stop it (see worker.exit_when_parent_gone)."""
        return {**os.environ, PARENT_ENV: str(os.getpid())}

    def _pause(self, crashes: int) -> bool:
        """Wait before the next start of a crash loop; True if the supervisor was stopped."""
        pause = self.backoff_seconds[min(crashes, len(self.backoff_seconds)) - 1]
        if pause:
            self._event("backoff", seconds=pause)
        return self._stop.wait(pause)

    def _watch_start(self, proc, spawned: float, detected: float | None) -> None:
        """Report when the new worker accepts connections, from a thread so the loop above
        keeps noticing an exit."""
        if self.ready_address is None:
            self._ready_at = spawned
            self._ready.set()
            return
        ready = self._ready

        def probe():
            deadline = spawned + self.ready_timeout
            while not self._stop.is_set() and proc.poll() is None:
                if time.monotonic() > deadline:
                    # A worker that never listens is stuck: kill it, so it is restarted.
                    self._event("ready_timeout", pid=proc.pid)
                    try:
                        faults.kill(proc.pid)
                    except psutil.Error:
                        pass
                    return
                try:
                    socket.create_connection(self.ready_address, timeout=0.2).close()
                except OSError:
                    time.sleep(self.poll_seconds)
                    continue
                now = time.monotonic()
                fields = {"pid": proc.pid, "seconds_since_spawn": round(now - spawned, 3)}
                if detected is not None:
                    self.last_respawn_seconds = round(now - detected, 3)
                    fields["respawn_seconds"] = self.last_respawn_seconds
                    log.info("worker back after %.2f s", self.last_respawn_seconds)
                self._ready_at = now
                self._event("ready", **fields)  # logged before waiters are released
                ready.set()
                return

        threading.Thread(target=probe, name="ready-probe", daemon=True).start()

    def _event(self, event: str, **fields) -> None:
        record = {"event": event, "time": time.time(), "mono": round(time.monotonic(), 3)}
        record.update(fields)
        with self._log_lock:
            if self._log_file is not None:
                self._log_file.write(json.dumps(record) + "\n")
                self._log_file.flush()
        log.info("%s", record)


# ---------------------------------------------------------------------- control port


def serve_control(supervisor: Supervisor, bind: tuple[str, int]) -> ThreadingHTTPServer:
    """Serve fault requests on bind (port 0 picks a free one) from a background thread.

    POST /kill, /suspend, /resume answer 200 with {"ok": true, "action", "pid"}, 409 with
    {"ok": false, "error"} when no worker is running, or 500 if the operating system refuses.
    A POST must carry Content-Type: application/json, which a web page cannot send to
    another host without a preflight that this server never answers, so a page open on the
    lab network cannot kill a worker. GET /status answers the supervisor's state. Anything
    else is 404; a connection that stalls is dropped after REQUEST_TIMEOUT_SECONDS.
    """

    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_SECONDS

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/status":
                self._send(200, supervisor.status())
            else:
                self._send(404, {"ok": False, "error": f"no such path {self.path}"})

        def do_POST(self):  # noqa: N802
            action = self.path.lstrip("/")
            if action not in FAULT_ACTIONS:
                self._send(404, {"ok": False, "error": f"no such path {self.path}"})
                return
            kind = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if kind != "application/json":
                self._send(415, {"ok": False, "error": "send Content-Type: application/json"})
                return
            try:
                pid = supervisor.apply(action)
            except NoWorker as err:
                self._send(409, {"ok": False, "error": str(err)})
            except psutil.Error as err:
                self._send(500, {"ok": False, "error": f"{type(err).__name__}: {err}"})
            else:
                self._send(200, {"ok": True, "action": action, "pid": pid})

        def log_message(self, format, *args):  # noqa: A002
            log.debug("control: " + format, *args)

    server = ThreadingHTTPServer(bind, Handler)
    threading.Thread(target=server.serve_forever, name="control", daemon=True).start()
    return server


# ------------------------------------------------------------------------------ CLI


def worker_command(model: str, stage: int, threads: int | None) -> list[str]:
    command = [sys.executable, "-m", "pllm.worker", model, "--stage", str(stage), "--serve"]
    if threads is not None:
        command += ["--threads", str(threads)]
    return command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Supervise one stage worker.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--stage", type=int, choices=(0, 1), required=True)
    parser.add_argument("--threads", type=int, help="torch threads for the worker")
    parser.add_argument("--log", help="JSON-lines event log (default runs/supervisor_stageN.jsonl)")
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    try:
        hosts = config.load()
        control = config.supervisor_endpoint(args.stage)
    except (ValueError, OSError) as err:
        parser.error(str(err))
    stage_port = getattr(hosts, f"stage{args.stage}").port
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    supervisor = Supervisor(
        worker_command(args.model, args.stage, args.threads),
        ready_address=("127.0.0.1", stage_port),
        log_path=args.log or config.REPO_ROOT / "runs" / f"supervisor_stage{args.stage}.jsonl",
        cwd=config.REPO_ROOT,
    )
    try:
        server = serve_control(supervisor, (control.host, control.port))
    except OSError as err:
        parser.error(f"cannot listen on {control.address}: {err}")
    done = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):  # SIGBREAK exists on Windows only
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: done.set())
    supervisor.start()
    log.info("supervising stage %d; control %s; Ctrl-C to stop", args.stage, control.address)
    while not done.wait(0.5):  # a timeout, so a signal handler can run while this waits
        pass
    supervisor.stop()  # first: no new worker is started while the control port closes
    server.shutdown()


if __name__ == "__main__":
    main()
