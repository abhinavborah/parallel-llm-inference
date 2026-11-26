"""The Host B side of X9: reads the stage worker's memory counters on request (R8.7).

It runs on Host B beside the supervisor. It asks the supervisor's /status for the worker's
pid and reads that process tree's counters (pllm.procmem). pllm.x9 on Host A calls it over
HTTP at the edges of its measured windows, so every memory number is read on Host B and no
clocks are compared. It stays up while the stages are restarted for each split point.

- GET /snapshot: the counters now.
- GET /start: begin a window: sample the tree's resident set every --interval seconds
  from now, and answer the counters now.
- GET /stop: end the window and answer the counters now with the largest sample, the
  number of samples and the longest gap between two. A sampled maximum can miss a peak
  shorter than the gap, so it is a lower bound on the window's peak.

Every answer also carries the worker's own command line arguments, the commit of this
checkout, and the monitor's own resident set: it is a second process on Host B, outside
the worker's budget, and it imports no torch. It answers 503 with the reason when there
is no worker to read. It has no authentication and changes nothing on the host, but it
tells whoever can reach it about the worker: use it only on a network you trust.

Usage (Host B, after the launcher, with the same PLLM_CONFIG):
uv run python -m pllm.x9monitor
"""

import argparse
import json
import logging
import socket
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

from pllm import config, procmem, runrecord

log = logging.getLogger("pllm.x9monitor")

PORT = 50071
INTERVAL_S = 0.05
REQUEST_TIMEOUT_SECONDS = 10.0
WORKER_MODULE = "pllm.worker"


class NoWorker(RuntimeError):
    """There is no worker process to read."""


def supervisor_status(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def worker_args(pid: int) -> list[str] | None:
    """The worker's arguments after 'pllm.worker', or None if they cannot be read. Only
    these are reported: the interpreter's path before them names folders on this laptop."""
    try:
        command = psutil.Process(pid).cmdline()
    except psutil.Error:
        return None
    if WORKER_MODULE not in command:
        return None
    return command[command.index(WORKER_MODULE) + 1 :]


class Monitor:
    """status() gives the supervisor's state; snapshot and args read a pid, and
    reader(pid) gives the function a window samples."""

    def __init__(
        self,
        status: Callable[[], dict],
        *,
        interval: float = INTERVAL_S,
        commit: str | None = None,
        snapshot: Callable[[int], dict] = procmem.tree_snapshot,
        reader: Callable[[int], Callable[[], int]] = procmem.rss_reader,
        args: Callable[[int], list[str] | None] = worker_args,
    ):
        self.status, self.interval, self.commit = status, interval, commit
        self._snapshot, self._reader, self._args = snapshot, reader, args
        self._lock = threading.Lock()  # one window at a time
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._sample: dict = {}

    def _worker(self) -> tuple[int, int | None]:
        try:
            status = self.status()
        except (OSError, ValueError) as err:
            raise NoWorker(f"the supervisor cannot be reached: {err}") from None
        if not isinstance(status.get("pid"), int):
            raise NoWorker("the supervisor has no worker process")
        return status["pid"], status.get("restarts")

    def snapshot(self) -> dict:
        pid, restarts = self._worker()
        try:
            counters = self._snapshot(pid)
        except psutil.Error as err:
            raise NoWorker(f"worker {pid} cannot be read: {err}") from None
        return {
            "host": socket.gethostname(),
            "platform": sys.platform,
            "commit": self.commit,
            "pid": pid,
            "restarts": restarts,
            "worker_args": self._args(pid),
            **counters,
            "disk_fault_counter": procmem.disk_counter_name(),
            "available_memory_bytes": psutil.virtual_memory().available,
            "swap_used_bytes": psutil.swap_memory().used,
            "monitor_rss_bytes": psutil.Process().memory_info().rss,
        }

    def _watch(self, read: Callable[[], int], done: threading.Event, sample: dict) -> None:
        last = time.monotonic()
        while True:
            try:
                size = read()
            except psutil.Error:
                sample["read_errors"] += 1
            else:
                sample["rss_max_bytes"] = max(sample["rss_max_bytes"] or 0, size)
                sample["samples"] += 1
                now = time.monotonic()
                sample["max_gap_s"] = max(sample["max_gap_s"], now - last)
                last = now
            if done.wait(self.interval):
                return

    def _end_sampling(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def start(self) -> dict:
        """Begin a window; a window already open is dropped."""
        with self._lock:
            self._end_sampling()
            now = self.snapshot()
            try:
                read = self._reader(now["pid"])
            except psutil.Error as err:
                raise NoWorker(f"worker {now['pid']} cannot be read: {err}") from None
            self._done = threading.Event()
            self._sample = {
                "pid": now["pid"],
                "rss_max_bytes": None,
                "samples": 0,
                "read_errors": 0,
                "max_gap_s": 0.0,
                "interval_s": self.interval,
            }
            self._thread = threading.Thread(
                target=self._watch,
                args=(read, self._done, self._sample),
                name="x9-sampler",
                daemon=True,
            )
            self._thread.start()
            return now

    def stop(self) -> dict:
        with self._lock:
            if self._thread is None:
                raise NoWorker("no window is open")
            self._end_sampling()
            return {**self.snapshot(), "window": dict(self._sample)}


def serve(monitor: Monitor, bind: tuple[str, int]) -> ThreadingHTTPServer:
    """Serve the monitor on bind (port 0 picks a free one) from a background thread."""
    actions = {"/snapshot": monitor.snapshot, "/start": monitor.start, "/stop": monitor.stop}

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
            action = actions.get(self.path)
            if action is None:
                self._send(404, {"error": f"no such path {self.path}"})
                return
            try:
                body = action()
            except NoWorker as err:
                self._send(503, {"error": str(err)})
            else:
                self._send(200, body)

        def log_message(self, format, *args):  # noqa: A002
            log.debug("monitor: " + format, *args)

    server = ThreadingHTTPServer(bind, Handler)
    threading.Thread(target=server.serve_forever, name="x9-monitor", daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="X9: read the stage worker's memory counters.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--interval", type=float, default=INTERVAL_S, help="seconds per sample")
    args = parser.parse_args(argv)
    if not 0 < args.interval <= 1:
        parser.error("--interval must be above 0 and at most 1 second")
    try:
        hosts = config.load()
        supervisor = config.supervisor_endpoint(args.stage)
    except (ValueError, OSError) as err:
        parser.error(str(err))
    host = getattr(hosts, f"stage{args.stage}").host
    url = f"http://{supervisor.address}/status"
    monitor = Monitor(
        lambda: supervisor_status(url), interval=args.interval, commit=runrecord.commit_hash()
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        server = serve(monitor, (host, args.port))
    except OSError as err:
        parser.error(f"cannot listen on {host}:{args.port}: {err}")
    log.info("serving on %s:%d, reading the worker of %s; Ctrl-C to stop", host, args.port, url)
    try:
        while True:
            time.sleep(0.5)  # short sleeps, so Ctrl-C is handled on Windows too
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
