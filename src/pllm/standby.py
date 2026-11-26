"""A standby supervisor: a replacement for a suspended stage, started on request (X6, R6.8).

A supervisor never restarts a suspended worker, since it is still running, and that worker
keeps its stage's port. So while a stage is suspended nothing replaces it. This process, on
the same laptop as the stage's supervisor, starts a second supervised worker of that stage
when asked and stops it when asked. The replacement listens on the stage's port plus
PORT_OFFSET and registers that address, so the gateway and the other stage follow it as
they follow any new registration. The worker and the supervisor are the usual ones: only
the replacement's hosts file differs, in its own stage's port. It is written to
runs/standby_stageN.toml from the hosts file this process reads.

It is idle until asked, and loads no weights while idle. It takes requests over HTTP on the
stage's supervisor port plus PORT_OFFSET: POST /start (409 if a worker is already there),
POST /stop (which also succeeds when idle) and GET /status ({"state": "idle"}, or the
replacement supervisor's status). As at the supervisor, a POST must carry
Content-Type: application/json, and there is no authentication.

--threads must be the torch thread count of the stage it replaces (give the launcher
--threads and pass the same number here), so that on one host a replayed token can be
compared bit for bit.

Usage (on the laptop of stage 1; pllm.x6 sends the requests):
uv run python -m pllm.standby smollm2-135m --stage 1 --threads 6
"""

import argparse
import json
import logging
import os
import signal
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

from pllm import config
from pllm.models import MODELS_DIR, PINNED
from pllm.supervisor import REQUEST_TIMEOUT_SECONDS, Supervisor, worker_command

log = logging.getLogger("pllm.standby")

PORT_OFFSET = 100  # from the stage's port, and from its supervisor's port
ACTIONS = ("start", "stop")


def _offset(port: int, what: str) -> int:
    if port + PORT_OFFSET > 65535:
        raise ValueError(f"{what} port {port} plus {PORT_OFFSET} is above 65535")
    return port + PORT_OFFSET


def standby_endpoint(stage: int, path: str | Path | None = None) -> config.Endpoint:
    """Where the standby of a stage takes requests."""
    control = config.supervisor_endpoint(stage, path)
    return config.Endpoint(control.host, _offset(control.port, "supervisor"))


def replacement_hosts(hosts: config.HostConfig, stage: int) -> str:
    """The hosts file of the replacement worker: the same addresses, its own port moved."""
    lines = [f"network = {json.dumps(hosts.network)}"]
    for role in config.ROLES:
        endpoint = getattr(hosts, role)
        port = endpoint.port
        if role == f"stage{stage}":
            port = _offset(port, role)
        lines += ["", f"[{role}]", f"host = {json.dumps(endpoint.host)}", f"port = {port}"]
    return "\n".join(lines) + "\n"


class Standby:
    """Holds at most one replacement supervisor. make() returns a new, unstarted one."""

    def __init__(self, make: Callable[[], Supervisor]):
        self.make = make
        self.supervisor: Supervisor | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start a replacement; False if one is already there."""
        with self._lock:
            if self.supervisor is not None:
                return False
            supervisor = self.make()
            supervisor.start()
            self.supervisor = supervisor
            return True

    def stop(self) -> bool:
        """Stop the replacement and kill its worker; False if there was none."""
        with self._lock:
            if self.supervisor is None:
                return False
            self.supervisor.stop()
            self.supervisor = None
            return True

    def status(self) -> dict:
        supervisor = self.supervisor
        if supervisor is None:
            return {"state": "idle", "free_ram_bytes": psutil.virtual_memory().available}
        return supervisor.status()


def serve_standby(standby: Standby, bind: tuple[str, int]) -> ThreadingHTTPServer:
    """Serve start, stop and status on bind (port 0 picks a free one) from a thread."""

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
                self._send(200, standby.status())
            else:
                self._send(404, {"ok": False, "error": f"no such path {self.path}"})

        def do_POST(self):  # noqa: N802
            action = self.path.lstrip("/")
            if action not in ACTIONS:
                self._send(404, {"ok": False, "error": f"no such path {self.path}"})
                return
            kind = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if kind != "application/json":
                self._send(415, {"ok": False, "error": "send Content-Type: application/json"})
                return
            try:
                changed = getattr(standby, action)()
            except (OSError, RuntimeError, psutil.Error) as err:
                self._send(500, {"ok": False, "error": f"{type(err).__name__}: {err}"})
                return
            if action == "start" and not changed:
                self._send(409, {"ok": False, "error": "a replacement is already running"})
            else:
                self._send(200, {"ok": True, "action": action, "changed": changed})

        def log_message(self, format, *args):  # noqa: A002
            log.debug("control: " + format, *args)

    server = ThreadingHTTPServer(bind, Handler)
    threading.Thread(target=server.serve_forever, name="standby-control", daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start a replacement stage worker on request.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument("--stage", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--threads", type=int, required=True, help="the torch threads of the stage replaced"
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    try:
        hosts = config.load()
        control = standby_endpoint(args.stage)
        text = replacement_hosts(hosts, args.stage)
    except (ValueError, OSError) as err:
        parser.error(str(err))
    missing = [f for f in PINNED[args.model].files if not (MODELS_DIR / args.model / f).is_file()]
    if missing:
        parser.error(
            f"model files missing ({', '.join(sorted(missing))}): "
            f"uv run python -m pllm.models {args.model}"
        )
    runs = config.REPO_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    derived = runs / f"standby_stage{args.stage}.toml"
    derived.write_text(text, encoding="utf-8")
    os.environ["PLLM_CONFIG"] = str(derived)  # inherited by the replacement worker only
    port = getattr(hosts, f"stage{args.stage}").port + PORT_OFFSET
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    def make() -> Supervisor:
        return Supervisor(
            worker_command(args.model, args.stage, args.threads),
            ready_address=("127.0.0.1", port),
            log_path=runs / f"standby_stage{args.stage}.jsonl",
            cwd=config.REPO_ROOT,
        )

    standby = Standby(make)
    try:
        server = serve_standby(standby, (control.host, control.port))
    except OSError as err:
        parser.error(f"cannot listen on {control.address}: {err}")
    done = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):  # SIGBREAK exists on Windows only
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: done.set())
    log.info(
        "standby for stage %d idle; control %s, replacement port %d; Ctrl-C to stop",
        args.stage,
        control.address,
        port,
    )
    try:
        while not done.wait(0.5):  # a timeout, so a signal handler can run while this waits
            pass
    finally:
        standby.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
