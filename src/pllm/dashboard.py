"""A live page for the demo (R9.2 to R9.5): send prompts, watch the system, press fault buttons.

The gateway starts it with --dashboard. It is a small HTTP server in the gateway process,
standard library only, so it adds no dependency and nothing to Host B. Reported numbers come
from the scripts that read the metrics files, never from this page.

- GET /           the page: one HTML file with plain JavaScript, no library and no address
                  outside this server.
- GET /events     server-sent events. "state" once a second: each stage alive, dead or
                  unknown with its epoch and the age of its last Beat; the running, queued
                  and in-flight counts; tokens per second; requests rejected; results
                  dropped as stale, by reason; duplicate tokens; and the free memory of each
                  laptop, as its supervisor reports it. And, for the requests sent from the
                  page, "start", "token" and "end" events, and "fault" for each button.
- POST /generate  {"prompt", "max_tokens", "mode"}, mode one, eight or flood. Each request
                  goes to the gateway's own Frontend over gRPC on loopback, so deadlines, the
                  admission queue and refusals are the ones a client gets. Answers 202 with
                  the ids the events use.
- POST /fault     {"stage": 0|1, "action": "kill"|"suspend"|"resume"}, passed to the
                  supervisor on that stage's laptop, which carries it out (R6.2).

Duplicate tokens are counted on the streams this server relays: a token whose step does not
go above the last one of its request. The gateway delivers each step once, so it stays 0.

The page can kill workers, so it takes the supervisor's precautions. A POST must carry
Content-Type: application/json and a Host this server answers to, an Origin must be its own,
no response carries a CORS header and no OPTIONS request is answered, so a page open
elsewhere cannot drive it from the browser. Beyond that there is no authentication; it is
for the course's LAN. Clients, requests started from the page and every queue are bounded.
Nothing here depends on the operating system (R3.15).
"""

import collections
import http.client
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc
import psutil

from pllm import config
from pllm.proto import protocol_pb2, protocol_pb2_grpc

log = logging.getLogger("pllm.dashboard")

PAGE = Path(__file__).with_name("dashboard.html")
MODES = {"one": 1, "eight": 8, "flood": 40}  # requests started by each button
MAX_PROMPT_CHARS = 2000
MAX_TOKENS = 64
MAX_BODY_BYTES = 8192
DRAIN_BYTES = 1 << 20  # an oversized body is read this far before the refusal is sent
MAX_CLIENTS = 16  # open event streams
BACKLOG = 2000  # events kept for a client that reads slowly; the oldest are dropped
MAX_ACTIVE = 64  # requests started from the page that may run at once
REQUEST_TIMEOUT_S = 60  # the deadline given to each request
RATE_WINDOW_S = 5.0  # tokens per second is measured over this long
KEEPALIVE_S = 10.0
SUPERVISOR_POLL_S = 2.0
SUPERVISOR_TIMEOUT_S = 1.0
FAULT_TIMEOUT_S = 5.0
FAULTS = ("kill", "suspend", "resume")
CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class Hub:
    """Events for the open event streams. Each client has a bounded queue: when one reads
    too slowly its oldest events are dropped, so a stuck browser never holds memory or
    blocks the requests that publish."""

    def __init__(self, max_clients: int = MAX_CLIENTS, backlog: int = BACKLOG):
        self.max_clients = max_clients
        self.backlog = backlog
        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self.closed = False
        self.dropped = 0

    def subscribe(self) -> queue.Queue | None:
        """A queue of (event, data) pairs, or None if there are too many clients."""
        with self._lock:
            if self.closed or len(self._clients) >= self.max_clients:
                return None
            box: queue.Queue = queue.Queue(self.backlog)
            self._clients.add(box)
            return box

    def unsubscribe(self, box: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(box)

    def publish(self, event: str, data: dict) -> None:
        text = json.dumps(data, separators=(",", ":"))
        with self._lock:
            for box in self._clients:
                self._put(box, (event, text))

    def _put(self, box: queue.Queue, item) -> None:
        while True:
            try:
                box.put_nowait(item)
                return
            except queue.Full:
                try:
                    box.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def close(self) -> None:
        """Wake every client with the end marker."""
        with self._lock:
            self.closed = True
            for box in self._clients:
                self._put(box, None)


class Rate:
    """Tokens per second over the last RATE_WINDOW_S seconds, from samples of a running
    total. The clock is a parameter so a test can drive it."""

    def __init__(self, window: float = RATE_WINDOW_S):
        self.window = window
        self._samples: collections.deque = collections.deque()
        self._lock = threading.Lock()  # the state is sampled from several threads

    def sample(self, now: float, total: int) -> float:
        with self._lock:
            if self._samples:
                now = max(now, self._samples[-1][0])  # callers read the clock before the lock
            self._samples.append((now, total))
            # keep one sample at or before the start of the window, and those after it
            while len(self._samples) >= 2 and now - self._samples[1][0] >= self.window:
                self._samples.popleft()
            then, before = self._samples[0]
        if now <= then:
            return 0.0
        return max(0.0, (total - before) / (now - then))


class Dashboard:
    """The server, its event hub and the threads that sample the system.

    gateway is the Gateway to describe; target the address of its Frontend, which requests
    from the page are sent to; supervisors maps a stage to the "host:port" of its
    supervisor; bind is the (host, port) to listen on, with port 0 for a free one.
    """

    def __init__(
        self,
        gateway,
        target: str,
        supervisors: dict[int, str],
        bind: tuple[str, int],
        interval: float = 1.0,
        poll: float = SUPERVISOR_POLL_S,
        clock=time.monotonic,
    ):
        host = bind[0]
        if host in ("", "0.0.0.0", "::"):
            raise ValueError(
                "the dashboard host must be the address the browser uses, not " + repr(host)
            )
        self.gateway = gateway
        self.target = target
        self.supervisors = supervisors
        self.interval = interval
        self.poll = poll
        self.clock = clock
        self.hub = Hub()
        self.rate = Rate()
        self.duplicates = 0
        self._hosts: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ids = 0
        self._active = threading.BoundedSemaphore(MAX_ACTIVE)
        self._stop = threading.Event()
        self._started = False
        self._channel = grpc.insecure_channel(target)
        self._stub = protocol_pb2_grpc.FrontendStub(self._channel)
        self._server = ThreadingHTTPServer(bind, self._handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.hosts_allowed = {f"{host}:{self.port}"}
        if host in ("127.0.0.1", "localhost"):
            self.hosts_allowed |= {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self._threads = [
            threading.Thread(target=self._serve, name="dashboard", daemon=True),
            threading.Thread(target=self._tick, name="dashboard-state", daemon=True),
            threading.Thread(target=self._poll, name="dashboard-hosts", daemon=True),
        ]

    # ------------------------------------------------------------------ life

    def start(self) -> "Dashboard":
        self._started = True
        for thread in self._threads:
            thread.start()
        log.info("dashboard at http://%s/", sorted(self.hosts_allowed)[0])
        return self

    def close(self) -> None:
        self._stop.set()
        self.hub.close()
        if self._started:  # shutdown() waits for a serve_forever that was never started
            self._server.shutdown()
        self._server.server_close()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=5)
        self._channel.close()

    def _serve(self) -> None:
        self._server.serve_forever(poll_interval=0.1)

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        """What the page shows, as the "state" event."""
        gateway = self.gateway.state()
        with self._lock:
            hosts = {key: dict(value) for key, value in self._hosts.items()}
            duplicates = self.duplicates
        hosts["gateway"] = {"free_ram_bytes": psutil.virtual_memory().available}
        return {
            **gateway,
            "tokens_per_second": round(self.rate.sample(self.clock(), gateway["tokens_out"]), 2),
            "rejects": gateway["outcomes"].get("rejected", 0),
            "stale": sum(gateway["discards"].values()),
            "duplicates": duplicates,
            "hosts": hosts,
        }

    def _tick(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.hub.publish("state", self.state())
            except Exception:  # a bad sample must not end the updates
                log.exception("dashboard state failed")

    def _poll(self) -> None:
        while True:
            for stage, address in self.supervisors.items():
                try:
                    found = self._supervisor_status(address)
                except Exception:  # the polling must outlive any one bad answer
                    log.exception("polling the supervisor of stage %s failed", stage)
                    found = {"reachable": False}
                with self._lock:
                    self._hosts[str(stage)] = found
            if self._stop.wait(self.poll):
                return

    @staticmethod
    def _supervisor_status(address: str) -> dict:
        try:
            with urllib.request.urlopen(
                f"http://{address}/status", timeout=SUPERVISOR_TIMEOUT_S
            ) as reply:
                body = json.load(reply)
        except (OSError, ValueError, http.client.HTTPException):
            return {"reachable": False}
        if not isinstance(body, dict):
            return {"reachable": False}
        keep = ("state", "restarts", "ready", "free_ram_bytes")
        return {"reachable": True, **{key: body.get(key) for key in keep}}

    # ------------------------------------------------------------------ requests

    def generate(self, prompt: str, max_tokens: int, mode: str) -> tuple[int, dict]:
        """Start the requests of one button press; (HTTP status, body)."""
        wanted = MODES[mode]
        ids = []
        for _ in range(wanted):
            if not self._active.acquire(blocking=False):
                break
            with self._lock:
                self._ids += 1
                did = f"d{self._ids}"
            ids.append(did)
            self.hub.publish("start", {"id": did, "prompt": prompt[:80], "max_tokens": max_tokens})
            try:
                threading.Thread(
                    target=self._run, args=(did, prompt, max_tokens), name=did, daemon=True
                ).start()
            except RuntimeError:  # no thread could be made: give the place back
                self._active.release()
                self.hub.publish("end", {"id": did, "outcome": "ERROR", "tokens": 0})
        if not ids:
            return 429, {"ok": False, "error": "too many requests are running"}
        return 202, {"ok": True, "ids": ids, "skipped": wanted - len(ids)}

    def _run(self, did: str, prompt: str, max_tokens: int) -> None:
        last, count = -1, 0
        try:
            request = protocol_pb2.GenerateReq(
                prompt=prompt, max_tokens=max_tokens, timeout_ms=REQUEST_TIMEOUT_S * 1000
            )
            for token in self._stub.Generate(request, timeout=REQUEST_TIMEOUT_S + 5):
                duplicate = token.step <= last
                last = max(last, token.step)
                count += 1
                if duplicate:
                    with self._lock:
                        self.duplicates += 1
                self.hub.publish(
                    "token",
                    {
                        "id": did,
                        "rid": token.rid,
                        "step": token.step,
                        "token": token.token,
                        "text": token.text,
                        "duplicate": duplicate,
                    },
                )
            self.hub.publish("end", {"id": did, "outcome": "done", "tokens": count})
        except grpc.RpcError as err:
            self.hub.publish(
                "end",
                {"id": did, "outcome": err.code().name, "detail": err.details(), "tokens": count},
            )
        except Exception:  # never leave a card waiting
            log.exception("request %s failed", did)
            self.hub.publish("end", {"id": did, "outcome": "ERROR", "tokens": count})
        finally:
            self._active.release()

    def fault(self, stage: int, action: str) -> tuple[int, dict]:
        """Pass a fault to the supervisor of a stage; (HTTP status, body)."""
        address = self.supervisors.get(stage)
        if address is None:
            return 400, {"ok": False, "error": f"no supervisor for stage {stage}"}
        request = urllib.request.Request(
            f"http://{address}/{action}",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=FAULT_TIMEOUT_S) as reply:
                code, body = reply.status, json.load(reply)
        except urllib.error.HTTPError as err:  # the supervisor refused: say why
            code = err.code
            try:
                body = json.load(err)
            except (OSError, ValueError, http.client.HTTPException):
                body = {"ok": False, "error": f"supervisor answered {err.code}"}
        except (OSError, ValueError, http.client.HTTPException):
            code, body = 502, {"ok": False, "error": f"supervisor of stage {stage} unreachable"}
        if not isinstance(body, dict):
            code, body = 502, {"ok": False, "error": f"supervisor of stage {stage} sent no object"}
        self.hub.publish(
            "fault",
            {
                "stage": stage,
                "action": action,
                "ok": bool(body.get("ok")),
                "code": code,
                "error": body.get("error"),
            },
        )
        return code, body

    # ------------------------------------------------------------------ HTTP

    def _handler(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            timeout = 30  # a connection that stalls is dropped
            protocol_version = "HTTP/1.1"

            def _send(self, code: int, body: bytes, kind: str, extra: dict | None = None):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if code >= 400:  # a body we did not read must not be taken for a request
                    self.send_header("Connection", "close")
                    self.close_connection = True
                for key, value in (extra or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, body: dict) -> None:
                self._send(code, json.dumps(body).encode(), "application/json")

            def _allowed(self) -> bool:
                """The Host must be one this server answers to and an Origin, if sent, its
                own: the checks against a page on another site, or a rebound name."""
                host = self.headers.get("Host", "")
                if host not in dashboard.hosts_allowed:
                    self._json(403, {"ok": False, "error": "unknown Host"})
                    return False
                origin = self.headers.get("Origin")
                if origin is not None and origin != f"http://{host}":
                    self._json(403, {"ok": False, "error": "foreign Origin"})
                    return False
                return True

            def do_GET(self):  # noqa: N802
                if not self._allowed():
                    return
                if self.path == "/":
                    headers = {"Content-Security-Policy": CSP, "X-Frame-Options": "DENY"}
                    page = PAGE.read_bytes().replace(b"__FLOOD__", str(MODES["flood"]).encode())
                    self._send(200, page, "text/html; charset=utf-8", headers)
                elif self.path == "/events":
                    self._events()
                else:
                    self._json(404, {"ok": False, "error": f"no such path {self.path}"})

            def do_POST(self):  # noqa: N802
                # The body is read before any answer, so an error never leaves unread bytes
                # on the connection (a reset would hide the answer from the client).
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    length = -1
                if length < 0:
                    self._json(411, {"ok": False, "error": "send Content-Length"})
                    return
                if length > MAX_BODY_BYTES:
                    self._drain(min(length, DRAIN_BYTES))
                    self._json(413, {"ok": False, "error": "body too large"})
                    return
                raw = self.rfile.read(length)
                if not self._allowed():
                    return
                if self.path not in ("/generate", "/fault"):
                    self._json(404, {"ok": False, "error": f"no such path {self.path}"})
                    return
                kind = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if kind != "application/json":
                    self._json(415, {"ok": False, "error": "send Content-Type: application/json"})
                    return
                try:
                    body = json.loads(raw or b"null")
                except ValueError:
                    body = None
                if not isinstance(body, dict):
                    self._json(400, {"ok": False, "error": "the body must be a JSON object"})
                    return
                if self.path == "/generate":
                    code, reply = self._generate(body)
                else:
                    code, reply = self._fault(body)
                self._json(code, reply)

            def _drain(self, count: int) -> None:
                while count > 0:
                    chunk = self.rfile.read(min(count, 65536))
                    if not chunk:
                        return
                    count -= len(chunk)

            def _generate(self, body: dict) -> tuple[int, dict]:
                prompt, count, mode = body.get("prompt"), body.get("max_tokens"), body.get("mode")
                if not isinstance(prompt, str) or not 1 <= len(prompt) <= MAX_PROMPT_CHARS:
                    error = f"prompt must be 1 to {MAX_PROMPT_CHARS} characters"
                    return 400, {"ok": False, "error": error}
                if type(count) is not int or not 1 <= count <= MAX_TOKENS:
                    return 400, {"ok": False, "error": f"max_tokens must be 1 to {MAX_TOKENS}"}
                if mode not in MODES:
                    return 400, {"ok": False, "error": f"mode must be one of {sorted(MODES)}"}
                return dashboard.generate(prompt, count, mode)

            def _fault(self, body: dict) -> tuple[int, dict]:
                stage, action = body.get("stage"), body.get("action")
                if type(stage) is not int or stage not in (0, 1):
                    return 400, {"ok": False, "error": "stage must be 0 or 1"}
                if action not in FAULTS:
                    return 400, {"ok": False, "error": f"action must be one of {list(FAULTS)}"}
                return dashboard.fault(stage, action)

            def _events(self) -> None:
                box = dashboard.hub.subscribe()
                if box is None:
                    self._json(503, {"ok": False, "error": "too many open event streams"})
                    return
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    self._write("retry: 2000\n\n")
                    self._write(f"event: state\ndata: {json.dumps(dashboard.state())}\n\n")
                    while True:
                        try:
                            item = box.get(timeout=KEEPALIVE_S)
                        except queue.Empty:
                            self._write(": keepalive\n\n")
                            continue
                        if item is None:
                            return
                        self._write(f"event: {item[0]}\ndata: {item[1]}\n\n")
                except OSError:
                    return  # the browser went away
                finally:
                    dashboard.hub.unsubscribe(box)

            def _write(self, text: str) -> None:
                self.wfile.write(text.encode())
                self.wfile.flush()

            def log_message(self, format, *args):  # noqa: A002
                log.debug("dashboard: " + format, *args)

        return Handler


def from_config(gateway, bind: tuple[str, int] | None = None) -> Dashboard:
    """The dashboard of a gateway, with its addresses read from the hosts file."""
    hosts = config.load()
    endpoint = config.dashboard_endpoint()
    return Dashboard(
        gateway,
        f"127.0.0.1:{hosts.gateway.port}",
        {stage: config.supervisor_endpoint(stage).address for stage in (0, 1)},
        bind or (endpoint.host, endpoint.port),
    )
