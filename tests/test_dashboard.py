"""The dashboard: the page, the event stream, requests sent from it and the fault buttons
(R9.2 to R9.5).

The integration tests run a real gateway and two real stage servers on loopback, with their
heartbeats, and stand-ins for the two supervisors that record what they are asked to do.
The page's JavaScript is not run here: the tests pin what it is served and every event it
reads. Equality with the request run alone is on one host, fp32, batch 1.
"""

import http.client
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc
import pytest
from test_gateway import IDS, PROMPT, WAIT, System, wait_for
from test_heartbeat import Beating
from test_model import tiny_reference

from pllm import config, dashboard, gateway, supervisor
from pllm.dashboard import Dashboard, Hub, Rate
from pllm.proto import protocol_pb2
from pllm.reference import generate
from pllm.split import InProcessSplit

LAYERS = 4
K = 2
PAGE_CHECK = Path(__file__).with_name("dashboard_page_check.js")


class FakeSupervisor:
    """What serve_control needs of a supervisor: it records the faults asked of it."""

    def __init__(self, free=123_456_789):
        self.calls = []
        self.fail = None
        self.free = free

    def status(self):
        return {
            "state": "running",
            "pid": 4242,
            "restarts": 0,
            "ready": True,
            "last_respawn_seconds": None,
            "free_ram_bytes": self.free,
        }

    def apply(self, action):
        if self.fail is not None:
            raise self.fail
        self.calls.append(action)
        return 4242


class Broken(BaseHTTPRequestHandler):
    """A supervisor whose reply breaks off: it promises 100 bytes and sends four."""

    def reply(self):
        self.send_response(200)
        self.send_header("Content-Length", "100")
        self.end_headers()
        self.wfile.write(b'{"st')

    do_GET = do_POST = reply

    def log_message(self, *args):
        pass


class NotAnObject(BaseHTTPRequestHandler):
    """A supervisor that answers 200 with JSON that is not an object."""

    def reply(self):
        body = b"[1, 2]"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = reply

    def log_message(self, *args):
        pass


def serving(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Events:
    """A client of GET /events: a thread reads the stream into a queue of (event, data)."""

    def __init__(self, port, headers=None):
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
        self.conn.request("GET", "/events", headers=headers or {})
        self.response = self.conn.getresponse()
        self.items: queue.Queue = queue.Queue()
        self.seen = []
        self._thread = threading.Thread(target=self._read, daemon=True)
        if self.response.status == 200:
            self._thread.start()

    def _read(self):
        event = None
        try:
            for raw in self.response:
                line = raw.decode().rstrip("\n")
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event is not None:
                    self.items.put((event, json.loads(line[6:])))
                    event = None
        except (OSError, ValueError, http.client.HTTPException):
            pass

    def next(self, kind, match=lambda data: True, timeout=WAIT):
        """The next event of this kind whose data satisfies match; others are kept."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                event, data = self.items.get(timeout=0.2)
            except queue.Empty:
                continue
            self.seen.append((event, data))
            if event == kind and match(data):
                return data
        raise AssertionError(f"no {kind} event; saw {[e for e, _ in self.seen][-12:]}")

    def collect(self, kind, count, timeout=WAIT):
        return [self.next(kind, timeout=timeout) for _ in range(count)]

    def close(self):
        self.conn.close()


@pytest.fixture
def system(tmp_path):
    """make(**kwargs) -> (run, dash, supervisors); everything is closed afterwards."""
    made = []

    def make(interval=0.1, **kwargs):
        if not (tmp_path / "config.json").exists():
            tiny_reference(tmp_path, layers=LAYERS)
        run = Beating(tmp_path, K, **kwargs)
        fakes = {0: FakeSupervisor(), 1: FakeSupervisor()}
        servers = {s: supervisor.serve_control(f, ("127.0.0.1", 0)) for s, f in fakes.items()}
        addresses = {s: f"127.0.0.1:{srv.server_address[1]}" for s, srv in servers.items()}
        dash = Dashboard(
            run.gateway, run.address, addresses, ("127.0.0.1", 0), interval=interval, poll=0.1
        ).start()
        made.append((run, dash, servers))
        return run, dash, fakes, servers

    yield make
    for run, dash, servers in made:
        dash.close()
        for server in servers.values():
            server.shutdown()
            server.server_close()
        run.close()


def request(dash, method, path, body=None, headers=None, raw=None):
    """One request to the dashboard; (status, headers, body bytes)."""
    conn = http.client.HTTPConnection("127.0.0.1", dash.port, timeout=WAIT)
    try:
        headers = dict(headers or {})
        if body is not None:
            raw = json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def post(dash, path, body, **kwargs):
    status, headers, data = request(dash, "POST", path, body, **kwargs)
    return status, json.loads(data)


def reference(tmp_path, count):
    return generate(InProcessSplit.load(tmp_path, K), IDS, count)


# -------------------------------------------------------------- the page


def test_the_page_is_served_with_a_content_security_policy_and_reaches_nothing_outside(system):
    run, dash, *_ = system()
    status, headers, body = request(dash, "GET", "/")
    page = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert "<title>pllm dashboard</title>" in page and "__FLOOD__" not in page
    assert f"Flood ({dashboard.MODES['flood']})" in page
    assert "connect-src 'self'" in headers["Content-Security-Policy"]
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Frame-Options"] == "DENY" and headers["Cache-Control"] == "no-store"
    assert not re.search(r"https?://", page)  # no address outside this server
    assert "innerHTML" not in page and "eval(" not in page  # model text is only text


def test_unknown_paths_are_not_found_and_options_is_not_answered(system):
    run, dash, *_ = system()
    assert request(dash, "GET", "/nothing")[0] == 404
    assert request(dash, "POST", "/nothing", {})[0] == 404
    status, headers, _ = request(dash, "OPTIONS", "/generate")
    assert status == 501 and not any(k.lower().startswith("access-control") for k in headers)


# -------------------------------------------------------------- the state


def both_stages_up(state):
    return (
        state["stages"]["0"]["state"] == "alive"
        and state["stages"]["1"]["state"] == "alive"
        and state["hosts"].get("1", {}).get("reachable")
    )


def test_the_state_event_describes_each_stage_the_queue_and_the_laptops(system):
    run, dash, fakes, _ = system(max_queued=3)
    events = Events(dash.port)
    try:
        state = events.next("state", both_stages_up)
    finally:
        events.close()
    for stage in (0, 1):
        info = state["stages"][str(stage)]
        assert info["epoch"] == run.registry.stages[stage].epoch
        assert info["addr"] == run.registry.stages[stage].addr
        assert 0 <= info["beat_age_s"] < 2
        host = state["hosts"][str(stage)]
        assert host["state"] == "running" and host["free_ram_bytes"] == 123_456_789
    assert state["hosts"]["gateway"]["free_ram_bytes"] > 0
    assert (state["running"], state["queued"], state["max_queued"]) == (0, 0, 3)
    assert state["in_flight"] == 0 and state["tokens_out"] == 0
    assert (state["rejects"], state["stale"], state["duplicates"]) == (0, 0, 0)
    assert state["tokens_per_second"] == 0


def test_a_stage_whose_epoch_is_declared_dead_is_shown_dead(system):
    run, dash, *_ = system()
    events = Events(dash.port)
    try:
        events.next("state", lambda s: s["stages"]["1"]["state"] == "alive")
        assert run.gateway.detector.declare(1, run.registry.stages[1].epoch, "heartbeat_timeout")
        state = events.next("state", lambda s: s["stages"]["1"]["state"] == "dead")
        assert state["stages"]["0"]["state"] == "alive"
    finally:
        events.close()


def test_a_stage_whose_last_beat_is_old_is_unknown_until_it_is_declared_dead(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    run = System(tmp_path, K)  # registered, with no heartbeat streams open
    try:
        now = time.monotonic()
        for stage, age in ((0, 100), (1, 0.1)):
            beat = protocol_pb2.Beat(stage=stage, epoch=run.registry.stages[stage].epoch)
            run.gateway._beats[stage] = (beat, now - age)
        state = run.gateway.state()
        assert state["stages"]["0"]["state"] == "unknown"
        assert state["stages"]["0"]["beat_age_s"] > 90
        assert state["stages"]["1"]["state"] == "alive"
        old = protocol_pb2.Beat(stage=1, epoch=run.registry.stages[1].epoch + 100)
        run.gateway._beats[1] = (old, now - 0.1)  # fresh, but from another incarnation
        assert run.gateway.state()["stages"]["1"]["state"] == "unknown"
        run.gateway.detector.declare(0, run.registry.stages[0].epoch, "heartbeat_timeout")
        assert run.gateway.state()["stages"]["0"]["state"] == "dead"
    finally:
        run.close()


def test_a_stage_that_has_not_registered_is_unknown_not_dead(tmp_path, system):
    run, dash, *_ = system(order=())
    state = dash.state()
    assert [state["stages"][s]["state"] for s in "01"] == ["unknown", "unknown"]
    assert state["stages"]["0"]["epoch"] is None


def test_an_unreachable_supervisor_is_reported_and_the_rest_still_shown(system):
    run, dash, fakes, servers = system()
    servers[1].shutdown()
    servers[1].server_close()
    events = Events(dash.port)
    try:
        state = events.next("state", lambda s: s["hosts"].get("1") == {"reachable": False})
    finally:
        events.close()
    assert state["hosts"]["0"]["reachable"] is True
    assert state["stages"]["1"]["state"] == "alive"  # the stage is judged by its beats


def test_a_supervisor_whose_reply_breaks_off_reads_as_unreachable_and_polling_goes_on(system):
    run, dash, fakes, _ = system()
    broken = serving(Broken)
    try:
        dash.supervisors[0] = f"127.0.0.1:{broken.server_address[1]}"
        wait_for(
            lambda: dash.state()["hosts"].get("0") == {"reachable": False},
            "stage 0's supervisor to read as unreachable",
        )
        assert dash._threads[2].is_alive()  # the polling thread survived the bad answer
        assert dash.state()["hosts"]["1"]["reachable"] is True  # and still reads the other
        status, body = post(dash, "/fault", {"stage": 0, "action": "kill"})
        assert status == 502 and "unreachable" in body["error"]
    finally:
        broken.shutdown()
        broken.server_close()


def test_a_supervisor_reply_that_is_not_an_object_is_a_bad_gateway(system):
    run, dash, *_ = system()
    odd = serving(NotAnObject)
    try:
        dash.supervisors[0] = f"127.0.0.1:{odd.server_address[1]}"
        status, body = post(dash, "/fault", {"stage": 0, "action": "kill"})
        assert status == 502 and "no object" in body["error"]
        wait_for(
            lambda: dash.state()["hosts"].get("0") == {"reachable": False},
            "a reply that is not an object to read as unreachable",
        )
    finally:
        odd.shutdown()
        odd.server_close()


def test_dropped_results_are_counted_as_stale_with_their_reasons(system):
    run, dash, *_ = system()
    run.gateway.discards["unknown_request"] += 2
    run.gateway.discards["duplicate"] += 1
    state = dash.state()
    assert state["stale"] == 3 and state["discards"] == {"unknown_request": 2, "duplicate": 1}


def test_the_gateway_counts_tokens_sent_and_how_requests_end(system):
    run, dash, *_ = system()
    assert len(run.tokens(max_tokens=4)) == 4
    state = run.gateway.state()
    assert state["tokens_out"] == 4 and state["outcomes"] == {"max_tokens": 1}


# -------------------------------------------------------------- the rate and the hub


def test_tokens_per_second_is_the_change_over_the_window():
    rate = Rate(window=5.0)
    assert rate.sample(0.0, 0) == 0
    assert rate.sample(1.0, 10) == pytest.approx(10.0)
    assert rate.sample(2.0, 30) == pytest.approx(15.0)
    assert rate.sample(5.0, 30) == pytest.approx(6.0)  # 30 tokens over the last 5 s
    assert rate.sample(7.0, 30) == pytest.approx(0.0)  # the busy seconds have left the window
    assert rate.sample(7.0, 30) == pytest.approx(0.0)  # a second sample at one instant


def test_samples_taken_out_of_order_never_run_the_window_backwards():
    rate = Rate(window=5.0)
    rate.sample(5.0, 50)
    value = rate.sample(4.0, 60)  # a caller that read the clock before the other one
    times = [at for at, _ in rate._samples]
    assert value >= 0 and times == sorted(times)


def test_the_hub_keeps_a_bounded_backlog_dropping_the_oldest_events():
    hub = Hub(max_clients=2, backlog=3)
    box = hub.subscribe()
    for i in range(10):
        hub.publish("e", {"i": i})
    got = [json.loads(box.get_nowait()[1])["i"] for _ in range(3)]
    assert got == [7, 8, 9] and hub.dropped == 7


def test_the_hub_refuses_too_many_clients_and_close_wakes_the_rest():
    hub = Hub(max_clients=2, backlog=3)
    a, b = hub.subscribe(), hub.subscribe()
    assert hub.subscribe() is None
    hub.unsubscribe(a)
    assert hub.subscribe() is not None
    hub.close()
    assert b.get_nowait() is None and hub.subscribe() is None


def test_too_many_open_event_streams_are_refused(system):
    run, dash, *_ = system()
    dash.hub.max_clients = 1
    first = Events(dash.port)
    try:
        first.next("state")
        status, _, body = request(dash, "GET", "/events")
        assert status == 503 and b"too many" in body
    finally:
        first.close()


# -------------------------------------------------------------- requests from the page


def test_send_streams_the_tokens_of_one_request_and_they_equal_the_request_run_alone(
    tmp_path, system
):
    run, dash, *_ = system()
    want_tokens, want_logits = reference(tmp_path, 6)
    events = Events(dash.port)
    try:
        status, body = post(dash, "/generate", {"prompt": PROMPT, "max_tokens": 6, "mode": "one"})
        assert status == 202 and body["ids"] == ["d1"] and body["skipped"] == 0
        start = events.next("start")
        assert start["id"] == "d1" and start["max_tokens"] == 6
        tokens = events.collect("token", 6)
        end = events.next("end")
    finally:
        events.close()
    assert [t["token"] for t in tokens] == want_tokens
    assert [t["step"] for t in tokens] == list(range(6))
    assert all(t["id"] == "d1" and t["duplicate"] is False for t in tokens)
    assert len({t["rid"] for t in tokens}) == 1
    assert end == {"id": "d1", "outcome": "done", "tokens": 6}
    state = run.gateway.state()
    assert state["tokens_out"] == 6 and dash.duplicates == 0


def test_send_eight_at_once_gives_each_request_the_tokens_it_gives_alone(tmp_path, system):
    run, dash, *_ = system(delay1=0.05)
    want_tokens, _ = reference(tmp_path, 4)
    events = Events(dash.port)
    try:
        status, body = post(dash, "/generate", {"prompt": PROMPT, "max_tokens": 4, "mode": "eight"})
        assert status == 202 and len(body["ids"]) == 8 and body["skipped"] == 0
        ends = events.collect("end", 8)
        tokens = [d for e, d in events.seen if e == "token"]  # all came before the last end
    finally:
        events.close()
    assert len(tokens) == 32
    assert [e["outcome"] for e in ends] == ["done"] * 8
    for did in body["ids"]:
        mine = sorted((t for t in tokens if t["id"] == did), key=lambda t: t["step"])
        assert [t["token"] for t in mine] == want_tokens, did
    assert run.gateway.state()["outcomes"] == {"max_tokens": 8} and dash.duplicates == 0


def test_a_flood_beyond_the_slots_and_the_queue_is_refused_and_counted_as_rejects(
    system, monkeypatch
):
    monkeypatch.setitem(dashboard.MODES, "flood", 10)
    run, dash, *_ = system(max_workers=gateway.RESERVED_THREADS + 2, max_queued=1, delay1=0.4)
    events = Events(dash.port)
    try:
        status, body = post(dash, "/generate", {"prompt": PROMPT, "max_tokens": 3, "mode": "flood"})
        assert status == 202 and len(body["ids"]) == 10
        ends = events.collect("end", 10)
        state = events.next("state", lambda s: s["rejects"] == 7)
    finally:
        events.close()
    outcomes = sorted(e["outcome"] for e in ends)
    assert outcomes == ["RESOURCE_EXHAUSTED"] * 7 + ["done"] * 3
    assert state["rejects"] == 7 and state["duplicates"] == 0
    refused = [e for e in ends if e["outcome"] == "RESOURCE_EXHAUSTED"]
    assert all("more are waiting" in e["detail"] for e in refused)


def test_a_repeated_step_in_a_relayed_stream_is_counted_as_a_duplicate(system):
    run, dash, *_ = system()

    def stream(request, timeout):
        for step in (0, 1, 1, 2):
            yield protocol_pb2.Token(rid="r", step=step, token=7, text="x")

    dash._stub = types.SimpleNamespace(Generate=stream)
    events = Events(dash.port)
    try:
        post(dash, "/generate", {"prompt": PROMPT, "max_tokens": 4, "mode": "one"})
        tokens = events.collect("token", 4)
        state = events.next("state", lambda s: s["duplicates"] == 1)
    finally:
        events.close()
    assert [t["duplicate"] for t in tokens] == [False, False, True, False]
    assert state["duplicates"] == 1


class Failed(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self):
        return "too slow"


@pytest.mark.parametrize(
    ("error", "want"),
    [
        (Failed(), {"outcome": "DEADLINE_EXCEEDED", "detail": "too slow"}),
        (ValueError("boom"), {"outcome": "ERROR"}),
    ],
)
def test_a_request_that_fails_ends_its_card_with_the_status_and_frees_its_place(
    system, error, want
):
    run, dash, *_ = system()
    dash._active = threading.BoundedSemaphore(1)  # so only a returned place can be taken

    def stream(request, timeout):
        yield protocol_pb2.Token(rid="r", step=0, token=7, text="x")
        raise error

    dash._stub = types.SimpleNamespace(Generate=stream)
    events = Events(dash.port)
    try:
        post(dash, "/generate", {"prompt": PROMPT, "max_tokens": 4, "mode": "one"})
        end = events.next("end")
    finally:
        events.close()
    assert end == {"id": "d1", **want, "tokens": 1}
    wait_for(lambda: dash._active.acquire(blocking=False), "the place to be given back", 5)


def test_requests_from_the_page_are_capped(system):
    run, dash, *_ = system(delay1=0.5)
    dash._active = threading.BoundedSemaphore(2)
    events = Events(dash.port)
    try:
        sent = {"prompt": PROMPT, "max_tokens": 2, "mode": "eight"}
        status, body = post(dash, "/generate", sent)
        assert status == 202 and len(body["ids"]) == 2 and body["skipped"] == 6
        status, body = post(dash, "/generate", {**sent, "mode": "one"})
        assert status == 429 and "too many" in body["error"]
        assert [e["outcome"] for e in events.collect("end", 2)] == ["done", "done"]
    finally:
        events.close()


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "", "max_tokens": 4, "mode": "one"},
        {"prompt": "x" * 2001, "max_tokens": 4, "mode": "one"},
        {"prompt": 5, "max_tokens": 4, "mode": "one"},
        {"prompt": PROMPT, "max_tokens": 0, "mode": "one"},
        {"prompt": PROMPT, "max_tokens": 65, "mode": "one"},
        {"prompt": PROMPT, "max_tokens": True, "mode": "one"},
        {"prompt": PROMPT, "max_tokens": "4", "mode": "one"},
        {"prompt": PROMPT, "max_tokens": 4, "mode": "ten"},
        {"prompt": PROMPT, "max_tokens": 4},
        {},
    ],
)
def test_a_bad_request_body_starts_nothing(system, body):
    run, dash, *_ = system()
    events = Events(dash.port)
    try:
        status, reply = post(dash, "/generate", body)
        assert status == 400 and reply["ok"] is False
        time.sleep(0.3)
        assert dash._ids == 0 and run.gateway.state()["outcomes"] == {}
        assert not [e for e, _ in events.seen if e == "start"]
    finally:
        events.close()


# -------------------------------------------------------------- the fault buttons


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("action", ["kill", "suspend", "resume"])
def test_a_fault_reaches_the_supervisor_of_that_stage_only(system, stage, action):
    run, dash, fakes, _ = system()
    events = Events(dash.port)
    try:
        status, body = post(dash, "/fault", {"stage": stage, "action": action})
        fault = events.next("fault")
    finally:
        events.close()
    assert status == 200 and body == {"ok": True, "action": action, "pid": 4242}
    assert fakes[stage].calls == [action] and fakes[1 - stage].calls == []
    assert fault == {"stage": stage, "action": action, "ok": True, "code": 200, "error": None}


def test_a_refusal_by_the_supervisor_is_passed_on_with_its_reason(system):
    run, dash, fakes, _ = system()
    fakes[0].fail = supervisor.NoWorker("no worker is running")
    events = Events(dash.port)
    try:
        status, body = post(dash, "/fault", {"stage": 0, "action": "kill"})
        fault = events.next("fault")
    finally:
        events.close()
    assert status == 409 and body == {"ok": False, "error": "no worker is running"}
    assert fault["ok"] is False and fault["code"] == 409


def test_a_supervisor_that_cannot_be_reached_is_a_bad_gateway(system):
    run, dash, fakes, servers = system()
    servers[1].shutdown()
    servers[1].server_close()
    status, body = post(dash, "/fault", {"stage": 1, "action": "suspend"})
    assert status == 502 and "unreachable" in body["error"]


@pytest.mark.parametrize(
    "body",
    [
        {"stage": 2, "action": "kill"},
        {"stage": -1, "action": "kill"},
        {"stage": "0", "action": "kill"},
        {"stage": True, "action": "kill"},
        {"stage": 0, "action": "explode"},
        {"stage": 0, "action": "status"},
        {"stage": 0},
        {"action": "kill"},
    ],
)
def test_a_bad_fault_is_refused_before_any_supervisor_hears_of_it(system, body):
    run, dash, fakes, _ = system()
    status, reply = post(dash, "/fault", body)
    assert status == 400 and reply["ok"] is False
    assert fakes[0].calls == fakes[1].calls == []


def test_a_fault_is_sent_to_the_supervisor_as_json(system, monkeypatch):
    run, dash, fakes, _ = system()
    seen = []
    real = dashboard.urllib.request.urlopen

    def spy(req, timeout):
        if hasattr(req, "get_header"):
            seen.append((req.get_method(), req.full_url, req.get_header("Content-type")))
        return real(req, timeout=timeout)

    monkeypatch.setattr(dashboard.urllib.request, "urlopen", spy)
    post(dash, "/fault", {"stage": 1, "action": "resume"})
    posts = [s for s in seen if s[0] == "POST"]
    assert posts == [("POST", f"http://{dash.supervisors[1]}/resume", "application/json")]


# -------------------------------------------------------------- who may drive it


def test_a_page_on_another_site_cannot_drive_it(system):
    run, dash, fakes, _ = system()
    foreign = {"Origin": "http://evil.example"}
    status, reply = post(dash, "/fault", {"stage": 0, "action": "kill"}, headers=foreign)
    assert status == 403 and fakes[0].calls == []
    generate = {"prompt": PROMPT, "max_tokens": 2, "mode": "one"}
    assert post(dash, "/generate", generate, headers=foreign)[0] == 403
    assert request(dash, "GET", "/", headers=foreign)[0] == 403
    stream = Events(dash.port, foreign)
    assert stream.response.status == 403
    stream.close()
    assert dash._ids == 0


def test_its_own_page_is_allowed_by_origin_and_by_either_loopback_name(system):
    run, dash, fakes, _ = system()
    for host in (f"127.0.0.1:{dash.port}", f"localhost:{dash.port}"):
        headers = {"Host": host, "Origin": f"http://{host}"}
        assert request(dash, "GET", "/", headers=headers)[0] == 200
        status, _ = post(dash, "/fault", {"stage": 0, "action": "resume"}, headers=headers)
        assert status == 200
    assert fakes[0].calls == ["resume", "resume"]


def test_a_request_under_another_host_name_is_refused(system):
    run, dash, fakes, _ = system()
    rebound = {"Host": f"attacker.example:{dash.port}"}
    assert request(dash, "GET", "/", headers=rebound)[0] == 403
    assert post(dash, "/fault", {"stage": 0, "action": "kill"}, headers=rebound)[0] == 403
    assert request(dash, "GET", "/events", headers=rebound)[0] == 403
    assert fakes[0].calls == []


def test_a_post_must_be_json_and_of_a_sane_size(system):
    run, dash, fakes, _ = system()
    body = b'{"stage": 0, "action": "kill"}'
    for kind in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        status, _, _ = request(dash, "POST", "/fault", headers={"Content-Type": kind}, raw=body)
        assert status == 415, kind
    status, _, _ = request(dash, "POST", "/fault", raw=body)  # no type at all
    assert status == 415 and fakes[0].calls == []
    json_type = {"Content-Type": "application/json"}
    assert request(dash, "POST", "/fault", headers=json_type, raw=b"{not json")[0] == 400
    assert request(dash, "POST", "/fault", headers=json_type, raw=b"[1]")[0] == 400
    big = b" " * (dashboard.MAX_BODY_BYTES + 1)
    assert request(dash, "POST", "/fault", headers=json_type, raw=big)[0] == 413
    typed = {"Content-Type": "application/json; charset=utf-8"}
    assert request(dash, "POST", "/fault", headers=typed, raw=body)[0] == 200
    assert fakes[0].calls == ["kill"]


def test_no_response_carries_a_cors_header(system):
    run, dash, *_ = system()
    seen = [
        request(dash, "GET", "/")[1],
        request(dash, "GET", "/nothing")[1],
        request(dash, "POST", "/fault", {"stage": 0, "action": "resume"})[1],
        request(dash, "POST", "/fault", {"stage": 9, "action": "kill"})[1],
    ]
    events = Events(dash.port)
    seen.append(dict(events.response.getheaders()))
    events.close()
    for headers in seen:
        assert not [k for k in headers if k.lower().startswith("access-control")]


# -------------------------------------------------------------- life and wiring


@pytest.mark.parametrize("host", ["0.0.0.0", "", "::"])
def test_it_will_not_listen_on_every_address_because_no_browser_would_be_let_in(host):
    with pytest.raises(ValueError, match="address the browser uses"):
        Dashboard(types.SimpleNamespace(), "127.0.0.1:1", {}, (host, 0))


def test_a_request_whose_thread_cannot_start_gives_its_place_back_and_ends_its_card(
    system, monkeypatch
):
    run, dash, *_ = system()
    dash._active = threading.BoundedSemaphore(1)
    real = threading.Thread

    class NoRoom(real):
        def start(self):
            if self.name.startswith("d") and self.name[1:].isdigit():
                raise RuntimeError("can't start new thread")
            super().start()

    monkeypatch.setattr(dashboard.threading, "Thread", NoRoom)
    events = Events(dash.port)
    try:
        status, body = post(dash, "/generate", {"prompt": PROMPT, "max_tokens": 2, "mode": "one"})
        assert status == 202 and body["ids"] == ["d1"]
        assert events.next("end") == {"id": "d1", "outcome": "ERROR", "tokens": 0}
    finally:
        events.close()
    assert dash._active.acquire(blocking=False)  # the place was given back


def test_the_server_can_be_closed_without_having_been_started(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    stub = types.SimpleNamespace()
    dash = Dashboard(stub, "127.0.0.1:1", {}, ("127.0.0.1", 0))
    dash.close()  # would hang waiting for a server loop that never ran


def test_closing_ends_the_open_event_streams(system):
    run, dash, *_ = system()
    events = Events(dash.port)
    events.next("state")
    dash.close()
    wait_for(lambda: not events._thread.is_alive(), "the event stream to end")
    events.close()


def test_the_addresses_come_from_the_hosts_file(tmp_path, monkeypatch):
    hosts = tmp_path / "hosts.toml"
    hosts.write_text(
        'network = "lab"\n'
        '[gateway]\nhost = "10.0.0.1"\nport = 50050\n'
        '[stage0]\nhost = "10.0.0.2"\nport = 50051\n'
        '[stage1]\nhost = "10.0.0.1"\nport = 50052\n'
        '[supervisor0]\nhost = "10.0.0.2"\nport = 50061\n'
        '[dashboard]\nhost = "127.0.0.1"\nport = 50080\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PLLM_CONFIG", str(hosts))
    dash = dashboard.from_config(types.SimpleNamespace(), bind=("127.0.0.1", 0))
    try:
        assert dash.target == "127.0.0.1:50050"  # its own gateway, on loopback
        assert dash.supervisors == {0: "10.0.0.2:50061", 1: "10.0.0.1:50062"}
    finally:
        dash.close()
    assert config.dashboard_endpoint(hosts).address == "127.0.0.1:50080"


def test_the_dashboard_stops_before_the_gateway_does():
    order = []

    class Part:
        def __init__(self, name):
            self.name = name
            self.recorder = types.SimpleNamespace(finish=lambda: order.append("finish"))

        def close(self):
            order.append(self.name)

        def stop(self, grace):
            order.append("server")
            return types.SimpleNamespace(wait=lambda: None)

    done = threading.Event()
    done.set()
    part = Part("gateway")
    gateway.run_until_stopped(part, Part("server"), done, Part("dashboard"))
    assert order == ["dashboard", "gateway", "server", "gateway", "finish"]


def test_a_plain_gateway_does_not_load_the_dashboard():
    code = "import sys, pllm.gateway; sys.exit('pllm.dashboard' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page's script")
def test_the_pages_script_draws_the_state_the_server_sends_and_asks_for_what_its_buttons_say(
    system, tmp_path
):
    """The page's own JavaScript runs in node against a small stand-in for the DOM, fed a
    real state event from a running dashboard and the request and fault events."""
    run, dash, *_ = system()
    events = Events(dash.port)
    try:
        state = events.next("state", both_stages_up)
    finally:
        events.close()
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    result = subprocess.run(
        ["node", str(PAGE_CHECK), str(dashboard.PAGE), str(tmp_path / "state.json")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "page script ok" in result.stdout
