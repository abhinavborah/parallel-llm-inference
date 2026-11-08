"""The supervisor: respawn, respawn time, fault requests, and registration after a kill
(R6.2, R6.1).

The workers here are small stand-in scripts, so the tests need no model weights and run
the same way on macOS, Windows and Linux. The real worker is covered by a test that is
skipped without the SmolLM2 weights.
"""

import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from concurrent import futures

import grpc
import psutil
import pytest

from pllm import config, control, faults
from pllm.control import ControlService, ExpectedModel, Registry
from pllm.models import MODELS_DIR, sha256_of
from pllm.proto import protocol_pb2_grpc
from pllm.supervisor import NoWorker, Supervisor, serve_control, worker_command
from pllm.worker import exit_when_parent_gone, register_with_gateway

WAIT = 30.0

LISTENER = """
import socket, sys, time
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", int(sys.argv[1])))
server.listen()
{extra}
while True:
    time.sleep(0.05)
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def script(tmp_path, extra="", name="worker.py"):
    path = tmp_path / name
    path.write_text(LISTENER.format(extra=textwrap.dedent(extra)))
    return path


def make(tmp_path, worker, port, **kwargs):
    kwargs.setdefault("log_path", tmp_path / "events.jsonl")
    return Supervisor(
        [sys.executable, str(worker), str(port)],
        ready_address=("127.0.0.1", port),
        poll_seconds=0.02,
        **kwargs,
    )


def events(tmp_path):
    path = tmp_path / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def names(tmp_path):
    return [e["event"] for e in events(tmp_path)]


def wait_for(condition, what, timeout=WAIT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def running():
    started = []

    def run(sup):
        started.append(sup)
        sup.start()
        assert sup.wait_ready(WAIT), "the first worker never listened"
        return sup

    yield run
    for sup in started:
        sup.stop()


def started_at(pid):
    return psutil.Process(pid).create_time()


def describe(pid):
    try:
        proc = psutil.Process(pid)
        return f"{proc.name()} {proc.status()} {proc.cmdline()}"
    except psutil.Error as err:
        return repr(err)


def alive(pid, created=None):
    """True if the process `pid` runs. Pass its `started_at` to ignore a new process that the
    operating system gave the same pid (Windows reuses pids quickly)."""
    try:
        proc = psutil.Process(pid)
        if created is not None and proc.create_time() != created:
            return False
        return proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


# ----------------------------------------------------------------------- respawn


def test_a_worker_killed_by_the_fault_injector_is_respawned(tmp_path, running):
    port = free_port()
    sup = running(make(tmp_path, script(tmp_path), port))
    first = sup.pid
    created = started_at(first)
    faults.kill(first)  # SIGKILL on macOS and Linux, TerminateProcess on Windows
    assert sup.wait_ready(WAIT, restarts=1)
    assert sup.pid != first and sup.restarts == 1 and not alive(first, created)
    socket.create_connection(("127.0.0.1", port), timeout=2).close()  # listening again


def test_a_worker_that_exits_by_itself_is_respawned(tmp_path, running):
    # lives about a second, then exits with code 3
    worker = script(tmp_path, "time.sleep(1.0); sys.exit(3)")
    sup = running(make(tmp_path, worker, free_port()))
    wait_for(lambda: sup.restarts >= 1, "the first respawn")
    exited = [e for e in events(tmp_path) if e["event"] == "exited"][0]
    assert exited["code"] == 3


def test_the_respawn_time_is_logged(tmp_path, running):
    # a 0.3 s pause before the restart, so that "since the exit" differs from "since the spawn"
    sup = running(make(tmp_path, script(tmp_path), free_port(), backoff_seconds=(0.3,)))
    faults.kill(sup.pid)
    assert sup.wait_ready(WAIT, restarts=1)
    wait_for(lambda: names(tmp_path).count("ready") == 2, "the second ready event")
    log = events(tmp_path)
    assert [e["event"] for e in log] == [
        "spawned",
        "ready",
        "exited",
        "backoff",
        "spawned",
        "ready",
    ]
    first_ready, exited, again = log[1], log[2], log[5]
    assert "respawn_seconds" not in first_ready  # a first start is not a respawn
    assert exited["code"] != 0
    assert 0.3 <= again["respawn_seconds"] < WAIT
    assert again["respawn_seconds"] == sup.last_respawn_seconds
    # the time runs from noticing the exit to listening again
    assert again["mono"] - exited["mono"] == pytest.approx(again["respawn_seconds"], abs=0.25)
    assert all({"time", "mono"} <= e.keys() for e in log)


def test_every_kill_is_followed_by_a_respawn(tmp_path, running):
    sup = running(make(tmp_path, script(tmp_path), free_port(), stable_seconds=0.0))
    for n in (1, 2, 3):
        faults.kill(sup.pid)
        assert sup.wait_ready(WAIT, restarts=n)
    assert sup.restarts == 3


def test_a_crash_loop_backs_off_and_never_gives_up(tmp_path):
    worker = script(tmp_path, "sys.exit(1)")  # dies at once, every time
    sup = make(tmp_path, worker, free_port(), backoff_seconds=(0.0, 0.2, 0.4), stable_seconds=5.0)
    sup.start()  # not through `running`: this worker never listens
    try:
        wait_for(lambda: names(tmp_path).count("spawned") >= 5, "five starts")
        pauses = [e["seconds"] for e in events(tmp_path) if e["event"] == "backoff"]
        assert pauses[:3] == [0.2, 0.4, 0.4]  # grows, then stays at the last value
        assert sup.status()["state"] in ("running", "starting")
    finally:
        sup.stop()


def test_a_command_that_cannot_start_is_retried_not_fatal(tmp_path):
    sup = Supervisor(
        [str(tmp_path / "no_such_program")],
        log_path=tmp_path / "events.jsonl",
        backoff_seconds=(0.05,),
        poll_seconds=0.02,
    )
    sup.start()
    try:
        wait_for(lambda: names(tmp_path).count("spawn_failed") >= 3, "three failed starts")
        assert sup.status()["state"] == "starting" and sup.pid is None
    finally:
        sup.stop()


def test_stop_leaves_no_worker_behind(tmp_path):
    sup = make(tmp_path, script(tmp_path), free_port())
    sup.start()
    assert sup.wait_ready(WAIT)
    pid = sup.pid
    created = started_at(pid)
    sup.stop()
    assert not alive(pid, created), f"worker {pid} still running after stop: {describe(pid)}"
    assert names(tmp_path)[-1] == "stopped" and sup.status()["state"] == "stopped"


def test_a_supervisor_needs_a_command_and_sane_backoff():
    with pytest.raises(ValueError, match="command is empty"):
        Supervisor([])
    with pytest.raises(ValueError, match="non-negative"):
        Supervisor(["x"], backoff_seconds=(-1.0,))
    with pytest.raises(ValueError, match="non-negative"):
        Supervisor(["x"], backoff_seconds=())


# ---------------------------------------------------------------- fault requests


def test_kill_suspend_and_resume_are_applied_through_the_fault_injector(
    tmp_path, running, monkeypatch
):
    calls = []
    for name in ("kill", "suspend", "resume"):
        real = getattr(faults, name)
        monkeypatch.setattr(
            faults, name, lambda pid, _n=name, _r=real: (calls.append((_n, pid)), _r(pid))[1]
        )
    sup = running(make(tmp_path, script(tmp_path), free_port()))
    pid = sup.pid
    assert sup.apply("suspend") == pid and sup.status()["state"] == "suspended"
    if sys.platform != "win32":  # the signal is delivered a moment later
        wait_for(lambda: psutil.Process(pid).status() == psutil.STATUS_STOPPED, "the stop")
    assert sup.apply("resume") == pid and sup.status()["state"] == "running"
    assert sup.apply("kill") == pid
    assert calls == [("suspend", pid), ("resume", pid), ("kill", pid)]
    assert [e["action"] for e in events(tmp_path) if e["event"] == "fault"] == [
        "suspend",
        "resume",
        "kill",
    ]


def test_a_suspended_worker_is_not_respawned_and_continues_after_resume(tmp_path, running):
    port = free_port()
    sup = running(make(tmp_path, script(tmp_path), port))
    pid = sup.pid
    sup.apply("suspend")
    time.sleep(1.0)  # many polls: a suspended process is alive, so nothing is restarted
    assert sup.pid == pid and sup.restarts == 0 and "exited" not in names(tmp_path)
    sup.apply("resume")
    assert sup.pid == pid
    socket.create_connection(("127.0.0.1", port), timeout=2).close()
    sup.apply("kill")  # a suspended-then-resumed worker is still killable and respawned
    assert sup.wait_ready(WAIT, restarts=1)


def test_a_worker_killed_while_suspended_is_respawned_running(tmp_path, running):
    sup = running(make(tmp_path, script(tmp_path), free_port()))
    sup.apply("suspend")
    sup.apply("kill")
    assert sup.wait_ready(WAIT, restarts=1)
    assert sup.status()["state"] == "running"  # the new worker is not suspended


def test_a_request_with_no_worker_running_is_refused(tmp_path):
    sup = Supervisor([sys.executable, "-c", "pass"], poll_seconds=0.02)  # never started
    for action in ("kill", "suspend", "resume"):
        with pytest.raises(NoWorker):
            sup.apply(action)
    with pytest.raises(ValueError, match="action must be one of"):
        sup.apply("reboot")


def call(port, method, path, json_type=True):
    headers = {"Content-Type": "application/json"} if json_type else {}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def test_the_control_port_applies_kill_suspend_and_resume(tmp_path, running):
    sup = running(make(tmp_path, script(tmp_path), free_port()))
    server = serve_control(sup, ("127.0.0.1", 0))
    try:
        port = server.server_address[1]
        pid = sup.pid
        assert call(port, "GET", "/status")[1]["state"] == "running"
        assert call(port, "POST", "/suspend") == (
            200,
            {"ok": True, "action": "suspend", "pid": pid},
        )
        assert call(port, "GET", "/status")[1]["state"] == "suspended"
        assert call(port, "POST", "/resume")[1]["ok"] is True
        assert call(port, "POST", "/kill") == (200, {"ok": True, "action": "kill", "pid": pid})
        assert sup.wait_ready(WAIT, restarts=1)
        status = call(port, "GET", "/status")[1]
        assert status["restarts"] == 1 and status["pid"] != pid
    finally:
        server.shutdown()


def test_the_control_port_answers_errors(tmp_path):
    sup = Supervisor([sys.executable, "-c", "pass"], poll_seconds=0.02)  # no worker yet
    server = serve_control(sup, ("127.0.0.1", 0))
    try:
        port = server.server_address[1]
        code, body = call(port, "POST", "/kill")
        assert code == 409 and body["ok"] is False and "no worker" in body["error"]
        assert call(port, "POST", "/reboot")[0] == 404
        assert call(port, "GET", "/kill")[0] == 404
    finally:
        server.shutdown()


def test_a_kill_asked_for_through_the_supervisor_is_not_a_crash(tmp_path, running):
    # the default stable_seconds (5 s): the workers here live well under that
    sup = running(make(tmp_path, script(tmp_path), free_port()))
    for n in (1, 2, 3):
        sup.apply("kill")
        assert sup.wait_ready(WAIT, restarts=n)
    assert "backoff" not in names(tmp_path)


def test_a_worker_that_dies_soon_after_it_starts_listening_is_a_crash(tmp_path):
    # listens, then exits 0.3 s later (as a worker does on a rejected registration)
    worker = script(tmp_path, "time.sleep(0.3); sys.exit(1)")
    sup = make(tmp_path, worker, free_port(), backoff_seconds=(0.0, 0.2), stable_seconds=5.0)
    sup.start()
    try:
        wait_for(lambda: "backoff" in names(tmp_path), "a backoff after repeated early exits")
    finally:
        sup.stop()
    assert [e["seconds"] for e in events(tmp_path) if e["event"] == "backoff"][0] == 0.2


def test_a_worker_that_never_listens_is_killed_and_started_again(tmp_path):
    worker = tmp_path / "stuck.py"
    worker.write_text("import time\nwhile True:\n    time.sleep(0.05)\n")
    sup = Supervisor(
        [sys.executable, str(worker)],
        ready_address=("127.0.0.1", free_port()),  # nothing ever listens here
        ready_timeout=0.5,
        backoff_seconds=(0.0,),
        poll_seconds=0.02,
        log_path=tmp_path / "events.jsonl",
    )
    sup.start()
    try:
        wait_for(lambda: names(tmp_path).count("spawned") >= 2, "a second start")
    finally:
        sup.stop()
    assert "ready_timeout" in names(tmp_path) and "exited" in names(tmp_path)


def test_a_refusal_from_the_operating_system_is_a_500_not_a_dropped_connection(
    tmp_path, running, monkeypatch
):
    sup = running(make(tmp_path, script(tmp_path), free_port()))

    def refuse(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(faults, "kill", refuse)
    server = serve_control(sup, ("127.0.0.1", 0))
    try:
        code, body = call(server.server_address[1], "POST", "/kill")
        assert code == 500 and body["ok"] is False and "AccessDenied" in body["error"]
        monkeypatch.undo()
        sup.apply("kill")  # the refused request did not leave the next exit marked as asked for
        assert sup.wait_ready(WAIT, restarts=1)
    finally:
        server.shutdown()


def test_a_post_without_a_json_content_type_is_refused_and_does_nothing(tmp_path, running):
    sup = running(make(tmp_path, script(tmp_path), free_port()))
    server = serve_control(sup, ("127.0.0.1", 0))
    try:
        pid = sup.pid
        for path in ("/kill", "/suspend"):
            code, body = call(server.server_address[1], "POST", path, json_type=False)
            assert code == 415 and body["ok"] is False
        time.sleep(0.3)
        assert sup.pid == pid and sup.restarts == 0 and sup.status()["state"] == "running"
    finally:
        server.shutdown()


# ----------------------------------------------------------- back and registered

# What a worker does after it is listening: register with the gateway given as argv[2].
REGISTERING = """
import grpc
from pllm.control import register
with grpc.insecure_channel(sys.argv[2]) as channel:
    register(channel, 1, (15, 30), "smollm2-135m", "ab" * 32, "127.0.0.1:" + sys.argv[1])
"""


class Gateway:
    """The gateway's Control service on a free port, with a registry that accepts one stage."""

    def __init__(self, weights_sha256="ab" * 32, port=0):
        expected = ExpectedModel(bytes.fromhex(weights_sha256), 30, 15)
        self.registry = Registry("smollm2-135m", expected)
        self.server = grpc.server(futures.ThreadPoolExecutor(2))
        protocol_pb2_grpc.add_ControlServicer_to_server(ControlService(self.registry), self.server)
        self.port = self.server.add_insecure_port(f"127.0.0.1:{port}")
        self.server.start()

    def epoch(self):
        entry = self.registry.stages.get(1)
        return entry.epoch if entry else 0


def test_a_killed_worker_comes_back_and_registers_again(tmp_path, running):
    gateway = Gateway()
    try:
        worker = script(tmp_path, REGISTERING)
        port = free_port()
        sup = Supervisor(
            [sys.executable, str(worker), str(port), f"127.0.0.1:{gateway.port}"],
            ready_address=("127.0.0.1", port),
            poll_seconds=0.02,
            log_path=tmp_path / "events.jsonl",
        )
        running(sup)
        wait_for(lambda: gateway.epoch() == 1, "the first registration")
        faults.kill(sup.pid)
        assert sup.wait_ready(WAIT, restarts=1)
        wait_for(lambda: gateway.epoch() == 2, "the registration after the respawn")
        assert gateway.registry.stages[1].addr == f"127.0.0.1:{port}"
    finally:
        gateway.server.stop(0)


# ------------------------------------------------------- the worker's registration


class FakeServicer:
    own_epoch = 0


def weights_file(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not real weights")
    return sha256_of(tmp_path / "model.safetensors")


def test_a_worker_registers_and_keeps_its_epoch(tmp_path):
    digest = weights_file(tmp_path)
    gateway = Gateway(digest)
    servicer = FakeServicer()
    try:
        epoch = register_with_gateway(
            f"127.0.0.1:{gateway.port}",
            1,
            (15, 30),
            "smollm2-135m",
            tmp_path,
            "127.0.0.1:9",
            servicer,
            threading.Event(),
        )
    finally:
        gateway.server.stop(0)
    assert epoch == 1 == servicer.own_epoch == gateway.epoch()


def test_a_worker_that_starts_before_the_gateway_registers_when_it_comes_up(tmp_path):
    digest = weights_file(tmp_path)
    port = free_port()
    servicer, result = FakeServicer(), []
    thread = threading.Thread(
        target=lambda: result.append(
            register_with_gateway(
                f"127.0.0.1:{port}",
                1,
                (15, 30),
                "smollm2-135m",
                tmp_path,
                "127.0.0.1:9",
                servicer,
                threading.Event(),
                retry_seconds=0.05,
            )
        )
    )
    thread.start()
    time.sleep(0.5)  # several failed attempts: nothing listens yet
    assert thread.is_alive() and servicer.own_epoch == 0
    gateway = Gateway(digest, port=port)
    try:
        thread.join(WAIT)
        assert result == [1] and servicer.own_epoch == 1
    finally:
        gateway.server.stop(0)


def test_a_rejected_registration_is_raised_not_retried(tmp_path):
    weights_file(tmp_path)
    gateway = Gateway("cd" * 32)  # the gateway expects other weights
    try:
        with pytest.raises(grpc.RpcError) as err:
            register_with_gateway(
                f"127.0.0.1:{gateway.port}",
                1,
                (15, 30),
                "smollm2-135m",
                tmp_path,
                "127.0.0.1:9",
                FakeServicer(),
                threading.Event(),
            )
    finally:
        gateway.server.stop(0)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert gateway.epoch() == 0


def test_a_worker_stopped_while_waiting_for_the_gateway_gives_up(tmp_path):
    weights_file(tmp_path)
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    got = register_with_gateway(
        f"127.0.0.1:{free_port()}",
        1,
        (15, 30),
        "smollm2-135m",
        tmp_path,
        "127.0.0.1:9",
        FakeServicer(),
        stop,
        retry_seconds=0.05,
    )
    assert got is None


def test_an_error_other_than_unreachable_is_raised_not_retried(tmp_path):
    weights_file(tmp_path)
    server = grpc.server(futures.ThreadPoolExecutor(1))  # serves no Control service
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with pytest.raises(grpc.RpcError) as err:
            register_with_gateway(
                f"127.0.0.1:{port}",
                1,
                (15, 30),
                "smollm2-135m",
                tmp_path,
                "127.0.0.1:9",
                FakeServicer(),
                threading.Event(),
                retry_seconds=0.05,
            )
    finally:
        server.stop(0)
    assert err.value.code() == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.skipif(
    not (MODELS_DIR / "smollm2-135m" / "model.safetensors").exists(),
    reason="model weights not downloaded",
)
@pytest.mark.skipif(
    psutil.virtual_memory().available < 1.5 * 2**30, reason="needs 1.5 GiB of free memory"
)
def test_a_real_worker_killed_by_the_fault_injector_is_back_and_registered(tmp_path, monkeypatch):
    stage_port, super_port = free_port(), free_port()
    gateway_server = grpc.server(futures.ThreadPoolExecutor(2))
    registry = Registry("smollm2-135m", control.expected_model("smollm2-135m"))
    protocol_pb2_grpc.add_ControlServicer_to_server(ControlService(registry), gateway_server)
    gateway_port = gateway_server.add_insecure_port("127.0.0.1:0")
    gateway_server.start()
    hosts = tmp_path / "hosts.toml"
    roles = [
        ("gateway", gateway_port),
        ("stage0", free_port()),
        ("stage1", stage_port),
        ("supervisor1", super_port),
    ]
    tables = [f'[{role}]\nhost = "127.0.0.1"\nport = {port}\n' for role, port in roles]
    hosts.write_text('network = "test"\n' + "\n".join(tables))
    monkeypatch.setenv("PLLM_CONFIG", str(hosts))
    sup = Supervisor(
        worker_command("smollm2-135m", 1, 2),
        ready_address=("127.0.0.1", stage_port),
        log_path=tmp_path / "events.jsonl",
        cwd=config.REPO_ROOT,
    )
    sup.start()
    try:
        assert sup.wait_ready(180)
        wait_for(lambda: 1 in registry.stages, "the first registration", 120)
        assert registry.stages[1].epoch == 1
        faults.kill(sup.pid)
        assert sup.wait_ready(180, restarts=1)
        wait_for(lambda: registry.stages[1].epoch == 2, "the registration after the kill", 120)
        assert sup.last_respawn_seconds is not None
    finally:
        sup.stop()
        gateway_server.stop(0)


def test_a_worker_exits_when_its_supervisor_is_gone():
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    left = []
    thread = exit_when_parent_gone(gone.pid, interval=0.02, leave=left.append)
    thread.join(WAIT)
    assert left == [1]


def test_a_worker_stays_while_its_supervisor_lives():
    alive_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    left = []
    try:
        exit_when_parent_gone(alive_proc.pid, interval=0.02, leave=left.append)
        time.sleep(0.5)
        assert left == []
    finally:
        alive_proc.kill()
        alive_proc.wait()
    wait_for(lambda: left == [1], "the guard to notice")


def test_the_supervisor_gives_its_worker_its_own_pid(tmp_path, running):
    seen = tmp_path / "parent.txt"
    worker = script(
        tmp_path, f"import os; open({str(seen)!r}, 'w').write(os.environ['PLLM_SUPERVISOR_PID'])"
    )
    running(make(tmp_path, worker, free_port()))
    wait_for(seen.exists, "the worker to write its environment")
    assert seen.read_text() == str(os.getpid())


# --------------------------------------------------------------------- the CLI


def test_the_worker_command_names_the_stage_and_serves():
    command = worker_command("smollm2-135m", 1, 6)
    assert command[0] == sys.executable
    assert command[1:] == [
        "-m",
        "pllm.worker",
        "smollm2-135m",
        "--stage",
        "1",
        "--serve",
        "--threads",
        "6",
    ]
    assert "--threads" not in worker_command("smollm2-135m", 0, None)


def test_the_supervisor_port_follows_the_hosts_file(tmp_path):
    base = (
        'network = "t"\n'
        '[gateway]\nhost = "10.0.0.1"\nport = 1\n'
        '[stage0]\nhost = "10.0.0.2"\nport = 2\n'
        '[stage1]\nhost = "10.0.0.3"\nport = 3\n'
    )
    old = tmp_path / "old.toml"
    old.write_text(base)  # a file written before supervisors existed
    assert config.supervisor_endpoint(0, old) == config.Endpoint("10.0.0.2", 50061)
    assert config.supervisor_endpoint(1, old) == config.Endpoint("10.0.0.3", 50062)
    new = tmp_path / "new.toml"
    new.write_text(base + '[supervisor1]\nhost = "10.0.0.3"\nport = 7000\n')
    assert config.supervisor_endpoint(1, new) == config.Endpoint("10.0.0.3", 7000)
    assert config.supervisor_endpoint(0, new).port == 50061  # only the table that is present
    with pytest.raises(ValueError, match="stage must be 0 or 1"):
        config.supervisor_endpoint(2, new)
    bad = tmp_path / "bad.toml"
    bad.write_text(base + '[supervisor0]\nhost = "x"\nport = 0\n')
    with pytest.raises(ValueError, match="port must be an integer"):
        config.supervisor_endpoint(0, bad)


def test_the_committed_hosts_file_has_distinct_supervisor_ports():
    ports = {config.supervisor_endpoint(s).port for s in (0, 1)}
    hosts = config.load()
    assert len(ports) == 2 and not ports & {
        hosts.stage0.port,
        hosts.stage1.port,
        hosts.gateway.port,
    }
