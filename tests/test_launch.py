import signal
import subprocess
import sys
import threading
import time

import psutil
import pytest

from pllm import config, launch


def test_threads_split_the_cores_between_this_laptops_stages():
    assert launch.default_threads(["gateway", "stage0", "stage1"], cores=12) == 6
    assert launch.default_threads(["stage0"], cores=6) == 6
    assert launch.default_threads(["gateway", "stage1"], cores=12) == 12
    assert launch.default_threads(["stage0", "stage1"], cores=1) == 1


def test_commands_start_the_gateway_first_through_the_same_entry_points():
    all_roles = launch.commands("smollm2-135m", ["stage1", "stage0", "gateway"], 3)
    assert list(all_roles) == ["gateway", "stage0", "stage1"]
    assert all_roles["gateway"][1:] == ["-m", "pllm.gateway", "smollm2-135m"]
    expected = ["-m", "pllm.supervisor", "smollm2-135m", "--stage", "1", "--threads", "3"]
    assert all_roles["stage1"][1:] == expected
    # one laptop's share is the same command as in single-host mode
    assert launch.commands("smollm2-135m", ["stage1"], 3) == {"stage1": all_roles["stage1"]}


def hosts_with_stage0_on(host):
    local = config.Endpoint("127.0.0.1", 50050)
    other = config.Endpoint(host, 50051)
    return config.HostConfig("test", local, other, config.Endpoint("127.0.0.1", 50052))


def test_a_role_placed_on_another_laptop_is_named():
    hosts = hosts_with_stage0_on("10.1.2.3")
    assert launch.elsewhere(hosts, ["gateway", "stage1"], {"127.0.0.1"}) == []
    assert launch.elsewhere(hosts, ["stage0"], {"127.0.0.1"}) == ["stage0 (10.1.2.3)"]
    assert launch.elsewhere(hosts, ["stage0"], {"127.0.0.1", "10.1.2.3"}) == []


SLEEP = [sys.executable, "-c", "import time; time.sleep(120)"]
QUIT = [sys.executable, "-c", "raise SystemExit(3)"]


def running(pid):
    try:
        return psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return False


def alive(pids, settle=5.0):
    """The pids still running. A stop has waited for each exit, but Windows can report the
    process for a moment while it is torn down, so allow a short settle."""
    end = time.monotonic() + settle
    while (left := [p for p in pids if running(p)]) and time.monotonic() < end:
        time.sleep(0.05)
    return left


def children_of_this_test():
    return {p.pid for p in psutil.Process().children()}


@pytest.mark.parametrize("failing", ["first", "last"])
def test_a_process_that_exits_stops_the_rest(failing):
    to_start = {"a": QUIT, "b": SLEEP} if failing == "first" else {"a": SLEEP, "b": QUIT}
    before = children_of_this_test()
    assert launch.run(to_start, threading.Event(), poll=0.05) == 1
    assert children_of_this_test() - before == set()


def test_a_requested_stop_leaves_nothing_behind():
    done = threading.Event()
    before = children_of_this_test()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(launch.run({"a": SLEEP, "b": SLEEP}, done, poll=0.05))
    )
    thread.start()
    deadline = time.monotonic() + 30
    while len(children_of_this_test() - before) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    started = children_of_this_test() - before
    assert len(started) == 2
    done.set()
    thread.join(60)
    assert result == [0]
    assert alive(started) == []


def test_a_process_that_ignores_the_stop_is_killed(monkeypatch, tmp_path):
    ready = tmp_path / "ready"
    stubborn = [
        sys.executable,
        "-c",
        "import pathlib, signal, sys, time\n"
        "for name in ('SIGINT', 'SIGBREAK'):\n"
        "    if hasattr(signal, name):\n"
        "        signal.signal(getattr(signal, name), signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).touch()\n"
        "time.sleep(120)\n",
        str(ready),
    ]
    monkeypatch.setattr(launch, "STOP_SECONDS", 0.5)
    killed = []
    real_kill = launch.faults.kill
    monkeypatch.setattr(launch.faults, "kill", lambda pid: (killed.append(pid), real_kill(pid)))
    done = threading.Event()

    def stop_once_ready():
        deadline = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        done.set()

    before = children_of_this_test()
    threading.Thread(target=stop_once_ready, daemon=True).start()
    assert launch.run({"a": stubborn}, done, poll=0.05) == 0
    assert len(killed) == 1
    assert children_of_this_test() - before == set()


def test_a_command_that_cannot_start_stops_those_already_running():
    before = children_of_this_test()
    with pytest.raises(OSError):
        launch.run({"a": SLEEP, "b": ["no-such-program-pllm"]}, threading.Event())
    assert children_of_this_test() - before == set()


DRIVER = """
import pathlib, sys
from pllm import launch, runrecord
tmp = pathlib.Path(sys.argv[1])
(tmp / "smollm2-135m").mkdir()
for name in launch.PINNED["smollm2-135m"].files:
    (tmp / "smollm2-135m" / name).touch()
launch.MODELS_DIR = tmp
runrecord.RUNS_DIR = tmp / "runs"
child = "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
child += "time.sleep(120)"
launch.commands = lambda *_: {"gateway": [sys.executable, "-c", child, str(tmp / "pid")]}
launch.main(["smollm2-135m"])
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; Windows stops with Ctrl-Break")
@pytest.mark.parametrize("name", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_a_signal_to_the_launcher_stops_its_processes(tmp_path, name):
    launcher = subprocess.Popen([sys.executable, "-c", DRIVER, str(tmp_path)])
    pid_file = tmp_path / "pid"
    deadline = time.monotonic() + 30
    while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.02)
    child = int(pid_file.read_text())
    launcher.send_signal(getattr(signal, name))
    assert launcher.wait(30) == 0
    assert alive([child]) == []


def run_main(args):
    with pytest.raises(SystemExit) as stop:
        launch.main(args)
    return stop.value.code


def test_main_refuses_a_role_the_hosts_file_places_elsewhere(tmp_path, monkeypatch, capsys):
    text = config.DEFAULT_PATH.read_text(encoding="utf-8").replace(
        '[stage0]\nhost = "127.0.0.1"', '[stage0]\nhost = "10.1.2.3"'
    )
    path = tmp_path / "hosts.toml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("PLLM_CONFIG", str(path))
    assert run_main(["smollm2-135m", "--roles", "stage0"]) == 2
    assert "stage0 (10.1.2.3)" in capsys.readouterr().err


def test_main_names_the_download_when_weights_are_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "MODELS_DIR", tmp_path)
    assert run_main(["smollm2-135m"]) == 2
    assert "uv run python -m pllm.models smollm2-135m" in capsys.readouterr().err


def test_main_refuses_a_partial_download(tmp_path, monkeypatch, capsys):
    (tmp_path / "smollm2-135m").mkdir()
    (tmp_path / "smollm2-135m" / "config.json").touch()
    monkeypatch.setattr(launch, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(launch, "run", lambda *_: 0)  # a passed check must not start anything
    assert run_main(["smollm2-135m", "--roles", "stage1"]) == 2
    err = capsys.readouterr().err
    assert "model.safetensors" in err and "uv run python -m pllm.models smollm2-135m" in err


def test_main_rejects_zero_threads(capsys):
    assert run_main(["smollm2-135m", "--threads", "0"]) == 2
