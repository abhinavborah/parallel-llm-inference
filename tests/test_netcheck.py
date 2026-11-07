import json
import socket
import subprocess
import sys
import threading
import time

import pytest

from pllm import netcheck, runrecord


def test_summary_uses_nearest_rank_for_p95():
    values = [float(v) for v in range(1, 21)]  # 1 to 20 ms
    assert netcheck.summary(values) == {
        "min_ms": 1.0,
        "median_ms": 10.5,
        "p95_ms": 19.0,
        "max_ms": 20.0,
    }
    assert netcheck.summary([4.0])["p95_ms"] == 4.0


@pytest.fixture
def echo_server():
    server = netcheck.listen(0, host="127.0.0.1")
    stop = threading.Event()
    thread = threading.Thread(target=netcheck.serve_forever, args=(server, stop), daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    thread.join(timeout=5)


def test_probe_times_every_echo(echo_server):
    result = netcheck.probe("127.0.0.1", echo_server, count=10)
    assert result["count"] == 10
    assert 0 <= result["min_ms"] <= result["median_ms"] <= result["p95_ms"] <= result["max_ms"]
    assert result["connect_ms"] >= 0


def test_two_probes_can_run_at_once(echo_server):
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(netcheck.probe("127.0.0.1", echo_server, 5)))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(results) == 2


def test_probe_reports_a_closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # bound but not listening, so connecting is refused
        with pytest.raises(OSError):
            netcheck.probe("127.0.0.1", port, count=1, timeout=2)


def test_the_probe_command_records_each_role(echo_server, tmp_path, monkeypatch, capsys):
    hosts = tmp_path / "hosts.toml"
    hosts.write_text(
        'network = "test-lan"\n'
        f'[gateway]\nhost = "127.0.0.1"\nport = {echo_server}\n'
        '[stage0]\nhost = "127.0.0.1"\nport = 1\n'
        '[stage1]\nhost = "127.0.0.1"\nport = 2\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PLLM_CONFIG", str(hosts))
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    netcheck.main(["probe", "--roles", "gateway", "--count", "3"])

    (path,) = (tmp_path / "runs").glob("*-network-check.json")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["status"] == "ok"
    assert rec["network"] == "test-lan"
    assert rec["results"]["gateway"]["address"] == f"127.0.0.1:{echo_server}"
    assert rec["results"]["gateway"]["count"] == 3
    assert "gateway 127.0.0.1" in capsys.readouterr().out


def test_multicast_message_round_trips():
    message = netcheck.parse_multicast(netcheck.multicast_message(7))
    assert message == {"sender": socket.gethostname(), "seq": 7}
    assert netcheck.parse_multicast(b"something else") is None


def test_multicast_reaches_a_listener_on_the_same_host():
    try:
        receiver = netcheck.multicast_receiver(port=0)
    except OSError as err:
        pytest.skip(f"multicast not available here: {err}")
    with receiver, netcheck.multicast_sender() as sender:
        port = receiver.getsockname()[1]
        receiver.settimeout(0.2)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                sender.sendto(netcheck.multicast_message(1), (netcheck.MCAST_GROUP, port))
            except OSError as err:
                pytest.skip(f"no multicast route here: {err}")
            try:
                data, _ = receiver.recvfrom(1024)
            except TimeoutError:
                continue
            assert netcheck.parse_multicast(data)["seq"] == 1
            return
    pytest.skip("multicast loopback did not deliver on this host")


WANTED = [("TCP", 50051), ("UDP", netcheck.MCAST_PORT)]


def rule(action, protocol="TCP", port=50051, profile="Private"):
    return {
        "name": f"{action} {port}",
        "action": action,
        "profile": profile,
        "protocol": protocol,
        "port": port,
        "ports": str(port),
        "program": "Any",
    }


def test_firewall_verdict_per_port_and_public_networks():
    raw = json.dumps(
        {
            "profiles": [
                {"interface": "Ethernet", "name": "lab", "category": "Private"},
                {"interface": "VPN", "name": "vpn", "category": "Public"},
            ],
            "rules": [rule("Allow"), rule("Allow", "UDP", netcheck.MCAST_PORT, "Public")],
            "rules_error": None,
        }
    )
    report = netcheck.parse_firewall(raw, WANTED)
    assert report["allowed_on_private"] == {"TCP 50051": True, f"UDP {netcheck.MCAST_PORT}": False}
    assert report["public_networks"] == ["vpn"]


def test_a_block_rule_wins_over_an_allow_rule():
    raw = json.dumps({"profiles": [], "rules": [rule("Allow"), rule("Block", profile="Any")]})
    assert netcheck.parse_firewall(raw, WANTED[:1])["allowed_on_private"] == {"TCP 50051": False}


def test_a_single_profile_object_is_read_as_a_list():
    raw = json.dumps({"profiles": {"interface": "e", "name": "n", "category": "Private"}})
    report = netcheck.parse_firewall(raw, WANTED[:1])
    assert report["profiles"] == [{"interface": "e", "name": "n", "category": "Private"}]
    assert report["rules"] == []


def test_unreadable_rules_give_no_verdict_instead_of_not_allowed():
    raw = json.dumps({"profiles": [], "rules": [], "rules_error": "Access is denied."})
    report = netcheck.parse_firewall(raw, WANTED)
    assert set(report["allowed_on_private"].values()) == {None}
    assert report["rules_error"] == "Access is denied."


def test_the_firewall_script_names_this_interpreter_and_no_other_python(monkeypatch):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, stdout='{"profiles": [], "rules": []}')

    monkeypatch.setattr(netcheck.subprocess, "run", fake_run)
    monkeypatch.setattr(netcheck.sys, "executable", r"C:\Envs\O'Neil\venv\python.exe")
    monkeypatch.setattr(netcheck.sys, "_base_executable", r"C:\Python\python.exe", raising=False)
    netcheck.firewall(WANTED)
    (script,) = seen
    assert "'*python*'" not in script
    assert "O''Neil" in script  # a quote in a path cannot end the PowerShell string
    assert "python.exe" in script and "Python" in script


def test_interpreters_lists_the_running_and_the_base_interpreter(monkeypatch):
    monkeypatch.setattr(netcheck.sys, "executable", "/envs/venv/bin/python")
    monkeypatch.setattr(netcheck.sys, "_base_executable", "/usr/bin/python3", raising=False)
    found = [p.replace("\\", "/") for p in netcheck.interpreters()]
    assert any(p.endswith("envs/venv/bin/python") for p in found)
    assert any(p.endswith("usr/bin/python3") for p in found)
    assert len(found) == len(set(found))


def test_a_rule_line_names_its_program():
    rule = {
        "name": "python",
        "action": "Allow",
        "protocol": "TCP",
        "ports": "Any",
        "profile": "Private",
        "program": r"C:\Python\python.exe",
    }
    assert netcheck.describe_rule(rule).endswith(r"program C:\Python\python.exe")
    assert netcheck.describe_rule(rule | {"program": ""}).endswith("program Any")


def test_the_firewall_check_covers_the_role_ports_and_both_udp_ports():
    wanted = netcheck.wanted_ports(["stage0", "gateway"])
    assert wanted[-2:] == [("UDP", 50059), ("UDP", 5353)]
    assert len([p for p in wanted if p[0] == "TCP"]) == 2


def test_probe_rejects_a_count_below_one(capsys):
    for count in ("0", "-3"):
        with pytest.raises(SystemExit) as stop:
            netcheck.main(["probe", "--roles", "stage0", "--count", count])
        assert stop.value.code == 2
    assert "--count must be at least 1" in capsys.readouterr().err


def test_a_failing_powershell_run_is_raised(monkeypatch):
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")

    monkeypatch.setattr(netcheck.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="boom"):
        netcheck.firewall(WANTED)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows firewall only")
def test_the_firewall_query_runs_on_windows():
    report = netcheck.firewall(WANTED)
    assert report["profiles"], "every Windows host has a network profile"
    if report["rules_error"] is None:
        assert set(report["allowed_on_private"]) == {"TCP 50051", f"UDP {netcheck.MCAST_PORT}"}
    else:
        assert set(report["allowed_on_private"].values()) == {None}


def test_a_failed_role_is_recorded_with_the_others(echo_server, tmp_path, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
        hosts = tmp_path / "hosts.toml"
        hosts.write_text(
            'network = "test-lan"\n'
            f'[gateway]\nhost = "127.0.0.1"\nport = {echo_server}\n'
            f'[stage0]\nhost = "127.0.0.1"\nport = {closed}\n'
            '[stage1]\nhost = "127.0.0.1"\nport = 2\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("PLLM_CONFIG", str(hosts))
        monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(netcheck, "probe", _probe_with_short_timeout)
        with pytest.raises(SystemExit):
            netcheck.main(["probe", "--roles", "stage0", "gateway", "--count", "2"])

    (path,) = (tmp_path / "runs").glob("*-network-check.json")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["status"] == "error"
    assert "error" in rec["results"]["stage0"]
    assert rec["results"]["gateway"]["count"] == 2


_real_probe = netcheck.probe


def _probe_with_short_timeout(host, port, count):
    return _real_probe(host, port, count, timeout=2)
