"""Network check between the two laptops: TCP on the role ports, round-trip time,
multicast, and on Windows the firewall rules and network profile.

Addresses and ports come from the hosts file, so the check also tests the file that
the workers will use. With Stage 0 on Windows and Stage 1 plus the gateway on macOS:

    Windows:  uv run python -m pllm.netcheck serve --roles stage0
    macOS:    uv run python -m pllm.netcheck probe --roles stage0
    macOS:    uv run python -m pllm.netcheck serve --roles stage1 gateway
    Windows:  uv run python -m pllm.netcheck probe --roles stage1 gateway

Multicast, on the mDNS group used for discovery (with its own port, so it does not
compete with the system's mDNS responder). With several network adapters, pass
--interface with this laptop's LAN address so the right adapter is used:

    one laptop:    uv run python -m pllm.netcheck multicast-listen
    the other:     uv run python -m pllm.netcheck multicast-send

Windows only, from an administrator PowerShell (reading firewall rules needs it): the
network profile of each adapter and the inbound rules for the role ports, the multicast
test port and UDP 5353, which discovery will use. Only rules that apply to any program or
to the interpreter running the check count.

    uv run python -m pllm.netcheck firewall --roles stage0

Every probe, listen and firewall run writes a run record.
"""

import argparse
import json
import math
import os
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time

from pllm import config, runrecord

PAYLOAD = 64
MCAST_GROUP = "224.0.0.251"
MCAST_PORT = 50059
# The port discovery (R9.7, optional) will announce on. The multicast test above uses its
# own port so it does not compete with the system's mDNS responder, so the firewall check
# looks at this one as well.
MDNS_PORT = 5353
MCAST_TAG = b"pllm-netcheck"


def summary(values_ms: list[float]) -> dict:
    """Minimum, median, 95th percentile (nearest rank) and maximum, in milliseconds."""
    ordered = sorted(values_ms)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p95_ms": p95,
        "max_ms": ordered[-1],
    }


def _recv_exactly(conn: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        data += chunk
    return data


def _echo(conn: socket.socket) -> None:
    with conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while chunk := conn.recv(4096):
            conn.sendall(chunk)


def listen(port: int, host: str = "0.0.0.0") -> socket.socket:
    server = socket.create_server((host, port))
    server.settimeout(0.5)
    return server


def serve_forever(server: socket.socket, stop: threading.Event) -> None:
    """Echo every connection on `server` until `stop` is set."""
    with server:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            conn.settimeout(None)
            threading.Thread(target=_echo, args=(conn,), daemon=True).start()


def probe(host: str, port: int, count: int, timeout: float = 5.0) -> dict:
    """Connect, then time `count` echoes of a small message."""
    start = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout) as conn:
        connect_ms = (time.perf_counter() - start) * 1000
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rtts = []
        for i in range(count):
            message = struct.pack("!I", i).ljust(PAYLOAD, b".")
            sent = time.perf_counter()
            conn.sendall(message)
            if _recv_exactly(conn, PAYLOAD) != message:
                raise ConnectionError("echo did not match what was sent")
            rtts.append((time.perf_counter() - sent) * 1000)
    return {"connect_ms": connect_ms, "count": count, **summary(rtts)}


def multicast_sender(interface: str = "0.0.0.0") -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    if interface != "0.0.0.0":
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(interface))
    return sock


def multicast_receiver(port: int = MCAST_PORT, interface: str = "0.0.0.0") -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    group = struct.pack("4s4s", socket.inet_aton(MCAST_GROUP), socket.inet_aton(interface))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group)
    return sock


def multicast_message(seq: int) -> bytes:
    return b" ".join([MCAST_TAG, socket.gethostname().encode(), str(seq).encode()])


def parse_multicast(data: bytes) -> dict | None:
    parts = data.split(b" ")
    if len(parts) != 3 or parts[0] != MCAST_TAG:
        return None
    return {"sender": parts[1].decode(errors="replace"), "seq": int(parts[2])}


# Profiles need no rights; rules need an administrator shell, so an error there is
# reported, never read as "no rule". A rule counts when its local port is the port or a
# range holding it, or any port for one of the interpreters that run this check (what the
# "Allow access" prompt creates), and its program is any program or one of those
# interpreters. Another python on the laptop does not count, and neither does a rule that
# names its program with an environment variable or a short path. Block rules are listed too:
# on Windows a block rule wins over an allow rule.
FIREWALL_SCRIPT = """
$ErrorActionPreference = 'Stop'
$wanted = @({wanted})
$interpreters = @({interpreters})
$profiles = @(Get-NetConnectionProfile | ForEach-Object {{
  [pscustomobject]@{{ interface = $_.InterfaceAlias; name = $_.Name;
                      category = $_.NetworkCategory.ToString() }} }})
function Covers($localPorts, $port) {{
  foreach ($p in @($localPorts)) {{
    if ("$p" -eq "$port") {{ return $true }}
    if ("$p" -match '^(\\d+)-(\\d+)$' -and [int]$matches[1] -le $port -and
        $port -le [int]$matches[2]) {{ return $true }}
  }}
  return $false
}}
function Row($rule, $filter, $want) {{
  $program = ($rule | Get-NetFirewallApplicationFilter).Program
  [pscustomobject]@{{ name = $rule.DisplayName; action = $rule.Action.ToString();
    profile = $rule.Profile.ToString(); protocol = $want.protocol; port = $want.port;
    ports = (@($filter.LocalPort) -join ','); program = "$program" }}
}}
$rules = @()
$rulesError = $null
try {{
  foreach ($want in $wanted) {{
    $rules += @(Get-NetFirewallPortFilter -Protocol $want.protocol |
      Where-Object {{ Covers $_.LocalPort $want.port }} | ForEach-Object {{
        $f = $_
        Get-NetFirewallRule -AssociatedNetFirewallPortFilter $f |
          Where-Object {{ $_.Direction -eq 'Inbound' -and $_.Enabled -eq 'True' }} |
          ForEach-Object {{ Row $_ $f $want }} |
      Where-Object {{ $_.program -eq 'Any' -or $interpreters -contains $_.program }} }})
    $rules += @(Get-NetFirewallApplicationFilter |
      Where-Object {{ $interpreters -contains $_.Program }} | ForEach-Object {{
        $r = Get-NetFirewallRule -AssociatedNetFirewallApplicationFilter $_
        $f = $r | Get-NetFirewallPortFilter
        if ($r.Direction -eq 'Inbound' -and $r.Enabled -eq 'True' -and
            @('Any', $want.protocol) -contains "$($f.Protocol)" -and
            @($f.LocalPort) -contains 'Any') {{ Row $r $f $want }} }})
  }}
}} catch {{ $rulesError = $_.Exception.Message }}
@{{ profiles = $profiles; rules = $rules; rules_error = $rulesError }} |
  ConvertTo-Json -Depth 4 -Compress
"""


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _on_private(rule: dict) -> bool:
    return rule["profile"] == "Any" or "Private" in rule["profile"]


def parse_firewall(raw: str, wanted: list[tuple[str, int]]) -> dict:
    """Profiles, matching rules and a verdict per wanted (protocol, port).

    The verdict is None when the rules could not be read, True when an allow rule
    covers the port on the Private profile and no block rule does, else False.
    """
    data = json.loads(raw)
    rules = _as_list(data.get("rules"))
    error = data.get("rules_error")
    verdicts = {}
    for protocol, port in wanted:
        matching = [
            r for r in rules if r["protocol"] == protocol and r["port"] == port and _on_private(r)
        ]
        allowed = any(r["action"] == "Allow" for r in matching)
        blocked = any(r["action"] == "Block" for r in matching)
        verdicts[f"{protocol} {port}"] = None if error else allowed and not blocked
    profiles = _as_list(data.get("profiles"))
    return {
        "profiles": profiles,
        "public_networks": [p["name"] for p in profiles if p["category"] != "Private"],
        "rules": rules,
        "rules_error": error,
        "allowed_on_private": verdicts,
    }


def interpreters() -> list[str]:
    """The interpreter running this check and the one a virtual environment launches.

    In a virtual environment on Windows python.exe is a launcher, and the process that
    listens is the base interpreter, so a rule can name either of them.
    """
    found = []
    for path in (sys.executable, getattr(sys, "_base_executable", "")):
        for candidate in (path, os.path.realpath(path)) if path else ():
            candidate = os.path.normpath(candidate)
            if candidate not in found:
                found.append(candidate)
    return found


def describe_rule(rule: dict) -> str:
    return (
        f"rule {rule['name']}: {rule['action']} {rule['protocol']} {rule['ports']}, "
        f"{rule['profile']}, program {rule.get('program') or 'Any'}"
    )


def wanted_ports(roles: list[str]) -> list[tuple[str, int]]:
    """The role ports, this check's multicast port and the discovery port."""
    tcp = [("TCP", e.port) for e in _endpoints(roles).values()]
    return tcp + [("UDP", MCAST_PORT), ("UDP", MDNS_PORT)]


def firewall(wanted: list[tuple[str, int]], programs: list[str] | None = None) -> dict:
    spec = ",".join(f"@{{protocol='{proto}'; port={port}}}" for proto, port in wanted)
    quoted = ",".join("'" + p.replace("'", "''") + "'" for p in programs or interpreters())
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            FIREWALL_SCRIPT.format(wanted=spec, interpreters=quoted),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"PowerShell firewall query failed: {out.stderr.strip()}")
    return parse_firewall(out.stdout, wanted)


def _endpoints(roles: list[str]) -> dict[str, config.Endpoint]:
    hosts = config.load()
    return {role: getattr(hosts, role) for role in roles}


def _probe_roles(roles: list[str], count: int) -> None:
    failed = []
    with runrecord.record("network-check") as rec:
        results = {}
        rec.update(check="tcp", from_host=socket.gethostname(), results=results)
        for role, endpoint in _endpoints(roles).items():
            try:
                result = probe(endpoint.host, endpoint.port, count)
            except OSError as err:
                results[role] = {"address": endpoint.address, "error": str(err)}
                failed.append(role)
                print(f"{role} {endpoint.address}: FAILED, {err}")
                continue
            results[role] = {"address": endpoint.address, **result}
            print(
                f"{role} {endpoint.address}: connect {result['connect_ms']:.2f} ms, "
                f"rtt median {result['median_ms']:.2f} ms, p95 {result['p95_ms']:.2f} ms"
            )
        if failed:
            # Raised inside the record, so it is written with status error and every result.
            raise SystemExit(f"no connection for {', '.join(failed)}")


def _firewall(parser: argparse.ArgumentParser, roles: list[str]) -> None:
    if sys.platform != "win32":
        parser.error("the firewall check is for Windows")
    wanted = wanted_ports(roles)
    with runrecord.record("network-check") as rec:
        report = firewall(wanted)
        rec.update(check="firewall", **report)
    for p in report["profiles"]:
        print(f"network {p['name']} on {p['interface']}: {p['category']}")
    if report["public_networks"]:
        print("the adapter on the laptops' LAN must be Private; set it in Windows Settings")
    if report["rules_error"]:
        print(f"could not read the firewall rules: {report['rules_error']}")
        print("run this check again from an administrator PowerShell")
        return
    for r in report["rules"]:
        print(describe_rule(r))
    for target, ok in report["allowed_on_private"].items():
        print(f"{target}: {'allowed' if ok else 'NOT allowed'} on the Private profile")
        if not ok:
            protocol, port = target.split()
            print(
                f'  New-NetFirewallRule -DisplayName "pllm {port}" -Direction Inbound '
                f"-Protocol {protocol} -LocalPort {port} -Profile Private -Action Allow"
            )
    print(f"UDP {MDNS_PORT} is mDNS, which discovery will use; UDP {MCAST_PORT} is only the")
    print("port of this check's multicast test.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    roles = {"nargs": "+", "choices": config.ROLES, "required": True}
    interface = {"default": "0.0.0.0", "help": "this laptop's LAN address"}
    sub.add_parser("serve").add_argument("--roles", **roles)
    p = sub.add_parser("probe")
    p.add_argument("--roles", **roles)
    p.add_argument("--count", type=int, default=50)
    p = sub.add_parser("multicast-listen")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--interface", **interface)
    p = sub.add_parser("multicast-send")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--interface", **interface)
    sub.add_parser("firewall").add_argument("--roles", **roles)
    args = parser.parse_args(argv)
    if args.command == "probe" and args.count < 1:
        parser.error("--count must be at least 1")

    if args.command == "serve":
        stop = threading.Event()
        threads = []
        for role, endpoint in _endpoints(args.roles).items():
            server = listen(endpoint.port)
            threads.append(threading.Thread(target=serve_forever, args=(server, stop)))
            print(f"echoing for {role} on port {endpoint.port}; Ctrl+C to stop")
        for t in threads:
            t.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            stop.set()
    elif args.command == "probe":
        _probe_roles(args.roles, args.count)
    elif args.command == "multicast-listen":
        with runrecord.record("network-check") as rec:
            heard = []
            rec.update(check="multicast", group=MCAST_GROUP, port=MCAST_PORT, heard=heard)
            rec.update(interface=args.interface)
            with multicast_receiver(interface=args.interface) as sock:
                sock.settimeout(0.5)
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    try:
                        data, addr = sock.recvfrom(1024)
                    except TimeoutError:
                        continue
                    if (message := parse_multicast(data)) is not None:
                        heard.append({"from": addr[0], **message})
                        print(f"heard {message['sender']} ({addr[0]}) seq {message['seq']}")
        print(f"{len(heard)} messages heard")
    elif args.command == "multicast-send":
        with multicast_sender(args.interface) as sock:
            for seq in range(args.count):
                sock.sendto(multicast_message(seq), (MCAST_GROUP, MCAST_PORT))
                time.sleep(0.5)
        print(f"sent {args.count} messages to {MCAST_GROUP}:{MCAST_PORT}")
    elif args.command == "firewall":
        _firewall(parser, args.roles)


if __name__ == "__main__":
    main()
