"""Start this laptop's processes: one command, the same on macOS and Windows.

Each role runs through its own entry point, and a stage runs under its supervisor. With no
--roles every process starts on this laptop (single-host mode). Addresses come only from the
hosts file, so moving to another network never changes this command.
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence

import psutil

from pllm import config, faults, runrecord
from pllm.models import MODELS_DIR, PINNED

log = logging.getLogger("pllm.launch")

STOP_SECONDS = 40.0  # longer than the supervisor's own 30 s stop
if sys.platform == "win32":
    # A signal reaches a child on Windows only through its own process group.
    STOP_SIGNAL = signal.CTRL_BREAK_EVENT
    GROUP = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
else:
    STOP_SIGNAL = signal.SIGINT
    GROUP = {"start_new_session": True}


def default_threads(roles: Iterable[str], cores: int | None = None) -> int:
    """Physical cores shared by this laptop's stages, so on one host each gets half (R8.9)."""
    cores = cores or psutil.cpu_count(logical=False) or os.cpu_count() or 1
    stages = sum(role != "gateway" for role in roles)
    return max(1, cores // max(1, stages))


def commands(model: str, roles: Iterable[str], threads: int) -> dict[str, list[str]]:
    """The command of each role in start order: the stages register with the gateway."""
    wanted = set(roles)
    out = {}
    for role in config.ROLES:
        if role not in wanted:
            continue
        if role == "gateway":
            out[role] = [sys.executable, "-m", "pllm.gateway", model]
        else:
            stage = role.removeprefix("stage")
            out[role] = [
                sys.executable,
                "-m",
                "pllm.supervisor",
                model,
                "--stage",
                stage,
                "--threads",
                str(threads),
            ]
    return out


def this_laptop() -> set[str]:
    return {a.address for addrs in psutil.net_if_addrs().values() for a in addrs}


def elsewhere(hosts: config.HostConfig, roles: Iterable[str], local: set[str]) -> list[str]:
    """Roles whose address in the hosts file is not this laptop's, with that address."""

    def is_local(host: str) -> bool:
        try:
            return socket.gethostbyname(host) in local
        except OSError:
            return False

    return [
        f"{role} ({getattr(hosts, role).host})"
        for role in roles
        if not is_local(getattr(hosts, role).host)
    ]


def _stop(children: dict[str, subprocess.Popen]) -> None:
    order = list(reversed(children.items()))
    for _, child in order:
        try:
            child.send_signal(STOP_SIGNAL)
        except OSError:  # it exited just now; the others must still be stopped
            pass
    deadline = time.monotonic() + STOP_SECONDS
    for role, child in order:
        try:
            child.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            log.warning("%s did not stop; killing its process tree", role)
            faults.kill(child.pid)
            child.wait()


def run(to_start: dict[str, list[str]], done: threading.Event, poll: float = 0.2) -> int:
    """Start each command in order and stop them all when done is set or any one exits.

    A process that exits by itself ends the run: a supervisor never exits on its worker's
    behalf, so this means a real failure. Returns 0 for a requested stop, otherwise 1.
    """
    children: dict[str, subprocess.Popen] = {}
    try:
        for role, command in to_start.items():
            children[role] = subprocess.Popen(command, cwd=config.REPO_ROOT, **GROUP)
            log.info("started %s, pid %d", role, children[role].pid)
        while not done.wait(poll):  # a timeout, so a signal handler can run while this waits
            for role, child in children.items():
                if child.poll() is not None:
                    log.error("%s exited with code %s; stopping", role, child.returncode)
                    return 1
        return 0
    finally:
        _stop(children)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start this laptop's processes.")
    parser.add_argument("model", choices=sorted(PINNED))
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=config.ROLES,
        default=list(config.ROLES),
        help="what runs on this laptop (default: everything, single-host mode)",
    )
    parser.add_argument("--threads", type=int, help="torch threads per stage")
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    try:
        hosts = config.load()
        split = config.split_point(args.model)
    except (ValueError, OSError) as err:
        parser.error(str(err))
    wrong = elsewhere(hosts, args.roles, this_laptop())
    if wrong:
        parser.error(f"the hosts file does not place these on this laptop: {', '.join(wrong)}")
    # A stage that cannot load is restarted for ever by its supervisor, so refuse here.
    missing = [f for f in PINNED[args.model].files if not (MODELS_DIR / args.model / f).is_file()]
    if missing:
        parser.error(
            f"model files missing ({', '.join(sorted(missing))}): "
            f"uv run python -m pllm.models {args.model}"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    threads = args.threads or default_threads(args.roles)
    done = threading.Event()
    # SIGBREAK exists on Windows only. SIGHUP is a closed terminal: the children are in their
    # own sessions, so without this they would outlive the launcher and keep their ports.
    for name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: done.set())

    with runrecord.record(
        "launch",
        model=args.model,
        revision=PINNED[args.model].revision,
        split=split,
        threads=threads,
    ) as rec:
        rec["roles"] = list(args.roles)
        power = rec["power"]
        log.info(
            "network %s, available memory %.2f GiB, power %s",
            rec["network"],
            rec["available_memory_bytes"] / 1024**3,
            power,
        )
        if power is not None and not power["plugged_in"]:
            log.warning("on battery: use the same power state for every run")
        code = run(commands(args.model, args.roles, threads), done)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
