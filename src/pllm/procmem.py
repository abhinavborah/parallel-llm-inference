"""Memory and page fault counters of a process tree, for X9 (R7.16, R8.7).

A worker is a process tree: on Windows a venv's python.exe is a small launcher with the
real interpreter as its child, so every number is summed over the root and its descendants.
This module imports no torch, so the Host B monitor (pllm.x9monitor) stays small.

Per process:

- rss_bytes: the resident set (Windows: working set) now.
- peak_rss_bytes: the lifetime peak of that process. Windows (psutil peak_wset) and Linux
  (VmHWM) report it for another process; macOS does not, so there it is None. A tree's
  sum of per-process peaks is an upper bound on the largest size the tree had at one
  time, since the peaks need not coincide.
- page_faults: every page fault, most of which need no disk. On Windows (psutil
  num_page_faults) this includes soft faults, such as the first touch of a zeroed page or
  of a page still in memory, so it cannot show paging. None on Linux and unknown systems.
- disk_faults: only faults that had to read from disk (see DISK_COUNTERS). None where it
  cannot be read.

What disk_faults can show: a difference of 0 between two readings means the counter saw
the process wait for no page from disk in between. What it cannot show: whether a counted
read came from the page file or from a mapped file such as a program library or the
weights file; pages the system wrote out and the process never read back; or paging by
other processes.

An unreadable value is None, never 0.
"""

import ctypes
import sys
from collections.abc import Callable
from pathlib import Path

import psutil

DISK_COUNTERS = {
    "win32": (
        "per-process HardFaultCount from NtQuerySystemInformation(SystemProcessInformation): "
        "faults that read from disk (page file or a mapped file). Microsoft does not document "
        "this field; it is used only when the documented fields after it match psutil for "
        "the reading process"
    ),
    "darwin": "per-process pageins from psutil: pages read in from disk (swap or a mapped file)",
    "linux": "per-process majflt from /proc/PID/stat: faults that needed a read from disk",
}


def disk_counter_name(platform: str = sys.platform) -> str | None:
    return DISK_COUNTERS.get(platform)


# ------------------------------------------------------------------ Windows hard faults


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("Length", ctypes.c_ushort),
        ("MaximumLength", ctypes.c_ushort),
        ("Buffer", ctypes.c_void_p),
    ]


class _ProcessEntry(ctypes.Structure):
    """The start of SYSTEM_PROCESS_INFORMATION, as far as the fields read here. The field
    names are those of the System Informer (phnt) headers."""

    _fields_ = [
        ("NextEntryOffset", ctypes.c_uint32),
        ("NumberOfThreads", ctypes.c_uint32),
        ("WorkingSetPrivateSize", ctypes.c_int64),
        ("HardFaultCount", ctypes.c_uint32),
        ("NumberOfThreadsHighWatermark", ctypes.c_uint32),
        ("CycleTime", ctypes.c_uint64),
        ("CreateTime", ctypes.c_int64),
        ("UserTime", ctypes.c_int64),
        ("KernelTime", ctypes.c_int64),
        ("ImageName", _UnicodeString),
        ("BasePriority", ctypes.c_int32),
        ("UniqueProcessId", ctypes.c_void_p),
        ("InheritedFromUniqueProcessId", ctypes.c_void_p),
        ("HandleCount", ctypes.c_uint32),
        ("SessionId", ctypes.c_uint32),
        ("UniqueProcessKey", ctypes.c_void_p),
        ("PeakVirtualSize", ctypes.c_size_t),
        ("VirtualSize", ctypes.c_size_t),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
    ]


_SYSTEM_PROCESS_INFORMATION = 5
_LENGTH_MISMATCH = 0xC0000004


def parse_process_entries(data: bytes) -> dict[int, tuple[int, int, int]]:
    """pid to (hard faults, page faults, peak working set) from the bytes the system
    returned. Each entry gives the offset of the next; 0 ends the list."""
    out = {}
    offset = 0
    size = ctypes.sizeof(_ProcessEntry)
    while offset + size <= len(data):
        entry = _ProcessEntry.from_buffer_copy(data, offset)
        pid = entry.UniqueProcessId or 0
        out[pid] = (entry.HardFaultCount, entry.PageFaultCount, entry.PeakWorkingSetSize)
        if entry.NextEntryOffset == 0:
            break
        offset += entry.NextEntryOffset
    return out


def _query_process_information() -> bytes | None:
    query = ctypes.WinDLL("ntdll").NtQuerySystemInformation
    query.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    query.restype = ctypes.c_long
    size = 1 << 20
    for _ in range(8):  # the process list can grow between two calls
        buffer = ctypes.create_string_buffer(size)
        needed = ctypes.c_ulong(0)
        status = query(_SYSTEM_PROCESS_INFORMATION, buffer, size, ctypes.byref(needed))
        if status == 0:
            return buffer.raw[: needed.value or size]
        if status & 0xFFFFFFFF != _LENGTH_MISMATCH:
            return None
        size = max(2 * size, needed.value + (1 << 16))
    return None


def check_layout(entry: tuple[int, int, int] | None, before, after) -> bool:
    """Whether the entry of the reading process agrees with psutil: its page fault count
    and peak working set only grow, so each must lie between a psutil reading taken
    before the query and one taken after. Both fields sit after HardFaultCount, so a
    changed layout shows up here."""
    if entry is None:
        return False
    _, faults, peak = entry
    return (
        before.num_page_faults <= faults <= after.num_page_faults
        and before.peak_wset <= peak <= after.peak_wset
    )


def windows_hard_faults() -> dict[int, int] | None:
    """pid to hard fault count for every process, or None if it cannot be read or the
    layout check fails."""
    try:
        own = psutil.Process()
        before = own.memory_info()
        data = _query_process_information()
        after = own.memory_info()
    except (OSError, AttributeError, psutil.Error):
        return None
    if data is None:
        return None
    entries = parse_process_entries(data)
    if not check_layout(entries.get(own.pid), before, after):
        return None
    return {pid: values[0] for pid, values in entries.items()}


# ------------------------------------------------------------------ Linux


def parse_stat_majflt(text: str) -> int:
    """majflt, field 12 of /proc/PID/stat. The command name (field 2) may hold spaces and
    brackets, so fields are counted from its closing bracket."""
    return int(text.rpartition(")")[2].split()[9])


def parse_vm_hwm(text: str) -> int | None:
    """The peak resident set in bytes from /proc/PID/status, or None if it is not there."""
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _linux(pid: int) -> tuple[int | None, int | None]:
    try:
        peak = parse_vm_hwm(Path(f"/proc/{pid}/status").read_text(encoding="utf-8"))
        major = parse_stat_majflt(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8"))
    except (OSError, ValueError, IndexError):
        return None, None
    return peak, major


# ------------------------------------------------------------------ the tree


def process_tree(pid: int) -> list[psutil.Process]:
    root = psutil.Process(pid)
    return [root, *root.children(recursive=True)]


def process_counters(proc: psutil.Process, hard: dict[int, int] | None, platform: str) -> dict:
    """The counters of one process. hard is windows_hard_faults() on Windows."""
    info = proc.memory_info()
    row = {
        "pid": proc.pid,
        "rss_bytes": info.rss,
        "peak_rss_bytes": None,
        "page_faults": None,
        "disk_faults": None,
    }
    if platform == "win32":
        row["peak_rss_bytes"] = info.peak_wset
        row["page_faults"] = info.num_page_faults
        row["disk_faults"] = None if hard is None else hard.get(proc.pid)
    elif platform == "darwin":
        row["page_faults"] = info.pfaults
        row["disk_faults"] = info.pageins
    elif platform == "linux":
        row["peak_rss_bytes"], row["disk_faults"] = _linux(proc.pid)
    return row


def total(rows: list[dict], key: str) -> int | None:
    """The sum of key over the processes, or None if there are none or any lacks it."""
    values = [row[key] for row in rows]
    if not values or any(v is None for v in values):
        return None
    return sum(values)


def tree_snapshot(pid: int, platform: str = sys.platform) -> dict:
    """The counters of pid's tree now. Raises psutil.Error if a process cannot be read,
    for example because it exited meanwhile."""
    hard = windows_hard_faults() if platform == "win32" else None
    rows = [process_counters(proc, hard, platform) for proc in process_tree(pid)]
    keys = ("rss_bytes", "peak_rss_bytes", "page_faults", "disk_faults")
    return {"processes": rows, **{key: total(rows, key) for key in keys}}


def rss_reader(pid: int) -> Callable[[], int]:
    """A function that gives the resident set of pid's tree now, in bytes. The tree is
    found once, here, because finding it lists every process on the machine: too costly
    to repeat at every sample while the worker is being timed. A process that joins the
    tree later is not sampled; the snapshots at a window's ends show the tree changed."""
    processes = process_tree(pid)
    return lambda: sum(proc.memory_info().rss for proc in processes)


def delta(start: dict, end: dict, key: str) -> int | None:
    """How much the counter key grew between two tree snapshots, process by process. None
    if the processes are not the same ones, a value is missing, or one went down."""
    first = {row["pid"]: row.get(key) for row in start.get("processes") or []}
    last = {row["pid"]: row.get(key) for row in end.get("processes") or []}
    if not first or first.keys() != last.keys():
        return None
    out = 0
    for pid, value in first.items():
        if value is None or last[pid] is None or last[pid] < value:
            return None
        out += last[pid] - value
    return out
