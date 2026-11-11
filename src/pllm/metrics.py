"""Gateway metrics: one JSON-lines file per run and serialized bytes per hop (R7.1, R7.2).

Only the gateway writes the file and only the gateway's clock stamps it, so no time from
another machine is ever subtracted. Every *_ns value comes from one Recorder.clock.

The Stage 0 to Stage 1 hop never passes the gateway, so Stage 0 counts it per Steps stream
and returns the counters on that stream's gRPC trailing metadata. send_counters and
parse_counters are the only code that knows this transport. The parser takes integers and
known labels only, so a stage cannot put a timestamp into the log, and a stream with no
valid report is counted as unreported instead of being read as zero.
"""

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from pllm.proto import protocol_pb2

log = logging.getLogger("pllm.metrics")

CLOCK = "perf_counter_ns"
TRAILER_KEY = "pllm-hop-bytes"
GATEWAY_STAGE0 = "gateway_stage0"
STAGE1_GATEWAY = "stage1_gateway"
STAGE0_STAGE1 = "stage0_stage1"
ACTIVATIONS = "StepRequest.activations"
RESULT = "StepResult"
PHASES = ("prefill", "decode", "replay")


def phase_of(message) -> str:
    """By step, not size: the prefill of a one-token prompt is decode-sized."""
    if getattr(message, "replay", False):
        return "replay"
    return "prefill" if message.step == 0 else "decode"


def measure(message) -> tuple[str, str, int, int, int]:
    """Message type, phase, serialized size, tensor payload size and positions carried."""
    if isinstance(message, protocol_pb2.StepResult):
        return RESULT, phase_of(message), message.ByteSize(), 0, 0
    kind = f"StepRequest.{message.WhichOneof('payload')}"
    return (
        kind,
        phase_of(message),
        message.ByteSize(),
        len(message.activations.data),
        message.length,
    )


class Counters:
    """Messages, bytes, payload bytes and positions per message type and phase."""

    def __init__(self):
        self.buckets: dict[tuple[str, str], list[int]] = {}

    def add(self, msg: str, phase: str, *values: int) -> None:
        bucket = self.buckets.setdefault((msg, phase), [0, 0, 0, 0])
        for i, value in enumerate(values):
            bucket[i] += value

    def count(self, message) -> None:
        msg, phase, *sizes = measure(message)
        self.add(msg, phase, 1, *sizes)

    def rows(self) -> list[list]:
        return [[msg, phase, *values] for (msg, phase), values in self.buckets.items()]


def send_counters(context, counters: Counters) -> None:
    """Stage side: attach this stream's counters to its end."""
    context.set_trailing_metadata(((TRAILER_KEY, json.dumps(counters.rows())),))


def parse_counters(metadata) -> list[list] | None:
    """Gateway side: the rows of a stream's report, or None if it has no valid one."""
    values = [value for key, value in metadata or () if key == TRAILER_KEY]
    if len(values) != 1:
        return None
    try:
        rows = json.loads(values[0])
    except (TypeError, ValueError):
        return None
    if not isinstance(rows, list) or len(rows) > len(PHASES):
        return None
    seen = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != 6:
            return None
        msg, phase, *numbers = row
        if msg != ACTIVATIONS or phase not in PHASES or phase in seen:
            return None
        if any(type(number) is not int or number < 0 for number in numbers):
            return None
        seen.add(phase)
    return rows


class Recorder:
    """The metrics file of one run. With no path it writes and keeps nothing.

    A write error never fails a request: it is logged once and counted. finish() closes
    the run, and anything recorded after it is ignored.
    """

    def __init__(self, path: Path | str | None = None, info: dict | None = None, clock=None):
        self.clock = clock or time.perf_counter_ns
        self.write_errors = 0
        self.links = 0
        self.links_unreported = 0
        self._counts = {
            GATEWAY_STAGE0: Counters(),
            STAGE1_GATEWAY: Counters(),
            STAGE0_STAGE1: Counters(),
        }
        self._lock = threading.Lock()
        self._file = None
        if path is not None:
            # "x": a second run must not write into an earlier run's file
            self._file = open(path, "x", encoding="utf-8", buffering=1, newline="\n")
            with self._lock:
                self._write(
                    {
                        "type": "run",
                        "clock": CLOCK,
                        "start_ns": self.clock(),
                        "wall_time": datetime.now(UTC).isoformat(),  # a label, never subtracted
                        **(info or {}),
                    }
                )

    def now(self) -> int:
        return self.clock()

    def _write(self, line: dict) -> None:
        """Called with the lock held."""
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(line, separators=(",", ":")) + "\n")
        except OSError as err:
            if not self.write_errors:
                log.error("cannot write the metrics file: %s", err)
            self.write_errors += 1

    def hop(self, hop: str, message, accepted: bool = True) -> None:
        """One message the gateway sends or receives. A dropped result crossed the wire,
        so it is counted too, with accepted false."""
        if self._file is None:
            return
        msg, phase, size, payload, positions = measure(message)
        with self._lock:
            self._counts[hop].add(msg, phase, 1, size, payload, positions)
            self._write(
                {
                    "type": "hop",
                    "t_ns": self.clock(),
                    "hop": hop,
                    "msg": msg,
                    "phase": phase,
                    "rid": message.rid,
                    "attempt": message.attempt,
                    "step": message.step,
                    "bytes": size,
                    "payload_bytes": payload,
                    "positions": positions,
                    "accepted": accepted,
                }
            )

    def request(self, outcome: str, code, **fields) -> None:
        """A request's end. code is the gRPC status the gateway set, or None if it set none."""
        with self._lock:
            self._write(
                {
                    "type": "request",
                    "t_ns": self.clock(),
                    "outcome": outcome,
                    "code": code.name if code else None,
                    **fields,
                }
            )

    def event(self, record: dict) -> None:
        """A failure detection record, written as a line of its own type (detected or
        registered). Its times must come from this recorder's clock."""
        with self._lock:
            line = {"type": record["event"]}
            line.update((k, v) for k, v in record.items() if k != "event")
            self._write(line)

    def link_ended(self, rows: list[list] | None) -> None:
        """Add one Stage 0 stream's report; None means it gave no valid report."""
        if self._file is None:
            return
        with self._lock:
            self.links += 1
            if rows is None:
                self.links_unreported += 1
                log.warning("a stage 0 stream ended with no valid byte report")
                return
            for msg, phase, *values in rows:
                self._counts[STAGE0_STAGE1].add(msg, phase, *values)

    def finish(self) -> None:
        """Write the byte totals and the end of the run. Safe to call twice."""
        with self._lock:
            if self._file is None:
                return
            for hop, counters in self._counts.items():
                source = "stage0" if hop == STAGE0_STAGE1 else "gateway"
                for (msg, phase), (messages, size, payload, positions) in counters.buckets.items():
                    self._write(
                        {
                            "type": "bytes",
                            "hop": hop,
                            "msg": msg,
                            "phase": phase,
                            "messages": messages,
                            "bytes": size,
                            "payload_bytes": payload,
                            "positions": positions,
                            "source": source,
                        }
                    )
            self._write(
                {
                    "type": "run_end",
                    "t_ns": self.clock(),
                    "links": self.links,
                    "links_unreported": self.links_unreported,
                    "stage0_stage1_complete": self.links_unreported == 0,
                    "write_errors": self.write_errors,
                }
            )
            file, self._file = self._file, None
            try:
                file.close()
            except OSError as err:
                self.write_errors += 1
                log.error("cannot close the metrics file: %s", err)
