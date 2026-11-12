"""Step ordering on a hop (R3.7): the step of each (rid, attempt) must go up.

Every message a stage receives is checked when it is taken from the stream, before the
deadline, fencing and cache rules: a step at or below the highest step already seen for the
same request and attempt is a violation, counted and logged. A violation changes nothing
else: the message is handled by the usual rules, so a repeated step at a lower position
still truncates and appends (R3.6). Step numbers may skip (a step can be dropped upstream,
for example one that expired), and a new attempt starts again from any step, as a replay
does.

The highest step is kept per (rid, attempt) for the last MAX_TRACKED pairs seen, the
oldest forgotten first. It is not cleared by a Free, so a late step of a freed attempt is
still checked. A pair pushed out by MAX_TRACKED newer ones starts afresh: with at most 14
requests running at once, that only happens to requests that ended long before.
"""

import logging
import threading
from collections import OrderedDict

log = logging.getLogger("pllm.ordering")

MAX_TRACKED = 4096  # (rid, attempt) pairs whose highest step is kept


class StepOrder:
    """Counts the messages whose step does not go up for their (rid, attempt)."""

    def __init__(self, where: str, max_tracked: int = MAX_TRACKED):
        self.where = where  # names the hop in the log, for example "stage 0"
        self.max_tracked = max_tracked
        self.violations = 0
        self._highest: OrderedDict[tuple[str, int], int] = OrderedDict()
        self._lock = threading.Lock()  # streams run on several threads

    def check(self, rid: str, attempt: int, step: int) -> bool:
        """Record one arriving message; False (counted and logged) if it is out of order."""
        key = (rid, attempt)
        with self._lock:
            highest = self._highest.get(key)
            ok = highest is None or step > highest
            if ok:
                self._highest[key] = step
            if highest is not None:
                self._highest.move_to_end(key)
            while len(self._highest) > self.max_tracked:
                self._highest.popitem(last=False)
            if not ok:
                self.violations += 1
        if not ok:
            log.warning(
                "%s: step %d of %r attempt %d arrived after step %d: out of order",
                self.where,
                step,
                rid,
                attempt,
                highest,
            )
        return ok

    def __len__(self) -> int:
        with self._lock:
            return len(self._highest)
