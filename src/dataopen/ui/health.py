"""When the scene path is too slow, the help stays off: predictably, not "sometimes" (docs/LATENCY.md).

A pure state machine (no clock, no sleeping: the caller says what it saw and when), so that it is tested in virtual time.

    warming -- `recover_n` fresh scenes in a row ---------------> ok
    ok      -- `late_trip` late scenes in a row, or a slow detector --> off     (one empty scene is sent: the help ends now, not at the TTL)
    off     -- `hold_ms` passed AND `recover_n` fresh scenes in a row --> ok

"Late" = the scene was older than `publish_max_age_ms` when it was ready, or the frame was dropped for its age. "Fresh" = not older than
the stricter `recover_max_age_ms`. The two thresholds are the hysteresis: a scene between them neither switches the help off nor back on, so
an age that wanders around one threshold can not make the help blink. A single late scene is simply not sent.

While the state is not `ok`, nothing is sent; the bridge lets its last scene expire (or has just been told there is none), K goes back to 1
at the core's own slew rate, and the person has the unassisted pointer plus the tremor filter, which does not depend on the scene."""

from __future__ import annotations

from collections import deque
from typing import Optional

from .policy import V1, ScenePolicy

STATES = ("none", "warming", "ok", "off")             # `none`: no detector / not started (the state the phone shows before anything runs)
REASONS = ("", "late", "slow_detector")


class SceneHealth:
    def __init__(self, policy: ScenePolicy = V1) -> None:
        self.p = policy
        self.state = "warming"
        self.reason = ""
        self.late = 0              # late scenes in a row
        self.fresh = 0             # fresh scenes in a row
        self.off_since_ms: Optional[float] = None
        self.trips = 0
        self.transitions = 0
        self._detect: deque[float] = deque(maxlen=policy.infer_window)
        self.skipped = 0           # scenes not sent because they were late
        self._announce_off = False

    # ---- what the caller feeds in
    def detect(self, ms: float) -> None:
        """One inference time, ms."""
        self._detect.append(ms)

    def stale_frame(self, now_ms: float) -> None:
        """A frame was dropped for its age before it was looked at: the pipeline is behind. Counts as a late scene."""
        self._late(now_ms, "late")

    def scene(self, age_ms: float, now_ms: float) -> bool:
        """A scene is ready (`age_ms`: its age now). Returns True when it may be SENT."""
        slow = self.infer_p95() > self.p.infer_budget_ms
        if slow and self.state != "off":
            self._trip(now_ms, "slow_detector")
            return False
        if age_ms > self.p.publish_max_age_ms:
            self.fresh = 0
            self._late(now_ms, "late")
            self.skipped += 1
            return False
        if age_ms <= self.p.recover_max_age_ms and not slow:
            self.late = 0
            self.fresh += 1
        else:
            self.fresh = 0                  # between the thresholds: usable, but not a reason to switch back on (and not one to switch off)
            self.late = 0
        if self.state == "warming" and self.fresh >= self.p.recover_n:
            self._set("ok", "")
        elif self.state == "off" and self.fresh >= self.p.recover_n and now_ms - (self.off_since_ms or 0.0) >= self.p.hold_ms:
            self._set("ok", "")
        return self.state == "ok"

    def take_announcement(self) -> bool:
        """True once after the switch-off: the publisher then sends ONE empty scene so the bridge ends the help at once."""
        a, self._announce_off = self._announce_off, False
        return a

    # ---- derived
    def infer_p95(self) -> float:
        if len(self._detect) < min(10, self.p.infer_window):
            return 0.0                      # too few to say; one slow start is not a verdict
        s = sorted(self._detect)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    def _late(self, now_ms: float, why: str) -> None:
        self.late += 1
        self.fresh = 0
        if self.state == "ok" and self.late >= self.p.late_trip:
            self._trip(now_ms, why)
        elif self.state == "warming" and self.late >= self.p.late_trip:
            self._set("off", why)
            self.off_since_ms, self.trips = now_ms, self.trips + 1

    def _trip(self, now_ms: float, why: str) -> None:
        self._set("off", why)
        self.off_since_ms = now_ms
        self.trips += 1
        self.late = self.fresh = 0
        self._announce_off = True

    def _set(self, state: str, why: str) -> None:
        if state != self.state:
            self.transitions += 1
        self.state, self.reason = state, why
        self.late = self.fresh = 0
