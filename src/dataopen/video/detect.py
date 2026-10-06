"""Which input is the GPU on? A small state machine over electrical facts, with hysteresis, so a flaky cable does not flap the capture path.

Inputs per interface: `power5v` (the +5 V the source supplies on HDMI / DP_PWR), `link` (HDMI: TMDS clock locked; DP: link training done and
valid video). The device has an HDMI input and a DP input; a GPU normally drives one. Outputs: the active interface (or None) and why."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

LOCK_US = 150_000  # a link must stay up this long before it is believed
LOSS_US = 400_000  # and stay down this long before it is given up (a mode switch takes up to a few hundred ms)


@dataclass
class Link:
    power5v: bool = False
    link: bool = False


class InputDetector:
    def __init__(self) -> None:
        self.active: Optional[str] = None
        self.reason = "no signal"
        self._up_since: dict[str, Optional[int]] = {"hdmi": None, "dp": None}
        self._down_since: Optional[int] = None
        self.conflict = False

    def update(self, t_us: int, hdmi: Link, dp: Link) -> Optional[str]:
        live = {"hdmi": hdmi.power5v and hdmi.link, "dp": dp.power5v and dp.link}
        for k, v in live.items():
            if v:
                if self._up_since[k] is None:
                    self._up_since[k] = t_us
            else:
                self._up_since[k] = None
        stable = [k for k, since in self._up_since.items() if since is not None and t_us - since >= LOCK_US]
        self.conflict = len(stable) == 2
        if self.active is not None:
            if live[self.active]:
                self._down_since = None
                self.reason = f"{self.active} locked" + (" (the other input is also live: kept)" if self.conflict else "")
                return self.active
            if self._down_since is None:
                self._down_since = t_us
            if t_us - self._down_since < LOSS_US:
                self.reason = f"{self.active} lost, waiting"
                return self.active
            self.active, self._down_since = None, None
        if stable:
            self.active = stable[0]
            self.reason = f"{self.active} locked"
            self._down_since = None
        else:
            self.reason = "no signal" if not any(live.values()) else "locking"
        return self.active
