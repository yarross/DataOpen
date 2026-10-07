"""The device's front panel as a model: two buttons and four LEDs (docs/SLOTS.md section 4). SIMULATION: on the board these are GPIO pins;
the logic (what a press means, how long a hold must be, what the LEDs say) is what is fixed here so that the firmware can be built to it.

  SLOT button     a short press goes to the next slot.
  CONFIRM button  a short press is 'the person is here': it keeps a running trial, or else opens the 30 s window in which the actions that
                  need the device's own button (erase, forget phones, export for another device, accept from a new sender, apply an update)
                  are allowed. Held 10 s and released: erase all personal data. Held 20 s and released: factory reset (new identity).
                  Released earlier than 10 s but not a short press: nothing happens (that is how a hold is cancelled).
  LEDs            one per slot. Steady: the active slot. Blinking: a trial is running. All breathing slowly: the confirm window is open.
                  At 10 s of holding CONFIRM all blink fast ('release now to erase'); at 20 s they flicker ('release now for a
                  factory reset').

Debounce: a press shorter than 30 ms is a contact bounce, not a press. Not every constant here has been tried on a person with tremor;
the times (10 s / 20 s, 2 s for 'short') are starting points to be tuned with users, not findings.
"""
from __future__ import annotations

from typing import Optional

DEBOUNCE_US = 30_000
SHORT_MAX_US = 2_000_000
ERASE_US = 10_000_000
FACTORY_US = 20_000_000

BUTTONS = ("slot", "confirm")
# what a release means
SLOT_NEXT, CONFIRM_SHORT, ERASE, FACTORY = "slot_next", "confirm_short", "erase", "factory"
# LED modes, highest priority first
M_ERROR, M_FACTORY, M_ERASE, M_TRIAL, M_WINDOW, M_CALIB, M_STEADY = "error", "factory", "erase", "trial", "window", "calibrating", "steady"


class Panel:
    def __init__(self) -> None:
        self.down: dict[str, Optional[int]] = {b: None for b in BUTTONS}

    def press(self, button: str, now: int) -> None:
        if button not in self.down:
            raise ValueError(f"no such button: {button}")
        if self.down[button] is None:
            self.down[button] = now

    def release(self, button: str, now: int) -> Optional[str]:
        t0 = self.down.get(button)
        if t0 is None:
            return None
        self.down[button] = None
        held = now - t0
        if held < DEBOUNCE_US:
            return None
        if button == "slot":
            return SLOT_NEXT if held <= SHORT_MAX_US else None
        if held >= FACTORY_US:
            return FACTORY
        if held >= ERASE_US:
            return ERASE
        return CONFIRM_SHORT if held <= SHORT_MAX_US else None

    def held_us(self, button: str, now: int) -> int:
        t0 = self.down.get(button)
        return 0 if t0 is None else max(0, now - t0)

    def hold_warning(self, now: int) -> str:
        h = self.held_us("confirm", now)
        return M_FACTORY if h >= FACTORY_US else M_ERASE if h >= ERASE_US else ""


def leds(mode: str, active: int, now: int, count: int = 4) -> tuple[bool, ...]:
    """Which LEDs are lit at `now` (microseconds) in `mode`."""
    ms = now // 1000
    if mode == M_ERROR:
        return (True,) * count if (ms // 100) % 2 == 0 else (False,) * count
    if mode == M_FACTORY:
        return (True,) * count if (ms // 40) % 2 == 0 else (False,) * count
    if mode == M_ERASE:
        return (True,) * count if (ms // 200) % 2 == 0 else (False,) * count
    if mode == M_WINDOW:
        return (True,) * count if (ms // 500) % 2 == 0 else (False,) * count
    on = (ms // 250) % 2 == 0 if mode == M_TRIAL else (ms // 1000) % 2 == 0 if mode == M_CALIB else True
    return tuple(on and i == active for i in range(count))
