"""Types of the ASC module. Geometry is in pixels, time in microseconds, raw input in HID counts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Protocol


@dataclass(frozen=True)
class ObjectOfInterest:
    """The nearest object, as supplied by an external adapter (accessibility API, UI detector, app markup, ...)."""
    id: int
    x: float
    y: float
    radius: float = 0.0
    t_appear_us: Optional[int] = None    # when the object became visible/new; None for static objects (no stimulus to react to)


class ObjectProvider(Protocol):
    """What an adapter looks like (not part of the core): called by the integration, never by the module."""

    def nearest(self, x: float, y: float) -> Optional[ObjectOfInterest]: ...


class Guard(IntEnum):
    LOCKED = 0      # the person is still: K is exactly 1.0
    WAIT = 1        # moving, but the pre-reaction guard has not opened (no object / no profile / movement not a reaction to it)
    OPEN = 2        # assistance allowed


class Reason(IntEnum):
    OK = 0
    NO_PROFILE = 1
    NO_OBJECT = 2
    LOCKED = 3
    STIMULUS_LOCK = 4        # the movement began sooner after the object appeared than this person can react
    WAIT_OBJECT = 5


@dataclass(frozen=True)
class TickOut:
    k: float                 # effective-sensitivity coefficient, 0.1 .. 1.0
    dx: int                  # raw delta scaled by k (integer counts, |dx| <= |raw dx|, same sign)
    dy: int
    guard: Guard
    reason: Reason
    s: float = 0.0           # resistance 1/k - 1 (diagnostics)
