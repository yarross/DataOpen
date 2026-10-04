"""Plain data types shared by the engine, the adapter and the simulator. Angles are degrees, times microseconds (monotonic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class TargetObs:
    """The aim target as seen by the detector in one frame, as angles from the crosshair (x right, y down)."""
    t_us: int                    # capture time of the frame
    x_deg: float
    y_deg: float
    vis: float = 1.0             # 0..1 how visible the target is (aim-region contrast / keypoint confidence)
    degraded: bool = False       # smoke / heavy noise / low contrast reported by the caller
    radius_deg: float = 0.5      # angular radius of the aim region (hit tolerance)


class State(str, Enum):
    IDLE = "idle"
    ARMED = "armed"              # a target appeared while the hand was still: waiting for the first movement
    MOVING = "moving"            # first (ballistic) movement toward the target
    SETTLE = "settle"            # movement paused: a corrective movement may follow
    TRACK = "track"              # target close: following it


class Scenario(str, Enum):
    WIDE_FLICK = "wide_flick"
    MICRO_TRACK = "micro_track"
    SURPRISE = "surprise"        # modifier: the target appeared while the hand was still
    LOW_VIS = "low_vis"          # modifier: low contrast / smoke / noise at appearance


@dataclass
class Episode:
    """One completed episode with everything measured about it (logged, and folded into the rolling statistics)."""
    kind: str                                   # 'flick' | 'track' | 'miss' | 'censored'
    t_appear_us: int = 0
    surprise: bool = False
    lowvis: bool = False
    d0_deg: float = 0.0
    t_motor_ms: Optional[float] = None
    anticipation: bool = False
    v_max: Optional[float] = None
    d_brake_frac: Optional[float] = None
    overshoot: Optional[float] = None
    err_final_deg: Optional[float] = None
    n_corrections: int = 0
    heading_dev_deg: Optional[float] = None     # angle between the first 10% of the movement and the direction to the target
    undershoot: bool = False
    main_known: bool = False                    # the ballistic phase was measured (a wide flick)
    phase_lag_ms: Optional[float] = None
    jitter_hz: Optional[float] = None
    jitter_amp_deg: Optional[float] = None
    scenarios: list[str] = field(default_factory=list)
