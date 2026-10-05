"""Float reference of Adaptive Sensitivity Correction: the model in readable form (fixed-point and C ports follow it).

    tick(t_us, dx, dy, px, py, obj)  ->  TickOut(k, dx_scaled, dy_scaled, ...)

Pipeline per HID report: velocity estimate -> pre-reaction guard -> geometry of the nearest object -> resistance S -> K = 1/(1+S)
-> critically damped smoothing -> scaled delta. See docs/ASSIST.md for the derivation of every term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .params import AscParams
from .types import Guard, ObjectOfInterest, Reason, TickOut


def smoothstep(x: float) -> float:
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)             # smootherstep: C2, exactly 0 below 0 and exactly 1 above 1


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class State:
    k: float = 1.0
    kd: float = 0.0
    vx: float = 0.0                       # smoothed input velocity, counts/ms
    vy: float = 0.0
    last_t: Optional[int] = None
    guard: Guard = Guard.LOCKED
    on_ms: float = 0.0
    still_ms: float = 0.0
    t_move: int = 0
    t_open: int = 0
    d0: float = -1.0                      # distance to the object when the movement started (px), -1 = not latched
    zone: float = 0.0                     # braking zone radius R (px)
    obj_id: int = -1
    away_ms: float = 0.0
    pxp: float = 0.0                      # previous cursor position
    pyp: float = 0.0
    have_p: bool = False
    vpx: float = 0.0                      # smoothed cursor velocity, px/ms
    vpy: float = 0.0
    carry_x: float = 0.0
    carry_y: float = 0.0
    s: float = 0.0


class AdaptiveSensitivity:
    def __init__(self, params: Optional[AscParams] = None) -> None:
        self.p = params or AscParams.disabled()
        self.st = State()

    def set_params(self, params: AscParams) -> None:
        self.p = params

    def reset(self) -> None:
        self.st = State()

    # ------------------------------------------------------------------ the tick
    def tick(self, t_us: int, dx: int, dy: int, px: float, py: float, obj: Optional[ObjectOfInterest]) -> TickOut:
        p, c, s = self.p, self.p.cfg, self.st
        # 1. time step; a gap (or a repeated/backward timestamp) restarts the velocity estimate
        dt_us = 1000 if s.last_t is None else t_us - s.last_t
        if s.last_t is not None and not (0 < dt_us <= c.gap_us):
            s.vx = s.vy = s.vpx = s.vpy = 0.0
            s.on_ms = 0.0
            dt_us = 1000
        s.last_t = t_us
        dt_ms = dt_us / 1000.0
        # 2. velocity (counts/ms): EMA of the raw input
        a = dt_ms / (c.v_tau_ms + dt_ms)
        s.vx += a * (dx / dt_ms - s.vx)
        s.vy += a * (dy / dt_ms - s.vy)
        speed = math.hypot(s.vx, s.vy)
        if s.have_p:                                                  # the cursor's own velocity, for looking ahead
            ap = dt_ms / (c.vp_tau_ms + dt_ms)
            s.vpx += ap * ((px - s.pxp) / dt_ms - s.vpx)
            s.vpy += ap * ((py - s.pyp) / dt_ms - s.vpy)
        s.pxp, s.pyp, s.have_p = px, py, True
        reason = Reason.OK
        # 3. guard
        if p.enabled:
            s.on_ms = s.on_ms + dt_ms if speed >= p.v_on else 0.0
            s.still_ms = s.still_ms + dt_ms if speed < p.v_still else 0.0
        if not p.enabled:
            self._lock(s)
            reason = Reason.NO_PROFILE
        elif s.guard == Guard.LOCKED:
            if s.on_ms >= c.on_ms:                                    # the person has started to move
                s.guard, s.t_move, s.d0, s.obj_id = Guard.WAIT, t_us - int(s.on_ms * 1000), -1.0, -1
        elif s.guard != Guard.LOCKED and s.still_ms >= c.still_ms:
            self._lock(s)
        if p.enabled and s.guard == Guard.WAIT:
            if obj is None:
                reason = Reason.WAIT_OBJECT
            elif obj.t_appear_us is not None and s.t_move < obj.t_appear_us + p.t_lo_us:
                reason = Reason.STIMULUS_LOCK                          # moving earlier than this person can react to the new object
            else:
                s.guard, s.t_open = Guard.OPEN, t_us
        if p.enabled and s.guard == Guard.LOCKED:
            reason = Reason.LOCKED
        # 4. target resistance
        s_tgt = 0.0
        if s.guard == Guard.OPEN and obj is not None:
            s_tgt = self._resistance(t_us, dt_ms, px, py, obj, speed)
        elif s.guard == Guard.OPEN:
            reason = Reason.NO_OBJECT
        k_tgt = max(1.0 / (1.0 + s_tgt), c.k_floor)
        # 5. smoothing (a closed guard means exactly 1.0 and no filter memory)
        if s.guard != Guard.OPEN:
            s.k, s.kd, s.s = 1.0, 0.0, 0.0
            k_out = 1.0
        else:
            w = 2.0 * math.pi * (c.f_attack_hz if k_tgt < s.k else c.f_release_hz)
            dt_s = dt_ms / 1000.0
            s.kd += (w * w * (k_tgt - s.k) - 2.0 * w * s.kd) * dt_s
            s.kd = clamp(s.kd, -c.slew_per_s, c.slew_per_s)
            s.k = clamp(s.k + s.kd * dt_s, c.k_floor, 1.0)
            s.s = 1.0 / s.k - 1.0
            k_out = s.k
        # 6. apply to the raw delta (never larger, never another sign, nothing from nothing)
        ox, oy = self._apply(k_out, dx, dy)
        return TickOut(k_out, ox, oy, s.guard, reason, s.s)

    # ------------------------------------------------------------------ pieces
    def _lock(self, s: State) -> None:
        s.guard, s.d0, s.zone, s.obj_id, s.away_ms, s.on_ms = Guard.LOCKED, -1.0, 0.0, -1, 0.0, 0.0
        s.carry_x = s.carry_y = 0.0

    def _resistance(self, t_us: int, dt_ms: float, px: float, py: float, obj: ObjectOfInterest, speed: float) -> float:
        p, c, s = self.p, self.p.cfg, self.st
        dxq, dyq = obj.x - px, obj.y - py
        dist = math.hypot(dxq, dyq)
        d = max(0.0, dist - obj.radius)
        v_rad = (s.vpx * dxq + s.vpy * dyq) / dist if dist > 1e-3 else 0.0        # px/ms toward the object
        lead_ms = c.lead_base_ms + c.lead_ov_gain_ms * p.ov_med
        d_look = max(0.0, d - max(v_rad, 0.0) * lead_ms)             # where the cursor will be when the smoothing has caught up
        if s.d0 < 0.0 or obj.id != s.obj_id:                          # latch the zone for this movement / this object
            s.obj_id, s.d0 = obj.id, d
            r_min = max(2.0 * obj.radius, c.r_min_px)
            s.zone = clamp(d * p.f_b * (1.0 + c.ov_zone_gain * p.ov_rate), r_min, max(d, r_min))
            s.away_ms = 0.0
        # direction of the motion relative to the object
        cos = 0.0
        if speed > 1e-6 and dist > 1e-3:
            cos = (s.vx * dxq + s.vy * dyq) / (speed * dist)
        a_rec = smoothstep((-cos - c.c0) / (c.c1 - c.c0))
        # braking slope: the zone reaches further behind the object for people who overshoot (they will pass it)
        zone = s.zone + a_rec * max(0.0, c.back_gain * p.ov_med * s.d0 - s.zone)
        g = smoothstep(1.0 - d_look / zone)
        # a narrow deep well around the object: holding still / damping tremor
        r_deep = max(c.deep_radius_mult * obj.radius, c.r_min_px)
        g_deep = smoothstep(1.0 - d_look / r_deep)
        # coming in faster than the person can brake from here: slow harder
        v_ref = p.v_ref * math.sqrt(clamp(d_look / s.zone, 0.0, 1.0))
        e = clamp((speed - v_ref) / p.v_ref, 0.0, 1.0)
        s_hold = self._s_hold(obj.radius)
        s_pos = p.s_brake * g * (1.0 + c.lambda_speed * e) + s_hold * g_deep
        # moving away (overshoot): extra damping, released by a fast deliberate departure, by persistence and by time
        leave = smoothstep((speed - 0.7 * p.v_leave) / (0.3 * p.v_leave))
        if a_rec > 0.5:
            s.away_ms += dt_ms
        elif cos > 0.0:
            s.away_ms = 0.0
        persist = smoothstep((s.away_ms - c.away_ms) / c.away_ramp_ms)
        age_ms = (t_us - s.t_open) / 1000.0
        timeout = smoothstep((age_ms - c.max_open_ms) / c.max_open_ramp_ms)
        s_rec = c.mu_recede * (p.s_brake * g + s_hold * g_deep) * a_rec * (1.0 - leave) * (1.0 - max(persist, timeout))
        s_all = min(s_pos + s_rec, c.s_cap) * (1.0 - timeout)
        ramp = smoothstep((t_us - s.t_open) / max(p.ramp_us, 1))
        return s_all * ramp

    def _s_hold(self, radius: float) -> float:
        """Resistance inside the object: just enough to bring the person's tremor (in pixels) down to a third of the object's radius."""
        p, c = self.p, self.p.cfg
        a_px = p.tremor_counts * c.px_per_count
        if a_px <= 1e-6:
            return 0.0
        k_hold = clamp((max(radius, c.r_min_px / c.deep_radius_mult)) / (c.tremor_hold_div * a_px), c.k_floor, 1.0)
        return (1.0 / k_hold - 1.0) * p.hold_scale

    def _apply(self, k: float, dx: int, dy: int) -> tuple[int, int]:
        s = self.st
        if dx == 0 and dy == 0:
            s.carry_x = s.carry_y = 0.0                               # nothing in, nothing out; no stored motion is released later
            return 0, 0
        out = []
        for raw, name in ((dx, "carry_x"), (dy, "carry_y")):
            if raw == 0:
                out.append(0)
                continue
            v = k * raw + getattr(s, name)
            o = int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)
            if abs(o) > abs(raw) or (o != 0 and (o > 0) != (raw > 0)):
                o = raw if abs(o) > abs(raw) else 0
            setattr(s, name, v - o)
            out.append(o)
        return out[0], out[1]
