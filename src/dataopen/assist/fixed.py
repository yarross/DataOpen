"""Fixed-point (Q16.16) golden model of ASC: pure integer arithmetic, the exact specification of the C implementation (csrc/asc_core.c).

Conventions (identical in C): values are Q16.16 in int32 range with int64 intermediates; `mulq` rounds half up on the arithmetic shift;
divisions truncate toward zero (`sdiv`), square roots are floor integer square roots; times are integer microseconds. Nothing here uses
floating point except converting the inputs (cursor/object coordinates) and the parameters at the boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .params import AscParams
from .types import Guard, ObjectOfInterest, Reason, TickOut

ONE = 1 << 16
HALF = ONE >> 1
COORD_MAX = (1 << 30) - 1                                            # coordinate differences are clamped to +-16384 px (Q16)


def q(x: float) -> int:
    return int(round(x * ONE))


def mulq(a: int, b: int) -> int:
    return (a * b + HALF) >> 16


def sdiv(a: int, b: int) -> int:
    """a / b truncated toward zero (C semantics); b != 0."""
    n = abs(a) // abs(b)
    return n if (a >= 0) == (b >= 0) else -n


def divq(a: int, b: int) -> int:
    """(a / b) in Q16 for Q16 a, b: truncated toward zero."""
    return sdiv(a << 16, b)


def isqrt(n: int) -> int:
    return math.isqrt(n) if n > 0 else 0


def clampi(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else (hi if x > hi else x)


def smoothstep(x: int) -> int:
    x = clampi(x, 0, ONE)
    x3 = mulq(mulq(x, x), x)
    inner = mulq(x, 6 * x - 15 * ONE) + 10 * ONE
    return mulq(x3, inner)


def scale_us(x: int, dt_us: int) -> int:
    """x * dt_us / 1e6, rounded half away from zero."""
    n = x * dt_us
    return (n + 500_000) // 1_000_000 if n >= 0 else -((-n + 500_000) // 1_000_000)


@dataclass(frozen=True)
class FixedParams:
    enabled: int = 0
    v_on: int = 0
    v_still: int = 0
    on_us: int = 6000
    still_us: int = 150_000
    t_lo_us: int = 0
    ramp_us: int = 20_000
    f_b: int = 0
    ov_rate: int = 0
    ov_med: int = 0
    s_brake: int = 0
    tremor_px: int = 0                   # tremor amplitude in px (counts * pointer gain)
    hold_scale: int = ONE
    v_ref: int = ONE
    v_leave: int = ONE
    # constants of the model
    ov_zone_gain: int = 0
    r_min_px: int = 0
    deep_mult: int = 0
    back_gain: int = 0
    hold_div: int = 0
    lam: int = 0
    mu: int = 0
    c0: int = 0
    inv_c: int = ONE
    k_floor: int = 0
    s_cap: int = 0
    lead_base: int = 0
    lead_gain: int = 0
    away_us: int = 0
    away_ramp_us: int = 1
    open_us: int = 0
    open_ramp_us: int = 1
    w_att: int = 0
    w_rel: int = 0
    slew: int = 0
    v_tau: int = 0
    vp_tau: int = 0
    gap_us: int = 20_000

    @staticmethod
    def from_params(p: AscParams) -> "FixedParams":
        c = p.cfg
        tp = 2.0 * math.pi
        return FixedParams(
            int(p.enabled), q(p.v_on), q(p.v_still), int(c.on_ms * 1000), int(c.still_ms * 1000), p.t_lo_us, max(p.ramp_us, 1),
            q(p.f_b), q(p.ov_rate), q(p.ov_med), q(p.s_brake), q(p.tremor_counts * c.px_per_count), q(p.hold_scale), q(p.v_ref),
            q(p.v_leave), q(c.ov_zone_gain), q(c.r_min_px), q(c.deep_radius_mult), q(c.back_gain), q(c.tremor_hold_div),
            q(c.lambda_speed), q(c.mu_recede), q(c.c0), q(1.0 / (c.c1 - c.c0)), q(c.k_floor), q(c.s_cap),
            q(c.lead_base_ms), q(c.lead_ov_gain_ms), int(c.away_ms * 1000), max(int(c.away_ramp_ms * 1000), 1),
            int(c.max_open_ms * 1000), max(int(c.max_open_ramp_ms * 1000), 1), q(tp * c.f_attack_hz), q(tp * c.f_release_hz),
            q(c.slew_per_s), q(c.v_tau_ms), q(c.vp_tau_ms), c.gap_us)


class FixedAsc:
    """Same interface as `AdaptiveSensitivity`, integer arithmetic."""

    def __init__(self, params: Optional[FixedParams] = None) -> None:
        self.p = params or FixedParams()
        self.reset()

    def set_params(self, params: FixedParams) -> None:
        self.p = params

    def reset(self) -> None:
        self.k, self.kd = ONE, 0
        self.vx = self.vy = self.vpx = self.vpy = 0
        self.last_t: Optional[int] = None
        self.guard = Guard.LOCKED
        self.on_us = self.still_us = 0
        self.t_move = self.t_open = 0
        self.d0, self.zone, self.obj_id = -1, 0, -1
        self.away_us = 0
        self.pxp = self.pyp = 0
        self.have_p = False
        self.carry_x = self.carry_y = 0
        self.s = 0

    # ------------------------------------------------------------------
    def tick(self, t_us: int, dx: int, dy: int, px: float, py: float, obj: Optional[ObjectOfInterest]) -> TickOut:
        p = self.p
        pxq, pyq = q(px), q(py)
        # 1. time
        dt_us = 1000 if self.last_t is None else t_us - self.last_t
        if self.last_t is not None and not (0 < dt_us <= p.gap_us):
            self.vx = self.vy = self.vpx = self.vpy = 0
            self.on_us = 0
            dt_us = 1000
        self.last_t = t_us
        dt_q = (dt_us << 16) // 1000                                  # dt in ms, Q16
        # 2. velocities (counts/ms and px/ms)
        a_v = divq(dt_q, p.v_tau + dt_q)
        self.vx += mulq(a_v, sdiv(dx << 32, dt_q) - self.vx)
        self.vy += mulq(a_v, sdiv(dy << 32, dt_q) - self.vy)
        speed = isqrt(self.vx * self.vx + self.vy * self.vy)
        if self.have_p:
            a_p = divq(dt_q, p.vp_tau + dt_q)
            self.vpx += mulq(a_p, sdiv((pxq - self.pxp) << 16, dt_q) - self.vpx)
            self.vpy += mulq(a_p, sdiv((pyq - self.pyp) << 16, dt_q) - self.vpy)
        self.pxp, self.pyp, self.have_p = pxq, pyq, True
        reason = Reason.OK
        # 3. guard
        if p.enabled:
            self.on_us = self.on_us + dt_us if speed >= p.v_on else 0
            self.still_us = self.still_us + dt_us if speed < p.v_still else 0
        if not p.enabled:
            self._lock()
            reason = Reason.NO_PROFILE
        elif self.guard == Guard.LOCKED:
            if self.on_us >= p.on_us:
                self.guard, self.t_move, self.d0, self.obj_id = Guard.WAIT, t_us - self.on_us, -1, -1
        elif self.still_us >= p.still_us:
            self._lock()
        if p.enabled and self.guard == Guard.WAIT:
            if obj is None:
                reason = Reason.WAIT_OBJECT
            elif obj.t_appear_us is not None and self.t_move < obj.t_appear_us + p.t_lo_us:
                reason = Reason.STIMULUS_LOCK
            else:
                self.guard, self.t_open = Guard.OPEN, t_us
        if p.enabled and self.guard == Guard.LOCKED:
            reason = Reason.LOCKED
        # 4. resistance
        s_tgt = 0
        if self.guard == Guard.OPEN and obj is not None:
            s_tgt = self._resistance(t_us, dt_us, pxq, pyq, obj, speed)
        elif self.guard == Guard.OPEN:
            reason = Reason.NO_OBJECT
        k_tgt = max(divq(ONE, ONE + s_tgt), p.k_floor)
        # 5. smoothing
        if self.guard != Guard.OPEN:
            self.k, self.kd, self.s = ONE, 0, 0
            k_out = ONE
        else:
            w = p.w_att if k_tgt < self.k else p.w_rel
            term = mulq(mulq(w, w), k_tgt - self.k) - mulq(2 * w, self.kd)
            self.kd = clampi(self.kd + scale_us(term, dt_us), -p.slew, p.slew)
            self.k = clampi(self.k + scale_us(self.kd, dt_us), p.k_floor, ONE)
            self.s = divq(ONE, self.k) - ONE
            k_out = self.k
        ox, oy = self._apply(k_out, dx, dy)
        return TickOut(k_out / ONE, ox, oy, self.guard, reason, self.s / ONE)

    # ------------------------------------------------------------------
    def _lock(self) -> None:
        self.guard, self.d0, self.zone, self.obj_id, self.away_us, self.on_us = Guard.LOCKED, -1, 0, -1, 0, 0
        self.carry_x = self.carry_y = 0

    def _s_hold(self, radius: int) -> int:
        p = self.p
        if p.tremor_px < 66:                                          # < 0.001 px
            return 0
        r_floor = divq(p.r_min_px, p.deep_mult)
        k_hold = clampi(divq(max(radius, r_floor), mulq(p.hold_div, p.tremor_px)), p.k_floor, ONE)
        return mulq(divq(ONE, k_hold) - ONE, p.hold_scale)

    def _resistance(self, t_us: int, dt_us: int, pxq: int, pyq: int, obj: ObjectOfInterest, speed: int) -> int:
        p = self.p
        rq = q(obj.radius)
        dxq = clampi(q(obj.x) - pxq, -COORD_MAX, COORD_MAX)
        dyq = clampi(q(obj.y) - pyq, -COORD_MAX, COORD_MAX)
        dist = isqrt(dxq * dxq + dyq * dyq)
        d = max(0, dist - rq)
        v_rad = sdiv(self.vpx * dxq + self.vpy * dyq, dist) if dist > 66 else 0
        lead = p.lead_base + mulq(p.lead_gain, p.ov_med)
        d_look = max(0, d - mulq(max(v_rad, 0), lead))
        if self.d0 < 0 or obj.id != self.obj_id:
            self.obj_id, self.d0 = obj.id, d
            r_min = max(2 * rq, p.r_min_px)
            raw_zone = mulq(mulq(d, p.f_b), ONE + mulq(p.ov_zone_gain, p.ov_rate))
            self.zone = clampi(raw_zone, r_min, max(d, r_min))
            self.away_us = 0
        cos = 0
        if speed > 66 and dist > 66:
            cos = clampi(sdiv(self.vx * dxq + self.vy * dyq, mulq(speed, dist)), -ONE, ONE)
        a_rec = smoothstep(mulq(-cos - p.c0, p.inv_c))
        zone = self.zone + mulq(a_rec, max(0, mulq(p.back_gain, mulq(p.ov_med, self.d0)) - self.zone))
        g = smoothstep(ONE - divq(d_look, zone))
        r_deep = max(mulq(p.deep_mult, rq), p.r_min_px)
        g_deep = smoothstep(ONE - divq(d_look, r_deep))
        frac = clampi(divq(d_look, self.zone), 0, ONE)
        v_ref = mulq(p.v_ref, isqrt(frac << 16))
        e = clampi(divq(max(0, speed - v_ref), p.v_ref), 0, ONE)
        s_hold = self._s_hold(rq)
        s_brake_g = mulq(p.s_brake, g)
        s_hold_g = mulq(s_hold, g_deep)
        s_pos = mulq(s_brake_g, ONE + mulq(p.lam, e)) + s_hold_g
        leave = smoothstep(divq(max(0, speed - mulq(45875, p.v_leave)), mulq(19661, p.v_leave)))
        if a_rec > HALF:
            self.away_us += dt_us
        elif cos > 0:
            self.away_us = 0
        persist = smoothstep(((self.away_us - p.away_us) << 16) // p.away_ramp_us)
        age = t_us - self.t_open
        timeout = smoothstep(((age - p.open_us) << 16) // p.open_ramp_us)
        relief = max(persist, timeout)
        s_rec = mulq(mulq(mulq(p.mu, s_brake_g + s_hold_g), a_rec), mulq(ONE - leave, ONE - relief))
        s_all = mulq(min(s_pos + s_rec, p.s_cap), ONE - timeout)
        ramp = smoothstep((age << 16) // p.ramp_us)
        return mulq(s_all, ramp)

    def _apply(self, k: int, dx: int, dy: int) -> tuple[int, int]:
        if dx == 0 and dy == 0:
            self.carry_x = self.carry_y = 0
            return 0, 0
        out = []
        for raw, which in ((dx, 0), (dy, 1)):
            if raw == 0:
                out.append(0)
                continue
            carry = self.carry_x if which == 0 else self.carry_y
            v = k * raw + carry
            o = ((abs(v) + HALF) >> 16) * (1 if v >= 0 else -1)
            if abs(o) > abs(raw):
                o = raw
            elif o != 0 and (o > 0) != (raw > 0):
                o = 0
            if which == 0:
                self.carry_x = v - (o << 16)
            else:
                self.carry_y = v - (o << 16)
            out.append(o)
        return out[0], out[1]
