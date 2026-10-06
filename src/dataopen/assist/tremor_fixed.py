"""Fixed-point golden model of the tremor suppressor: the exact specification of csrc/tremor_core.c.

Filter states and energies are Q24 (int64), everything else Q16.16; `mulq24` rounds half up on the arithmetic shift; the arithmetic
helpers and the output rule (round half away from zero, clamp to the input, carry) are shared with fixed.py / asc_core.c.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .fixed import HALF, ONE, clampi, divq, mulq, q, smoothstep
from .tremor import TremorParams

HALF24 = 1 << 23


def mulq24(a: int, b: int) -> int:
    return (a * b + HALF24) >> 24


def sq24(v: int) -> int:
    """v^2 for a Q24 value, in Q24 (through Q16 so that int64 never overflows)."""
    w = v >> 8
    return ((w * w) >> 16) << 8


@dataclass(frozen=True)
class FixedTremorParams:
    enabled: int = 0
    a_lp: int = 0                        # Q24
    a_band: int = 0                      # Q24
    a_e: int = 0                         # Q24
    s_max: int = 0                       # Q16
    trim_cap: int = 1                    # counts
    v_t: int = ONE                       # Q16 counts per report
    lp_weight: int = ONE
    eps: int = 0
    r_lo: int = 0
    inv_r: int = ONE
    big_lo: int = ONE
    inv_big: int = ONE
    reset_us: int = 300_000
    reset_ms: int = 300

    @staticmethod
    def from_params(p: TremorParams) -> "FixedTremorParams":
        c = p.cfg
        q24 = lambda x: int(round(x * (1 << 24)))     # noqa: E731
        return FixedTremorParams(
            int(p.enabled), q24(p.a_lp), q24(p.a_band), q24(p.a_e), q(p.s_max), int(p.trim_cap), q(p.v_t), q(c.lp_weight), q(c.eps),
            q(c.r_lo), q(1.0 / (c.r_hi - c.r_lo)), q(c.big_lo), q(1.0 / (c.big_hi - c.big_lo)), c.reset_ms * 1000, c.reset_ms)


class FixedTremor:
    def __init__(self, params: Optional[FixedTremorParams] = None) -> None:
        self.p = params or FixedTremorParams()
        self.reset()

    def set_params(self, params: FixedTremorParams) -> None:
        self.p = params

    def reset(self) -> None:
        self.l1 = [0, 0]
        self.l2 = [0, 0]
        self.b1 = [0, 0]
        self.b2 = [0, 0]
        self.carry = [0, 0]
        self.e_band = self.e_lp = 0
        self.last_t: Optional[int] = None
        self.zero_ms = 0
        self.s = self.r = 0

    def _forget(self) -> None:
        self.l1, self.l2, self.b1, self.b2 = [0, 0], [0, 0], [0, 0], [0, 0]
        self.e_band = self.e_lp = 0
        self.zero_ms = 0
        self.s = self.r = 0

    def _step(self, dx: int, dy: int) -> list[int]:
        """One millisecond; returns the high-pass part of the input per axis (Q24)."""
        p = self.p
        hps, bsq, lsq = [], 0, 0
        for i, x in enumerate((dx, dy)):
            x24 = x << 24
            self.l1[i] += mulq24(p.a_lp, x24 - self.l1[i])
            self.l2[i] += mulq24(p.a_lp, self.l1[i] - self.l2[i])
            hp = x24 - self.l2[i]
            self.b1[i] += mulq24(p.a_band, hp - self.b1[i])
            self.b2[i] += mulq24(p.a_band, self.b1[i] - self.b2[i])
            hps.append(hp)
            bsq += sq24(self.b2[i])
            lsq += sq24(self.l2[i])
        self.e_band += mulq24(p.a_e, bsq - self.e_band)
        self.e_lp += mulq24(p.a_e, lsq - self.e_lp)
        eb, el = self.e_band >> 8, self.e_lp >> 8                       # Q16
        self.r = divq(eb, eb + mulq(p.lp_weight, el) + p.eps)
        self.s = mulq(p.s_max, smoothstep(mulq(self.r - p.r_lo, p.inv_r)))
        return hps

    def tick(self, t_us: int, dx: int, dy: int) -> tuple[int, int]:
        p = self.p
        gap = 0
        if self.last_t is not None:
            dt = t_us - self.last_t
            if dt > p.reset_us:
                self._forget()
            elif dt > 1500:
                gap = min((dt + 500) // 1000 - 1, p.reset_ms)
        self.last_t = t_us
        if not p.enabled:
            return dx, dy
        for _ in range(gap):
            self._step(0, 0)
        self.zero_ms += gap
        if dx == 0 and dy == 0:
            self.zero_ms += 1
            self._step(0, 0)
            if self.zero_ms >= p.reset_ms:
                self._forget()
            self.carry = [0, 0]
            return 0, 0
        self.zero_ms = 0
        hp = self._step(dx, dy)
        big = divq(max(abs(dx), abs(dy)) << 16, p.v_t)
        s_eff = mulq(self.s, ONE - smoothstep(mulq(big - p.big_lo, p.inv_big)))
        out = []
        for i, x in enumerate((dx, dy)):
            if x == 0:
                out.append(0)
                continue
            xq = x << 16
            y = xq - mulq(s_eff, hp[i] >> 8)
            lo, hi = (0, xq) if x > 0 else (xq, 0)
            y = clampi(y, lo, hi)
            y = clampi(y, xq - (p.trim_cap << 16), xq + (p.trim_cap << 16))
            y = clampi(y, lo, hi)
            v = y + self.carry[i]
            o = ((abs(v) + HALF) >> 16) * (1 if v >= 0 else -1)
            if abs(o) > abs(x):
                o = x
            elif o != 0 and (o > 0) != (x > 0):
                o = 0
            self.carry[i] = v - (o << 16)
            out.append(o)
        return out[0], out[1]
