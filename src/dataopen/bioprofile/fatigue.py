"""Session fatigue: how reaction time (T_motor) and accuracy (final error) drift over a long session.

Per-episode values are grouped into fixed-length blocks of session time (default 2 min); each block is summarized by its median
(robust to the occasional lapse). The trend over blocks is the Theil-Sen slope (median of pairwise slopes), so one bad block cannot
fake or hide a trend. The z-score divides the slope by an approximate standard error built from the robust residual spread; it says
"how many standard errors away from zero", not a calibrated p-value. A rest longer than `rest_gap_s` ends the session: fatigue recovers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Optional

from .profile import Fatigue
from .rolling import MAD_TO_SIGMA


def theil_sen(x: list[float], y: list[float]) -> tuple[float, float]:
    """(slope, intercept) of the Theil-Sen line."""
    sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
    if not sl:
        return math.nan, math.nan
    m = median(sl)
    return m, median([yy - m * xx for xx, yy in zip(x, y)])


@dataclass
class Trend:
    slope_per_h: float
    z: float
    baseline: float
    n_blocks: int


def _trend(blocks: list[tuple[float, float]], min_blocks: int, min_span_s: float) -> Optional[Trend]:
    if len(blocks) < min_blocks or blocks[-1][0] - blocks[0][0] < min_span_s:
        return None
    xs = [t / 3600.0 for t, _ in blocks]
    ys = [v for _, v in blocks]
    m, b = theil_sen(xs, ys)
    if not math.isfinite(m):
        return None
    res = [y - (m * x + b) for x, y in zip(xs, ys)]
    sig = MAD_TO_SIGMA * median([abs(r - median(res)) for r in res])
    mx = sum(xs) / len(xs)
    sxx = sum((x - mx) ** 2 for x in xs)
    sig = max(sig, 0.02 * max(abs(median(ys)), 1e-6))        # a measurement floor: a perfect line must not give an infinite z
    se = 1.2 * sig / math.sqrt(sxx) if sxx > 0 else math.inf
    base = median(ys[:2])
    return Trend(m, max(-50.0, min(50.0, m / se)) if se > 0 else 0.0, base, len(blocks))


class DriftTracker:
    def __init__(self, block_s: float = 120.0, min_per_block: int = 3, min_blocks: int = 4, min_span_s: float = 480.0,
                 max_blocks: int = 60) -> None:
        self.block_s, self.min_per_block, self.min_blocks, self.min_span_s, self.max_blocks = block_s, min_per_block, min_blocks, \
            min_span_s, max_blocks
        self.reset()

    def reset(self) -> None:
        self._t: dict[int, list[float]] = {}
        self._e: dict[int, list[float]] = {}
        self.session_s = 0.0

    def add(self, t_s: float, t_motor_ms: Optional[float], err_deg: Optional[float]) -> None:
        """t_s: seconds since the session started."""
        self.session_s = max(self.session_s, t_s)
        k = int(t_s // self.block_s)
        if t_motor_ms is not None and math.isfinite(t_motor_ms):
            self._t.setdefault(k, []).append(t_motor_ms)
        if err_deg is not None and math.isfinite(err_deg):
            self._e.setdefault(k, []).append(err_deg)
        for d in (self._t, self._e):
            while len(d) > self.max_blocks:
                del d[min(d)]

    def _blocks(self, d: dict[int, list[float]], include_open: bool) -> list[tuple[float, float]]:
        cur = int(self.session_s // self.block_s)
        out = []
        for k in sorted(d):
            v = d[k]
            if len(v) >= self.min_per_block and (k < cur or include_open):
                out.append(((k + 0.5) * self.block_s, median(v)))
        return out

    def result(self) -> Fatigue:
        tt = _trend(self._blocks(self._t, False), self.min_blocks, self.min_span_s)
        te = _trend(self._blocks(self._e, False), self.min_blocks, self.min_span_s)
        f = Fatigue(session_s=self.session_s)
        if tt is not None and te is not None:
            f.valid = True
            f.slope_t_ms_per_h, f.z_t, f.baseline_t_ms = tt.slope_per_h, tt.z, tt.baseline
            f.slope_err_deg_per_h, f.z_err = te.slope_per_h, te.z
        return f
