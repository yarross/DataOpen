"""Rolling-window robust statistics: exact median and sigma (1.4826 * MAD) over the last N values.

Why median + MAD and not mean + std: reaction and movement metrics are heavy-tailed (a lapse of 900 ms among 220 ms reactions ruins a
mean and doubles a std), and the profile must describe the player's *typical* behaviour. Values far outside the window are winsorized
(clipped, counted) rather than dropped, so a genuine shift of the distribution is still followed after a few samples.
"""
from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque
from statistics import NormalDist
from typing import Optional

MAD_TO_SIGMA = 1.4826


class RollingMedianSigma:
    def __init__(self, window: int = 32, min_n: int = 8, clip_sigma: float = 6.0, sigma_floor: float = 0.0) -> None:
        if window < 3 or min_n < 3 or min_n > window:
            raise ValueError("window >= 3 and 3 <= min_n <= window")
        self.window, self.min_n, self.clip_sigma, self.sigma_floor = window, min_n, clip_sigma, sigma_floor
        self._ring: deque[float] = deque()
        self._sorted: list[float] = []
        self.n_total = 0                 # values ever accepted (seeds excluded)
        self.n_clipped = 0
        self._cache: Optional[tuple[float, float]] = None

    def __len__(self) -> int:
        return len(self._sorted)

    @property
    def ready(self) -> bool:
        return len(self._sorted) >= self.min_n

    def add(self, x: float) -> bool:
        """Add a value. False (and no change) if it is not finite."""
        if not math.isfinite(x):
            return False
        if self.ready:
            med, sig = self.stats()
            lim = self.clip_sigma * max(sig, self.sigma_floor, 1e-12)
            if abs(x - med) > lim:
                x = med + math.copysign(lim, x - med)
                self.n_clipped += 1
        if len(self._ring) >= self.window:
            old = self._ring.popleft()
            del self._sorted[bisect_left(self._sorted, old)]
        self._ring.append(x)
        insort(self._sorted, x)
        self.n_total += 1
        self._cache = None
        return True

    @staticmethod
    def _median(a: list[float]) -> float:
        n = len(a)
        m = n // 2
        return a[m] if n % 2 else 0.5 * (a[m - 1] + a[m])

    def stats(self) -> tuple[float, float]:
        """(median, sigma). (nan, nan) when empty; sigma is nan until two values exist."""
        if self._cache is not None:
            return self._cache
        s = self._sorted
        if not s:
            return math.nan, math.nan
        med = self._median(s)
        if len(s) < 2:
            self._cache = (med, math.nan)
        else:
            mad = self._median(sorted(abs(v - med) for v in s))
            sig = MAD_TO_SIGMA * mad
            if sig == 0.0:                                    # more than half the values identical: fall back to the IQR
                q1, q3 = s[len(s) // 4], s[(3 * len(s)) // 4]
                sig = (q3 - q1) / 1.349
            self._cache = (med, max(sig, self.sigma_floor))
        return self._cache

    @property
    def median(self) -> float:
        return self.stats()[0]

    @property
    def sigma(self) -> float:
        return self.stats()[1]

    def seed(self, median: float, sigma: float, k: int = 8) -> None:
        """Continue from a stored (median, sigma): fill the window with k quantile points of N(median, sigma). New data pushes the
        seeds out after `window` samples; until then the stored values keep the statistic from jumping on the first outlier."""
        if not (math.isfinite(median) and math.isfinite(sigma)) or k < 1:
            return
        nd = NormalDist()
        for i in range(k):
            z = nd.inv_cdf((i + 0.5) / k)
            v = median + z * max(sigma, 0.0)
            self._ring.append(v)
            insort(self._sorted, v)
        while len(self._ring) > self.window:
            old = self._ring.popleft()
            del self._sorted[bisect_left(self._sorted, old)]
        self._cache = None
