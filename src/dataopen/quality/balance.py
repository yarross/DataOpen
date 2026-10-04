"""Dataset-level balance control for the adaptive randomizer.

The per-bin tilts in `feedback.py` answer "which parameter values produce valuable frames". They do not answer two
questions that matter for the dataset as a whole:

  * the stream is drowning in easy frames          -> push harder toward the edge cases;
  * the generator has slid into extreme conditions -> too many hard / rejected frames: the distribution has collapsed
                                                      into a corner (out-of-distribution) -> pull back toward the prior.

`BalanceController` watches a sliding window of verdicts and turns three global knobs (all bounded):

    gamma_scale   multiplies the sharpness of the tilts (`FeedbackConfig.gamma`) and the share of edge-case reuse
    uniform_mix   share of draws that bypass every tilt and sample the full prior (coverage guarantee, unbiased bins)
    drop_penalty  extra suppression of bins that keep producing rejected frames (dark / unrealistic regions)

`drift_report` measures how far the REALIZED distribution of each parameter moved from the prior (KL divergence,
normalized entropy, least-covered bin), which is the number to show a reviewer: "the randomization did not collapse".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import QualityVerdict, Tier, Verdict


@dataclass
class BalanceConfig:
    enabled: bool = True
    window: int = 200                  # verdicts the shares are computed over
    min_usable: int = 20               # usable (clean + hard) frames needed before the knobs move
    target_hard_share: float = 0.30    # desired share of keep_hard among usable frames
    deadband: float = 0.04             # no reaction inside target +- deadband
    hard_margin: float = 0.10          # above target + margin: the generator is in a corner -> widen
    max_reject_share: float = 0.15     # share of attempts that may be rejected before bad regions are suppressed
    gain: float = 2.0                  # gamma_scale *= exp(-gain * error) per update
    gamma_min: float = 0.3
    gamma_max: float = 2.0
    exploit_max: float = 0.4
    uniform_min: float = 0.10
    uniform_max: float = 0.50
    uniform_step: float = 0.10
    drop_penalty: float = 2.0          # exponent on (1 - drop_rate) of a bin while over the reject budget
    kl_warn: float = 0.35              # per-parameter KL(realized || prior) above this is reported as drift
    min_share_warn: float = 0.08       # a bin realized below this fraction of its prior mass is reported as starved


class BalanceController:
    def __init__(self, cfg: BalanceConfig) -> None:
        self.cfg = cfg
        self.window: deque[tuple[str, bool]] = deque(maxlen=cfg.window)     # (verdict, wasted attempt)
        self.gamma_scale = 1.0
        self.uniform_mix = cfg.uniform_min
        self.drop_pressure = 0.0
        self.updates = 0

    def observe(self, v: QualityVerdict) -> None:
        self.window.append((v.verdict.value, v.tier is Tier.REJECTED))

    # ---- shares over the window ----
    def shares(self) -> dict[str, float]:
        n = len(self.window)
        if not n:
            return {"n": 0, "hard_share": 0.0, "reject_share": 0.0, "quarantine_share": 0.0, "clean_share": 0.0}
        c = {k: 0 for k in (v.value for v in Verdict)}
        for verdict, _ in self.window:
            c[verdict] += 1
        usable = c["keep_clean"] + c["keep_hard"]
        return {"n": n, "usable": usable, "hard_share": c["keep_hard"] / usable if usable else 0.0,
                "clean_share": c["keep_clean"] / n, "reject_share": c["reject"] / n,
                "quarantine_share": c["quarantine"] / n}

    def update(self) -> None:
        c = self.cfg
        if not c.enabled:
            return
        s = self.shares()
        self.updates += 1
        self.drop_pressure = c.drop_penalty if s["reject_share"] > c.max_reject_share else 0.0
        if s.get("usable", 0) < c.min_usable:
            return
        err = s["hard_share"] - c.target_hard_share
        if abs(err) > c.deadband:
            self.gamma_scale = float(np.clip(self.gamma_scale * np.exp(-c.gain * err), c.gamma_min, c.gamma_max))
        if err > c.hard_margin:
            self.uniform_mix = min(c.uniform_max, self.uniform_mix + c.uniform_step)
        else:
            self.uniform_mix = max(c.uniform_min, self.uniform_mix - c.uniform_step / 2)

    def exploit_p(self, base: float) -> float:
        return float(np.clip(base * self.gamma_scale, 0.0, self.cfg.exploit_max)) if self.cfg.enabled else base

    # ---- persistence / report ----
    def state(self) -> dict[str, Any]:
        return {"gamma_scale": self.gamma_scale, "uniform_mix": self.uniform_mix, "drop_pressure": self.drop_pressure,
                "updates": self.updates, "window": [list(w) for w in self.window]}

    def load_state(self, d: dict[str, Any]) -> None:
        self.gamma_scale = float(d.get("gamma_scale", 1.0))
        self.uniform_mix = float(d.get("uniform_mix", self.cfg.uniform_min))
        self.drop_pressure = float(d.get("drop_pressure", 0.0))
        self.updates = int(d.get("updates", 0))
        self.window.extend((str(a), bool(b)) for a, b in d.get("window", []))

    def report(self) -> dict[str, Any]:
        return {"enabled": self.cfg.enabled, "target_hard_share": self.cfg.target_hard_share,
                "max_reject_share": self.cfg.max_reject_share, "gamma_scale": round(self.gamma_scale, 3),
                "uniform_mix": round(self.uniform_mix, 3), "drop_pressure": self.drop_pressure,
                "window": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.shares().items()}}


def drift_stats(counts: np.ndarray, base_mass: np.ndarray) -> dict[str, float]:
    """Realized frequencies (`counts` per bin) against the prior mass of the same bins."""
    n = counts.sum()
    if n <= 0:
        return {"kl": 0.0, "entropy_ratio": 1.0, "min_share_ratio": 1.0}
    p = counts / n
    m = base_mass / base_mass.sum()
    nz = p > 0
    kl = float(np.sum(p[nz] * np.log(p[nz] / m[nz])))
    h = float(-np.sum(p[nz] * np.log(p[nz])))
    h0 = float(-np.sum(m[m > 0] * np.log(m[m > 0])))
    return {"kl": round(kl, 4), "entropy_ratio": round(h / h0, 4) if h0 > 0 else 1.0,
            "min_share_ratio": round(float(np.min(p[m > 0] / m[m > 0])), 4)}
