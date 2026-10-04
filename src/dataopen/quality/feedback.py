"""Feedback-driven domain randomization.

Mechanism: every parameter is sampled through its inverse CDF `from_unit(u)`. Between the stratified draw `u` and the
value we insert a piecewise-constant WARP over bins of the unit interval. After each verdict the frame's *utility*
(0 = wasted, 1 = ideal edge case) is credited to the bin of every parameter it used; bins that keep producing edge
cases get a higher tilt, bins that keep producing dropped or trivial frames a lower one. Properties:

  * marginal and cheap: one tilt vector per parameter, no model of the joint space, no tuning per game;
  * bounded: tilts are clipped to [floor, cap] and mixed with an exploration bonus (UCB), so coverage never collapses;
  * categorical parameters (skins, outfits, weapons) are bins too, so the loop learns WHICH APPEARANCES break the model
    without anybody enumerating combinations;
  * edge-case mining: with probability `exploit_p` a new draw is a jittered copy of a remembered hard example.

Reproducibility: values depend on the feedback history, so every record stores the unit draws that produced it
(`SceneSpec.units`, `FrameSpec.units`) and the state is checkpointed to feedback_state.json.
"""
from __future__ import annotations

import itertools
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.models import FrameSpec, SceneSpec
from ..core.randomization import Categorical, Constant, DomainRandomizationController, Param
from .interfaces import IFeedbackController
from .types import QualityVerdict, Tier


@dataclass
class FeedbackConfig:
    warmup_frames: int = 150          # uniform sampling until there is evidence
    update_every: int = 25            # frames between re-computing the tilts
    bins: int = 8                     # bins per continuous parameter
    gamma: float = 1.0                # sharpness of the tilt (utility ratio ** gamma)
    floor: float = 0.15               # lowest tilt: a bin keeps at least this share of its natural mass
    cap: float = 4.0                  # highest tilt
    shrink_k: float = 5.0             # pseudo-observations pulling a bin's mean toward the global mean
    ucb_c: float = 0.35               # exploration bonus for rarely seen bins
    smooth: float = 0.5               # weight of the previous tilt when updating
    exploit_p: float = 0.2            # share of draws that are jittered copies of remembered hard examples
    exploit_sigma: float = 0.07
    memory: int = 300
    hard_utility: float = 0.8         # utility at/above which a frame is remembered as an edge case


class UnitWarp:
    """Piecewise-constant density over the unit interval: base masses `m` times tilts `t`."""

    def __init__(self, masses: np.ndarray) -> None:
        self.m = np.asarray(masses, dtype=np.float64)
        self.m = self.m / self.m.sum()
        self.base_cum = np.concatenate([[0.0], np.cumsum(self.m)])
        self.t = np.ones_like(self.m)
        self._refresh()

    def _refresh(self) -> None:
        q = self.m * self.t
        self.q = q / q.sum()
        self.q_cum = np.concatenate([[0.0], np.cumsum(self.q)])

    def set_tilt(self, t: np.ndarray) -> None:
        self.t = np.asarray(t, dtype=np.float64)
        self._refresh()

    def apply(self, u: float) -> float:
        b = int(min(np.searchsorted(self.q_cum, u, side="right") - 1, len(self.q) - 1))
        b = max(b, 0)
        r = (u - self.q_cum[b]) / self.q[b] if self.q[b] > 0 else 0.0
        return float(min(self.base_cum[b] + np.clip(r, 0.0, 1.0 - 1e-12) * self.m[b], 1.0 - 1e-12))

    def bin_of(self, u: float) -> int:
        """Bin (in the ORIGINAL unit space) that a post-warp unit falls into."""
        return int(min(max(np.searchsorted(self.base_cum, u, side="right") - 1, 0), len(self.m) - 1))


class _Stats:
    def __init__(self, n_bins: int) -> None:
        self.n = np.zeros(n_bins)
        self.util = np.zeros(n_bins)
        self.hard = np.zeros(n_bins)
        self.drop = np.zeros(n_bins)
        self.oks = np.zeros(n_bins)
        self.oks_n = np.zeros(n_bins)

    def add(self, b: int, v: QualityVerdict) -> None:
        self.n[b] += 1
        self.util[b] += v.utility
        self.hard[b] += v.tier in (Tier.KEEP_HARD, Tier.HARD_NEGATIVE)
        self.drop[b] += v.tier.is_drop
        if v.metrics.evaluated and v.metrics.n_gt:
            self.oks[b] += v.metrics.mean_oks
            self.oks_n[b] += 1


class AdaptiveRandomizer(DomainRandomizationController, IFeedbackController):
    def __init__(self, *args: Any, feedback: Optional[FeedbackConfig] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fb = feedback or FeedbackConfig()
        self._params: dict[str, Param] = {}
        self._warps: dict[str, UnitWarp] = {}
        self._stats: dict[str, _Stats] = {}
        self._memory: deque[dict[str, Any]] = deque(maxlen=self.fb.memory)
        self.n_observed = 0
        self.last_update = 0
        self._pairs: set[tuple] = set()
        spaces = {"env": self.env_space, "actor": self.actor_space, "cam": self.camera_space,
                  "frame": self.actor_frame_space}
        self._spaces = spaces
        for group, space in spaces.items():
            for name, p in space.params.items():
                if isinstance(p, Constant):
                    continue
                key = f"{group}.{name}"
                masses = self._masses(p)
                self._params[key] = p
                self._warps[key] = UnitWarp(masses)
                self._stats[key] = _Stats(len(masses))
        self._cat_keys = [k for k, p in self._params.items() if isinstance(p, Categorical) and k.startswith("actor.")
                          and len(p.choices) <= 64][:6]

    def _masses(self, p: Param) -> np.ndarray:
        if isinstance(p, Categorical):
            w = np.asarray(p.weights if p.weights else [1.0] * len(p.choices), dtype=float)
            return w / w.sum()
        return np.full(self.fb.bins, 1.0 / self.fb.bins)

    # ---- sampling hook ----
    @property
    def adapting(self) -> bool:
        return self.n_observed >= self.fb.warmup_frames

    def _units(self, group: str, names: list[str], units: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.adapting or not names:
            return units
        out = np.array(units, dtype=np.float64)
        # edge-case mining: reuse a remembered hard example's draws for this group, jittered
        if self._memory and rng.random() < self.fb.exploit_p:
            mem = self._memory[int(rng.integers(0, len(self._memory)))]
            src = self._memory_units(mem, group, rng)
            if src is not None and len(src) == len(names):
                jitter = rng.normal(0.0, self.fb.exploit_sigma, size=len(names))
                return np.clip(np.asarray(src) + jitter, 0.0, 1.0 - 1e-9)
        for i, name in enumerate(names):
            w = self._warps.get(f"{group}.{name}")
            if w is not None:
                out[i] = w.apply(float(out[i]))
        return out

    @staticmethod
    def _memory_units(mem: dict[str, Any], group: str, rng: np.random.Generator):
        if group == "env":
            return mem["scene"].get("env")
        if group == "actor":
            a = mem["scene"].get("actors") or []
            return a[int(rng.integers(0, len(a)))] if a else None
        if group == "cam":
            return mem["frame"].get("cam")
        f = mem["frame"].get("frame") or []
        return f[int(rng.integers(0, len(f)))] if f else None

    # ---- feedback ----
    def _credit(self, group: str, names: list[str], units: list[float], v: QualityVerdict) -> None:
        for name, u in zip(names, units):
            key = f"{group}.{name}"
            w = self._warps.get(key)
            if w is not None:
                self._stats[key].add(w.bin_of(float(u)), v)

    def observe(self, scene: SceneSpec, frame: FrameSpec, verdict: QualityVerdict) -> None:
        if scene.units:
            self._credit("env", self.env_space.names(), scene.units.get("env", []), verdict)
            for u in scene.units.get("actors", []):
                self._credit("actor", self.actor_space.names(), u, verdict)
            self._note_pairs(scene)
        if frame.units:
            self._credit("cam", self.camera_space.names(), frame.units.get("cam", []), verdict)
            for u in frame.units.get("frame", []):
                self._credit("frame", self.actor_frame_space.names(), u, verdict)
        if verdict.utility >= self.fb.hard_utility and (scene.units or frame.units):
            self._memory.append({"scene": scene.units, "frame": frame.units, "utility": verdict.utility})
        self.n_observed += 1
        if self.adapting and self.n_observed - self.last_update >= self.fb.update_every:
            self._update_tilts()
            self.last_update = self.n_observed

    def _note_pairs(self, scene: SceneSpec) -> None:
        names = self.actor_space.names()
        for u in scene.units.get("actors", []):
            bins = {}
            for k in self._cat_keys:
                nm = k.split(".", 1)[1]
                if nm in names:
                    bins[k] = self._warps[k].bin_of(float(u[names.index(nm)]))
            for (ka, ba), (kb, bb) in itertools.combinations(sorted(bins.items()), 2):
                self._pairs.add((ka, ba, kb, bb))

    def _update_tilts(self) -> None:
        c = self.fb
        for key, st in self._stats.items():
            total_n = st.n.sum()
            if total_n < 10:
                continue
            mu_g = st.util.sum() / total_n
            if mu_g <= 1e-9:
                continue
            mu_b = (st.util + c.shrink_k * mu_g) / (st.n + c.shrink_k)
            ucb = c.ucb_c * np.sqrt(np.log(total_n + 1.0) / (st.n + 1.0))
            target = np.clip((mu_b / mu_g) ** c.gamma + ucb, c.floor, c.cap)
            w = self._warps[key]
            w.set_tilt(c.smooth * w.t + (1.0 - c.smooth) * target)

    # ---- persistence ----
    def state(self) -> dict[str, Any]:
        return {"version": 1, "n_observed": self.n_observed, "last_update": self.last_update,
                "tilts": {k: w.t.tolist() for k, w in self._warps.items()},
                "stats": {k: {f: getattr(s, f).tolist() for f in ("n", "util", "hard", "drop", "oks", "oks_n")}
                          for k, s in self._stats.items()},
                "memory": list(self._memory), "pairs": [list(p) for p in self._pairs]}

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            return
        self.n_observed, self.last_update = state["n_observed"], state["last_update"]
        for k, t in state["tilts"].items():
            if k in self._warps and len(t) == len(self._warps[k].t):
                self._warps[k].set_tilt(np.asarray(t))
        for k, d in state["stats"].items():
            if k in self._stats and len(d["n"]) == len(self._stats[k].n):
                for f, v in d.items():
                    setattr(self._stats[k], f, np.asarray(v, dtype=float))
        self._memory.extend(state.get("memory", []))
        self._pairs = {tuple(p) for p in state.get("pairs", [])}

    def save(self, path: Path) -> None:
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(self.state()))
        tmp.replace(path)

    # ---- report ----
    def _bin_label(self, key: str, b: int) -> str:
        p = self._params[key]
        if isinstance(p, Categorical):
            return str(p.choices[b])
        w = self._warps[key]
        lo, hi = p.from_unit(float(w.base_cum[b])), p.from_unit(float(min(w.base_cum[b + 1], 1 - 1e-9)))
        return f"{lo:.3g}..{hi:.3g}"

    def report(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key, st in self._stats.items():
            rows = []
            for b in range(len(st.n)):
                n = st.n[b]
                rows.append({"bin": self._bin_label(key, b), "n": int(n),
                             "mean_utility": round(st.util[b] / n, 3) if n else None,
                             "hard_rate": round(st.hard[b] / n, 3) if n else None,
                             "drop_rate": round(st.drop[b] / n, 3) if n else None,
                             "mean_oks": round(st.oks[b] / st.oks_n[b], 3) if st.oks_n[b] else None,
                             "tilt": round(float(self._warps[key].t[b]), 3)})
            params[key] = {"covered_bins": int((st.n > 0).sum()), "bins": len(st.n), "rows": rows}
        pair_cov = None
        if len(self._cat_keys) >= 2:
            possible = sum(len(self._params[a].choices) * len(self._params[b].choices)
                           for a, b in itertools.combinations(self._cat_keys, 2))
            pair_cov = {"keys": self._cat_keys, "pairs_seen": len(self._pairs), "pairs_possible": possible,
                        "coverage": round(len(self._pairs) / possible, 3) if possible else None}
        return {"observed_frames": self.n_observed, "adapting": self.adapting, "remembered_hard_examples": len(self._memory),
                "parameters": params, "pairwise_appearance_coverage": pair_cov}
