"""Test doubles for the closed loop. NEVER use for real validation: `SimulatedEvaluator` peeks at the ground truth.

It exists so the whole feedback loop (verdicts, adaptive randomization, reports) can be exercised without a model:
its "detections" are the ground truth degraded as a function of what the PIXELS show, i.e. a dark, low-contrast or
tiny person becomes hard to "detect", just like a real network.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ...core.models import Annotation
from ...core.randomization import derive_seed
from ...core.schema import SkeletonSchema
from ..features import FeatureConfig, person_features
from ..interfaces import IModelEvaluator
from ..types import Prediction


class SimulatedEvaluator(IModelEvaluator):
    name = "simulated"
    wants_hints = True

    def __init__(self, schema: SkeletonSchema, seed: int = 0, midpoint: float = 0.35, sharpness: float = 9.0,
                 noise: float = 0.25, cfg: Optional[FeatureConfig] = None, focus_midpoint: float = 0.10) -> None:
        self.schema, self.seed, self.mid, self.k, self.noise = schema, seed, midpoint, sharpness, noise
        self.focus_mid = focus_midpoint
        self.cfg = cfg or FeatureConfig()

    def predict(self, images: Sequence[np.ndarray], hints: Optional[Sequence[Sequence[Annotation]]] = None
                ) -> list[list[Prediction]]:
        if hints is None:
            raise ValueError("SimulatedEvaluator needs the ground-truth hints (it is a test double)")
        result = []
        for img, anns in zip(images, hints):
            fingerprint = int(img[::16, ::16].astype(np.int64).sum())
            preds = []
            for a in anns:
                pf = person_features(img, a, self.schema, self.cfg)
                evidence = pf.perceptibility * (1.0 - 0.5 * pf.occlusion_index)
                p = 1.0 / (1.0 + np.exp(-(evidence - self.mid) * self.k))
                rng = np.random.default_rng(derive_seed(self.seed, fingerprint, a.entity_id))
                if rng.random() > p:
                    continue
                sigma = (1.0 - p) * self.noise * np.sqrt(max(a.area, 1.0))
                kp = a.keypoints.copy()
                kp[:, :2] += rng.normal(0.0, sigma, size=(len(kp), 2))
                kp[:, 2] = np.where(a.keypoints[:, 2] > 0, 0.3 + 0.7 * p, 0.0)
                prim = [i for i in self.schema.primary_idx() if a.keypoints[i, 2] > 0]
                if prim and pf.has_focus:
                    # the aim point is judged on ITS OWN pixels: a small, low-contrast or half-hidden head is missed even
                    # when the body is found (this is what a real detector does and what the loop has to learn)
                    ev_f = pf.focus_perceptibility * (1.0 - 0.5 * (1.0 - pf.focus_visibility))
                    pf_ok = 1.0 / (1.0 + np.exp(-(ev_f - self.focus_mid) * self.k))
                    if rng.random() > pf_ok:
                        ang = rng.uniform(0, 2 * np.pi)
                        kp[prim, 0] += np.cos(ang) * 0.18 * a.bbox[3]
                        kp[prim, 1] += np.sin(ang) * 0.18 * a.bbox[3]
                        kp[prim, 2] = 0.1
                    else:
                        kp[prim, :2] = a.keypoints[prim, :2] + rng.normal(0.0, (1.0 - pf_ok) * 0.04 * a.bbox[3], (len(prim), 2))
                x, y, w, h = a.bbox
                cls = a.class_id
                if len(self.schema.classes) > 1 and rng.random() < 0.25 * (1.0 - p):
                    cls = (cls + 1) % len(self.schema.classes)        # a faint person gets the wrong team
                vis = np.where(a.keypoints[:, 2] == 2, 1.0, 0.0)
                preds.append(Prediction((x + rng.normal(0, sigma), y + rng.normal(0, sigma), w, h), float(0.3 + 0.7 * p), kp,
                                        cls if len(self.schema.classes) > 1 else None, vis))
            result.append(preds)
        return result


class CallableEvaluator(IModelEvaluator):
    """Wrap any function `fn(images) -> list[list[Prediction]]` (your own runtime, a remote service...)."""

    def __init__(self, fn: Callable[[Sequence[np.ndarray]], list[list[Prediction]]], name: str = "callable",
                 has_keypoints: bool = True) -> None:
        self.fn, self.name, self.has_keypoints = fn, name, has_keypoints

    def predict(self, images, hints=None):
        return self.fn(images)
