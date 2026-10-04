"""Quality verdict policy.

Key principle (selection-bias guard): a frame is DROPPED only on evidence that does not depend on the baseline model
being good: broken 3D skeleton, broken picture, a person that is imperceptible in the pixels, or a person the model
sees that the labels do not contain. "The model fails on a person who is clearly visible" is NOT a reason to drop:
that is the most valuable kind of training frame, so it is kept and tagged as hard.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .features import static_difficulty
from .interfaces import IQualityPolicy
from .types import FrameFeatures, QualityMetrics, QualityVerdict, Tier


@dataclass
class PolicyConfig:
    invisible_perceptibility: float = 0.12   # a labeled person below this is not perceivable in the pixels
    blank_luma_std: float = 0.01             # whole-frame contrast below this = blank picture
    saturated_frac: float = 0.97             # share of pure black/white pixels above this = broken exposure
    blind_oks: float = 0.15                  # mean OKS below this = the model does not see the person
    hard_oks: float = 0.60                   # mean OKS below this = the model struggles
    hard_difficulty: float = 0.65            # static difficulty above this = hard even without a model
    suspect_conf: float = 0.70               # confident prediction disagreeing with labels (keypoints elsewhere)
    suspect_action: str = "quarantine"       # quarantine | keep_hard
    phantom_action: str = "quarantine"       # quarantine | hard  ("drop" = quarantine, kept for old configs)
    min_contrast_rate: float = 0.0           # explicit gate: a person whose limbs differ less from the background is
                                             # dropped (0 = off, the combined perceptibility score decides)
    min_brightness: float = 0.0              # explicit gate on the visible limbs' luma (0 = off)


class DefaultQualityPolicy(IQualityPolicy):
    def __init__(self, cfg: PolicyConfig = PolicyConfig()) -> None:
        self.cfg = cfg
        self.version = 0

    def reconfigure(self, spec: dict) -> None:
        """Apply new thresholds between frames (hot reload). Unknown keys are an error: a typo must not silently
        leave the old rule in force. The version is stamped on every later verdict."""
        known = {f.name for f in dataclasses.fields(PolicyConfig)}
        unknown = sorted(set(spec) - known)
        if unknown:
            raise ValueError(f"unknown policy keys {unknown}; known: {sorted(known)}")
        self.cfg = PolicyConfig(**{**dataclasses.asdict(self.cfg), **spec})
        self.version += 1

    def _verdict(self, tier: Tier, reasons: list[str], difficulty: float, m: QualityMetrics,
                 f: FrameFeatures) -> QualityVerdict:
        weight = {Tier.KEEP: 1.0, Tier.KEEP_HARD: 1.0 + difficulty, Tier.HARD_NEGATIVE: 1.5}.get(tier, 0.0)
        if tier is Tier.KEEP_HARD:
            utility = 1.0 if any("blind" in r for r in reasons) else 0.8 + 0.2 * difficulty
        elif tier is Tier.HARD_NEGATIVE:
            utility = 0.9
        elif tier is Tier.KEEP:
            utility = 0.1 + 0.4 * difficulty
        else:
            utility = 0.0
        return QualityVerdict(tier, reasons, float(difficulty), f.mean_occlusion, f.mean_contrast, weight, utility, m, f,
                              self.version)

    def decide(self, metrics: QualityMetrics, features: FrameFeatures, is_negative: bool) -> QualityVerdict:
        c = self.cfg
        static = static_difficulty(features)
        difficulty = static
        if metrics.evaluated and metrics.persons and not is_negative:
            difficulty = 0.5 * static + 0.5 * (1.0 - metrics.mean_oks)

        if features.gt_issues:
            return self._verdict(Tier.DROP_GT_SANITY, features.gt_issues[:4], difficulty, metrics, features)
        if features.has_pixels:
            if features.luma_std < c.blank_luma_std or features.saturated_frac > c.saturated_frac:
                return self._verdict(Tier.DROP_RENDER, [f"blank_or_saturated(std={features.luma_std:.3f},"
                                                        f"sat={features.saturated_frac:.2f})"], difficulty, metrics, features)
        if not is_negative:
            bad = [p for p in features.persons if p.perceptibility < c.invisible_perceptibility
                   or (c.min_contrast_rate and p.contrast_rate < c.min_contrast_rate)
                   or (c.min_brightness and p.brightness < c.min_brightness)]
            if bad:
                why = ", ".join(f"entity {p.entity_id}: limited_by={p.limiting_factor or 'n/a'} "
                                f"(contrast={p.contrast_rate:.3f}, luma={p.brightness:.3f}, h={p.size_px:.0f}px)" for p in bad[:3])
                return self._verdict(Tier.DROP_INVISIBLE, [f"imperceptible_person({why})"], difficulty, metrics, features)

        if metrics.evaluated:
            if metrics.n_phantom > 0:
                if c.phantom_action == "hard":
                    tier = Tier.HARD_NEGATIVE if is_negative else Tier.KEEP_HARD
                    return self._verdict(tier, [f"phantom_detections={metrics.n_phantom}"], max(difficulty, 0.7), metrics, features)
                return self._verdict(Tier.DROP_PHANTOM, [f"unlabeled_person_detected({metrics.n_phantom})"], difficulty,
                                     metrics, features)
            if is_negative:
                return self._verdict(Tier.KEEP, [], 0.0, metrics, features)
            if metrics.max_disagreement_score >= c.suspect_conf:
                tier = Tier.SUSPECT if c.suspect_action == "quarantine" else Tier.KEEP_HARD
                return self._verdict(tier, [f"model_confident_but_disagrees(conf={metrics.max_disagreement_score:.2f})"],
                                     max(difficulty, 0.7), metrics, features)
            if metrics.mean_oks < c.blind_oks:
                return self._verdict(Tier.KEEP_HARD, [f"model_blind_but_visible(oks={metrics.mean_oks:.2f})"],
                                     max(difficulty, 0.8), metrics, features)
            if metrics.mean_oks < c.hard_oks:
                return self._verdict(Tier.KEEP_HARD, [f"model_struggles(oks={metrics.mean_oks:.2f})"], difficulty,
                                     metrics, features)
        if static >= c.hard_difficulty and not is_negative:
            return self._verdict(Tier.KEEP_HARD, [f"static_difficulty={static:.2f}"], difficulty, metrics, features)
        return self._verdict(Tier.KEEP, [], difficulty, metrics, features)
