"""Quality verdict policy.

Key principle (selection-bias guard): a frame is DROPPED only on evidence that does not depend on the baseline model
being good: broken 3D skeleton, broken picture, a person that is imperceptible in the pixels, or a person the model
sees that the labels do not contain. "The model fails on a person who is clearly visible" is NOT a reason to drop:
that is the most valuable kind of training frame, so it is kept and tagged as hard.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

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
    # ---- focus = the schema's primary keypoint (the aim point). All off when the schema has no primary keypoint. ----
    focus_priority: float = 0.7              # 0..1: how much the aim point steers difficulty and the randomizer's reward
    focus_blind_oks: float = 0.15            # aim-point OKS below this on a person that is clearly visible = model blind
    focus_hard_oks: float = 0.55             # below this = model struggles on the aim point
    focus_boost: float = 1.0                 # extra training weight: aim point missed, aim region itself fully visible
    focus_boost_partial: float = 0.5         # ... aim region partly hidden (occlusion is a legitimate hard case)
    focus_invisible_perceptibility: float = 0.03   # aim region indistinguishable from its surroundings: kept, no boost
    class_confusion_boost: float = 0.5       # extra weight when the model finds the person but the wrong class (team)


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

    def _focus_difficulty(self, m: QualityMetrics, f: FrameFeatures) -> float:
        """Difficulty of the aim point: its static perceptibility/visibility/size, blended with the model's trouble."""
        d = f.focus_difficulty
        if m.evaluated and m.focus_evaluated:
            d = 0.5 * d + 0.5 * (1.0 - m.mean_focus_oks)
        return float(np.clip(d, 0.0, 1.0))

    def _verdict(self, tier: Tier, reasons: list[str], difficulty: float, m: QualityMetrics,
                 f: FrameFeatures, boost: float = 0.0, focus_utility: Optional[float] = None) -> QualityVerdict:
        weight = {Tier.KEEP: 1.0, Tier.KEEP_HARD: 1.0 + difficulty, Tier.HARD_NEGATIVE: 1.5}.get(tier, 0.0)
        if weight > 0:
            weight += boost
        if tier is Tier.KEEP_HARD:
            utility = 1.0 if any("blind" in r for r in reasons) else 0.8 + 0.2 * difficulty
        elif tier is Tier.HARD_NEGATIVE:
            utility = 0.9
        elif tier is Tier.KEEP:
            utility = 0.1 + 0.4 * difficulty
        else:
            utility = 0.0
        fd = self._focus_difficulty(m, f) if f.has_focus else 0.0
        p = self.cfg.focus_priority if f.has_focus and not tier.is_drop else 0.0
        if p > 0:
            # The randomizer is rewarded for the AIM POINT being hard, not for the body being hard: blend the generic
            # reward with a focus reward (1.0 / 0.9 when the model misses it, otherwise the static aim-point difficulty).
            uf = focus_utility if focus_utility is not None else 0.1 + 0.8 * fd
            utility = max((1.0 - p) * utility + p * uf, uf if focus_utility is not None else 0.0)
        return QualityVerdict(tier, reasons, float(difficulty), f.mean_occlusion, f.mean_contrast, weight, utility, m, f,
                              self.version, float(fd), m.mean_focus_oks if (m.evaluated and m.focus_evaluated) else None)

    def _focus_verdict(self, m: QualityMetrics, f: FrameFeatures, difficulty: float) -> Optional[QualityVerdict]:
        """The model misses the aim point of a person who is clearly visible: the most valuable frame there is."""
        c = self.cfg
        feats = {p.entity_id: p for p in f.persons}
        worst = min((p for p in m.persons if p.focus_oks is not None), key=lambda p: p.focus_oks, default=None)
        if worst is None or worst.focus_oks >= c.focus_hard_oks:
            return None
        pf = feats.get(worst.entity_id)
        full = pf is not None and pf.focus_visibility >= 0.999
        hidden_look = pf is not None and full and pf.focus_perceptibility < c.focus_invisible_perceptibility
        boost = 0.0 if hidden_look else (c.focus_boost if full else c.focus_boost_partial)
        blind = worst.focus_oks < c.focus_blind_oks
        why = (f"entity={worst.entity_id}, focus_oks={worst.focus_oks:.2f}, aim_region="
               f"{'camouflaged' if hidden_look else 'visible' if full else 'partly_hidden'}")
        diff = max(difficulty, 0.85 if blind else 0.65)
        return self._verdict(Tier.KEEP_HARD, [f"model_blind_on_focus({why})" if blind else f"model_struggles_on_focus({why})"],
                             diff, m, f, boost=boost, focus_utility=1.0 if blind else 0.9)

    def decide(self, metrics: QualityMetrics, features: FrameFeatures, is_negative: bool) -> QualityVerdict:
        c = self.cfg
        static = static_difficulty(features)
        difficulty = static
        if metrics.evaluated and metrics.persons and not is_negative:
            difficulty = 0.5 * static + 0.5 * (1.0 - metrics.mean_oks)
        if features.has_focus and not is_negative and c.focus_priority > 0:
            difficulty = (1.0 - c.focus_priority) * difficulty + c.focus_priority * self._focus_difficulty(metrics, features)

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
            if metrics.focus_evaluated and c.focus_priority > 0:
                fv = self._focus_verdict(metrics, features, difficulty)
                if fv is not None:
                    return fv
            wrong = [p.entity_id for p in metrics.persons if p.class_ok is False and p.oks >= 0.5]
            if wrong:
                return self._verdict(Tier.KEEP_HARD, [f"class_confusion(entities={wrong})"], max(difficulty, 0.7), metrics,
                                     features, boost=c.class_confusion_boost, focus_utility=None)
            if metrics.mean_oks < c.blind_oks:
                return self._verdict(Tier.KEEP_HARD, [f"model_blind_but_visible(oks={metrics.mean_oks:.2f})"],
                                     max(difficulty, 0.8), metrics, features)
            if metrics.mean_oks < c.hard_oks:
                return self._verdict(Tier.KEEP_HARD, [f"model_struggles(oks={metrics.mean_oks:.2f})"], difficulty,
                                     metrics, features)
        if features.has_focus and not is_negative and c.focus_priority > 0:
            fd = self._focus_difficulty(metrics, features)
            if fd >= c.hard_difficulty:
                return self._verdict(Tier.KEEP_HARD, [f"static_focus_difficulty={fd:.2f}"], difficulty, metrics, features,
                                     boost=0.0)
        if static >= c.hard_difficulty and not is_negative:
            return self._verdict(Tier.KEEP_HARD, [f"static_difficulty={static:.2f}"], difficulty, metrics, features)
        return self._verdict(Tier.KEEP, [], difficulty, metrics, features)
