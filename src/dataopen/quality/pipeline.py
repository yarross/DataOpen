"""QualityPipeline: the in-memory validation stage between capture and commit.

    cheap gates (3D skeleton sanity, blank frame, imperceptible person)   -- no model, no GPU
        -> baseline model inference (only for frames that survived the gates, optionally every N-th frame)
        -> metrics (OKS / IoU / confidence, phantoms) -> verdict (tier, difficulty, weight, utility)

Evaluation runs on a worker pool so inference overlaps with the engine rendering the next frames; the orchestrator
keeps a bounded number of frames in flight (back-pressure by shared-memory slots).
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..core.imageio import write_png
from ..core.models import Annotation, EntityState
from ..core.randomization import derive_seed
from ..core.schema import SkeletonSchema
from .features import FeatureConfig, compute_features
from .interfaces import IModelEvaluator, IQualityMetricCalculator, IQualityPolicy, PixelHandle
from .metrics import DefaultMetricCalculator
from .policy import DefaultQualityPolicy, PolicyConfig
from .types import QualityMetrics, QualityVerdict


class EvaluatorFailure(RuntimeError):
    """The baseline model failed repeatedly: continuing would silently ship unvalidated frames."""


@dataclass
class QualityConfig:
    enabled: bool = True
    max_inflight: int = 2             # frames being evaluated while the engine renders the next ones
    workers: int = 1                  # evaluator threads (1 = serialized GPU use)
    sample_every: int = 1             # run the model on every N-th frame; the cheap gates always run
    max_side: Optional[int] = 1280    # peek at most this many pixels on the long side (downscaled by the mod if supported)
    match_conf: float = 0.25
    oks_match_thr: float = 0.5
    phantom_conf: float = 0.6
    audit_fraction: float = 0.02      # share of DROPPED frames saved to audit/ to measure the false-rejection rate
    reject_samples_per_tier: int = 25  # dropped frames saved per tier for human review
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass
class QualityItem:
    frame_id: str
    annotations: Sequence[Annotation]
    entities: Sequence[EntityState]
    is_negative: bool
    ignored_boxes: Sequence[Sequence[float]]
    pixels: Optional[PixelHandle]
    index: int = 0


@dataclass
class QualityOutcome:
    verdict: QualityVerdict
    pixels: Optional[PixelHandle]          # still held: the caller releases it after commit / reject handling


class QualityPipeline:
    def __init__(self, schema: SkeletonSchema, evaluator: Optional[IModelEvaluator], config: Optional[QualityConfig] = None,
                 calculator: Optional[IQualityMetricCalculator] = None, policy: Optional[IQualityPolicy] = None) -> None:
        self.schema, self.evaluator, self.cfg = schema, evaluator, config or QualityConfig()
        self.calc = calculator or DefaultMetricCalculator(schema, self.cfg.match_conf, self.cfg.oks_match_thr,
                                                          self.cfg.phantom_conf)
        self.policy = policy or DefaultQualityPolicy(self.cfg.policy)
        self.pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.workers), thread_name_prefix="dataopen-quality")
        self._lock = threading.Lock()
        self.tier_counts: Counter[str] = Counter()
        self.evaluated = 0
        self.gated = 0                      # dropped by the cheap gates, i.e. inference saved
        self.latency_ms_sum = 0.0
        self.no_pixels = 0
        self.errors = 0
        self._consecutive_errors = 0
        self.max_consecutive_errors = 3
        if evaluator is not None:
            evaluator.warmup()

    # ---- work unit ----
    def evaluate(self, item: QualityItem) -> QualityOutcome:
        img = item.pixels.array if item.pixels is not None else None
        if img is None:
            self.no_pixels += 1
        feats = compute_features(img, item.annotations, item.entities, self.schema, self.cfg.features)
        empty = QualityMetrics()
        v = self.policy.decide(empty, feats, item.is_negative)       # cheap gates first
        if v.tier.is_drop or self.evaluator is None or img is None:
            if v.tier.is_drop:
                self.gated += 1
            return self._done(v, item)
        if self.cfg.sample_every > 1 and item.index % self.cfg.sample_every != 0:
            return self._done(v, item)
        t0 = time.perf_counter()
        hints = [item.annotations] if self.evaluator.wants_hints else None
        try:
            preds = self.evaluator.predict([img], hints)[0]
        except Exception as e:       # a one-off failure degrades that frame to static validation; repeated ones stop the run
            with self._lock:
                self.errors += 1
                self._consecutive_errors += 1
                fatal = self._consecutive_errors >= self.max_consecutive_errors
            if fatal:
                raise EvaluatorFailure(f"{type(e).__name__}: {e}") from e
            v.reasons.append(f"evaluator_error({type(e).__name__})")
            return self._done(v, item)
        with self._lock:
            self._consecutive_errors = 0
        ms = (time.perf_counter() - t0) * 1000.0
        metrics = self.calc.compute(item.annotations, preds, self.evaluator.name, ms, item.ignored_boxes)
        with self._lock:
            self.evaluated += 1
            self.latency_ms_sum += ms
        return self._done(self.policy.decide(metrics, feats, item.is_negative), item)

    def _done(self, verdict: QualityVerdict, item: QualityItem) -> QualityOutcome:
        with self._lock:
            self.tier_counts[verdict.tier.value] += 1
        return QualityOutcome(verdict, item.pixels)

    def submit(self, item: QualityItem) -> "Future[QualityOutcome]":
        return self.pool.submit(self.evaluate, item)

    def stats(self) -> dict[str, Any]:
        n = sum(self.tier_counts.values())
        return {"frames": n, "tiers": dict(self.tier_counts), "evaluated_by_model": self.evaluated,
                "dropped_by_cheap_gates": self.gated, "frames_without_pixels": self.no_pixels,
                "evaluator_errors": self.errors,
                "mean_inference_ms": round(self.latency_ms_sum / self.evaluated, 2) if self.evaluated else None,
                "backend": self.evaluator.name if self.evaluator else None}

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        if self.evaluator is not None:
            self.evaluator.close()


class RejectSink:
    """Keeps evidence about dropped frames without polluting the dataset.

    rejects.jsonl           one line per dropped frame (verdict only)
    rejects/<tier>/*.png    a capped number of examples per tier, for human review
    audit/<tier>/*.png      a random share of ALL drops: if many audited frames look fine, the thresholds are too strict
    """

    def __init__(self, root: Path, cap_per_tier: int, audit_fraction: float, seed: int) -> None:
        self.root, self.cap, self.audit, self.seed = Path(root), cap_per_tier, audit_fraction, seed
        self.saved: dict[str, int] = defaultdict(int)
        self.audited: dict[str, int] = defaultdict(int)
        self.total: Counter[str] = Counter()

    def handle(self, frame_id: str, verdict: QualityVerdict, image: Optional[np.ndarray]) -> None:
        tier = verdict.tier.value
        self.total[tier] += 1
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "rejects.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"frame_id": frame_id, **verdict.to_dict()}, separators=(",", ":")) + "\n")
        if image is None:
            return
        picked = (derive_seed(self.seed, "audit", frame_id) % 10_000) / 10_000 < self.audit
        if picked and self.audited[tier] < 200:
            self.audited[tier] += 1
            write_png(self.root / "audit" / tier / f"{frame_id}.png", image)
        elif self.saved[tier] < self.cap:
            self.saved[tier] += 1
            write_png(self.root / "rejects" / tier / f"{frame_id}.png", image)
