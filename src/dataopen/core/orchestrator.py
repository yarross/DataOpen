"""Dataset Orchestrator: session -> scenes -> frames.

Two-level loop on purpose: scene-level changes (level, time of day, weather, actor
population) are expensive; frame-level changes (camera, animation phase) are cheap.
Collecting many frames per scene is the single biggest throughput lever.

Production guarantees:
  * every committed image has a record (partial scenes are flushed even when the adapter fails);
  * a systematically wrong adapter (probe mismatch, repeating errors) aborts loudly instead of
    silently producing a bad or empty dataset;
  * sessions are resumable (scene-granular manifest) and reproducible (seeds derive from indices);
  * N parallel game instances can each take every N-th scene (`shard_index`/`shard_count`).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .annotation import AnnotationBuilder, AnnotationConfig, BuildResult
from .calibration import ProbeResult, check_probes, diagnose
from .card import write_dataset_card
from .export import CanonicalStore, write_coco, write_yolo_label, write_yolo_yaml
from .interfaces import AdapterError, IGameAdapter
from .models import CaptureRequest, FrameKind, FrameRecord, FrameSnapshot, FrameSpec, SceneSpec
from .randomization import DomainRandomizationController, IDomainRandomizer, derive_seed
from .validation import IFrameValidator, default_validators

if TYPE_CHECKING:  # the quality subsystem is optional: the core runs without importing it
    from ..quality.pipeline import QualityOutcome, QualityPipeline, RejectSink

log = logging.getLogger("dataopen")


class SessionAborted(RuntimeError):
    """The session stopped because continuing would produce bad data or no data."""


class CalibrationError(SessionAborted):
    """The engine's own projection disagrees with the labels we would write."""


@dataclass
class SessionConfig:
    out_dir: Path
    seed: int = 0
    target_frames: int = 1000            # accepted frames (positives + negatives)
    frames_per_scene: int = 50
    negative_ratio: float = 0.1
    max_wall_time_s: Optional[float] = None
    max_attempt_factor: float = 3.0      # hard cap on attempts = factor * target
    max_consecutive_failures: int = 25   # scenes in a row that produced nothing
    adapter_restart_after: int = 3       # consecutive AdapterErrors before a hard restart
    max_adapter_error_streak: int = 10   # consecutive AdapterErrors before giving up
    probe_tolerance_px: float = 3.0
    probe_abort_after: int = 5           # consecutive probe mismatches before CalibrationError
    image_ext: str = "png"
    resume: bool = False
    formats: Sequence[str] = ("yolo", "coco")
    shard_index: int = 0                 # this instance takes scenes shard_index, +count, +2*count...
    shard_count: int = 1
    provenance: dict[str, Any] = field(default_factory=dict)  # asset licence/source notes for the card

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        if not (0 <= self.shard_index < self.shard_count):
            raise ValueError("need 0 <= shard_index < shard_count")


@dataclass
class SessionReport:
    accepted: int = 0
    attempts: int = 0
    scenes: int = 0
    adapter_errors: int = 0
    probe_checked: int = 0
    probe_failed: int = 0
    quarantined: int = 0               # frames stored in <out>/quarantine for human audit (not part of the dataset)
    rejects: dict[str, int] = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0
    stop_reason: str = ""
    error: str = ""
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.accepted / self.wall_seconds if self.wall_seconds else 0.0


@dataclass
class _Pending:
    """A frame that passed the cheap checks and is being validated by the quality pipeline while the engine moves on."""

    spec: FrameSpec
    snap: FrameSnapshot
    scene: SceneSpec
    fi: int
    kind: FrameKind
    built: BuildResult
    future: "Future[QualityOutcome]"
    probe_err: Optional[float]


class DatasetOrchestrator:
    def __init__(
        self,
        adapter: IGameAdapter,
        config: SessionConfig,
        randomizer: Optional[IDomainRandomizer] = None,
        builder: Optional[AnnotationBuilder] = None,
        validators: Optional[Sequence[IFrameValidator]] = None,
        quality: Optional["QualityPipeline"] = None,
    ) -> None:
        self.adapter, self.cfg = adapter, config
        self.schema = adapter.info.schema
        self.randomizer = randomizer or DomainRandomizationController(
            config.seed, adapter.parameter_space())
        self.builder = builder or AnnotationBuilder(self.schema, AnnotationConfig())
        self.validators = list(validators) if validators is not None else default_validators()
        self.store = CanonicalStore(config.out_dir)
        self.manifest_path = config.out_dir / "manifest.json"
        self._t: Counter[str] = Counter()
        self._stop = threading.Event()
        self._probe_streak = 0
        self._last_probe: Optional[ProbeResult] = None
        self.quality = quality
        self.feedback = randomizer_feedback(self.randomizer)
        self.rejects: Optional["RejectSink"] = None
        # quarantine is a small dataset of its own (images/, labels/, annotations/): the same viewers work on it
        self.qroot = config.out_dir / "quarantine"
        self.qstore = CanonicalStore(self.qroot)
        self._q_records: list[FrameRecord] = []
        if quality is not None:
            from ..quality.pipeline import RejectSink
            self.rejects = RejectSink(config.out_dir / "quality", quality.cfg.reject_samples_per_tier,
                                      quality.cfg.audit_fraction, config.seed)

    def request_stop(self) -> None:
        """Graceful stop (SIGINT handler): finish the current frame, flush, write reports."""
        self._stop.set()

    # ---- manifest (scene-granular checkpoint) ----
    def _load_manifest(self) -> dict:
        if self.cfg.resume and self.manifest_path.exists():
            fb = self.cfg.out_dir / "feedback_state.json"
            if self.feedback is not None and fb.exists():
                self.feedback.load_state(json.loads(fb.read_text()))
            return json.loads(self.manifest_path.read_text())
        if self.cfg.out_dir.exists() and any(p.name != "quality_policy.toml" for p in self.cfg.out_dir.iterdir()):
            raise FileExistsError(f"{self.cfg.out_dir} is not empty; use resume=True or another dir")
        return {"scenes_done": 0, "accepted": 0, "attempts": 0, "rejects": {}}

    def _save_manifest(self, rep: SessionReport, scenes_done: int) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"scenes_done": scenes_done, "accepted": rep.accepted,
                                   "attempts": rep.attempts, "rejects": rep.rejects}))
        tmp.replace(self.manifest_path)

    # ---- main loop ----
    def run(self) -> SessionReport:
        cfg = self.cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        m = self._load_manifest()
        rep = SessionReport(accepted=m["accepted"], attempts=m["attempts"], rejects=dict(m["rejects"]))
        local_scene = m["scenes_done"]
        t0 = time.perf_counter()
        consecutive_fail = since_restart = adapter_streak = 0
        max_attempts = int(cfg.target_frames * cfg.max_attempt_factor) + rep.attempts
        abort: Optional[SessionAborted] = None
        last_error = ""

        try:
            self.adapter.connect()
            while rep.accepted < cfg.target_frames:
                stop = self._stop_reason(rep, t0, max_attempts, consecutive_fail)
                if stop:
                    rep.stop_reason = stop
                    break
                scene = self.randomizer.sample_scene(cfg.shard_index + local_scene * cfg.shard_count)
                records: list[FrameRecord] = []
                try:
                    self._health_check()
                    self._run_scene(scene, rep, t0, max_attempts, records)
                    adapter_streak = since_restart = 0
                except SessionAborted:
                    self._flush_scene(scene, records)
                    raise
                except AdapterError as e:
                    rep.adapter_errors += 1
                    adapter_streak += 1
                    since_restart += 1
                    last_error = str(e)
                    log.warning("scene %d adapter error: %s", scene.scene_index, e)
                    self._flush_scene(scene, records)  # frames committed before the failure keep their labels
                    if adapter_streak >= cfg.max_adapter_error_streak:
                        raise SessionAborted(
                            f"{adapter_streak} consecutive adapter errors, giving up. Last: {e}") from e
                    if since_restart >= cfg.adapter_restart_after:
                        log.warning("restarting adapter after %d consecutive errors", since_restart)
                        self.adapter.restart()
                        since_restart = 0
                    local_scene += 1
                    continue
                consecutive_fail = consecutive_fail + 1 if not records else 0
                self._flush_scene(scene, records)
                local_scene += 1
                rep.scenes += 1
                self._save_manifest(rep, local_scene)
            else:
                rep.stop_reason = "target_reached"
            if (rep.accepted == m["accepted"] and rep.adapter_errors and
                    rep.stop_reason in ("attempt_limit", "too_many_empty_scenes")):
                raise SessionAborted(f"no frames were collected and the adapter kept failing. Last: {last_error}")
        except SessionAborted as e:
            abort = e
            rep.stop_reason = "aborted"
            rep.error = str(e)
        except AdapterError as e:  # connect()/restart() failed
            abort = SessionAborted(f"adapter unavailable: {e}")
            rep.stop_reason = "aborted"
            rep.error = str(abort)
        finally:
            try:
                self.adapter.close()
            except AdapterError:
                pass
            if self.quality is not None:
                self.quality.close()
        self._save_manifest(rep, local_scene)
        self._finalize(rep, time.perf_counter() - t0)
        if abort is not None:
            raise abort
        return rep

    def _stop_reason(self, rep, t0, max_attempts, consecutive_fail) -> str:
        if self._stop.is_set():
            return "interrupted"
        if self.cfg.max_wall_time_s and time.perf_counter() - t0 > self.cfg.max_wall_time_s:
            return "wall_time_limit"
        if rep.attempts >= max_attempts:
            return "attempt_limit"
        if consecutive_fail >= self.cfg.max_consecutive_failures:
            return "too_many_empty_scenes"
        return ""

    def _timed(self, stage: str, t_start: float) -> None:
        self._t[stage] += time.perf_counter() - t_start

    def _health_check(self) -> None:
        health = getattr(self.adapter, "health", None)
        if callable(health):
            res = health()
            if isinstance(res, dict) and res.get("ok") is False:
                raise AdapterError(f"mod reports unhealthy: {res}")

    def _check_probes(self, snap, rep: SessionReport) -> Optional[str]:
        res = check_probes(snap)
        if res is None:
            return None
        rep.probe_checked += 1
        self._last_probe = res
        if res.ok(self.cfg.probe_tolerance_px):
            self._probe_streak = 0
            return None
        rep.probe_failed += 1
        self._probe_streak += 1
        if self._probe_streak >= self.cfg.probe_abort_after:
            cam = snap.camera
            raise CalibrationError(
                f"the engine's projection disagrees with the reported camera on {self._probe_streak} frames in a row "
                f"(max error {res.max_err:.1f}px, median {res.median_err:.1f}px, tolerance "
                f"{self.cfg.probe_tolerance_px}px). Likely cause: "
                f"{diagnose(res, cam.width, cam.height, self.cfg.probe_tolerance_px)}")
        return "projection_probe_mismatch"

    def _run_scene(self, scene: SceneSpec, rep: SessionReport, t0: float, max_attempts: int,
                   records: list[FrameRecord]) -> None:
        a = self.adapter
        t = time.perf_counter()
        a.environment.apply(scene)
        handles = a.spawner.spawn(scene)
        self._timed("scene_setup", t)
        inflight: deque[_Pending] = deque()
        max_inflight = self.quality.cfg.max_inflight if self.quality is not None else 0
        try:
            for fi in range(self.cfg.frames_per_scene):
                self._drain(inflight, rep, records, wait=False)
                if rep.accepted + len(inflight) >= self.cfg.target_frames or rep.attempts >= max_attempts:
                    break
                if self._stop.is_set() or (self.cfg.max_wall_time_s
                                            and time.perf_counter() - t0 > self.cfg.max_wall_time_s):
                    break
                kind_u = (derive_seed(self.cfg.seed, "kind", scene.scene_index, fi) % 10_000) / 10_000
                kind = FrameKind.NEGATIVE if kind_u < self.cfg.negative_ratio else FrameKind.POSITIVE
                spec = self.randomizer.sample_frame(scene, fi, kind)
                rep.attempts += 1

                t = time.perf_counter()
                a.spawner.set_active(handles, kind is FrameKind.POSITIVE)
                a.spawner.update_actors(handles, spec)
                w, h = a.info.image_size
                snap = a.capture.capture(CaptureRequest(spec.frame_id, spec, w, h))
                self._timed("capture", t)
                try:
                    res = self._stage_a(spec, snap, scene, fi, kind, rep, records)
                except BaseException:
                    a.capture.discard(snap)
                    raise
                if isinstance(res, str):
                    rep.rejects[res] = rep.rejects.get(res, 0) + 1
                    a.capture.discard(snap)
                    if self.feedback is not None:       # a wasted attempt is information too: avoid these parameters
                        from ..quality.types import rejected_verdict
                        self.feedback.observe(scene, spec, rejected_verdict(res))
                elif isinstance(res, _Pending):
                    inflight.append(res)
                    while len(inflight) > max_inflight:
                        self._finish(inflight.popleft(), rep, records)
            self._drain(inflight, rep, records, wait=True)
        except BaseException:
            for p in inflight:                       # best effort: free engine-side pixels and shared-memory slots
                p.future.cancel()
                try:
                    a.capture.discard(p.snap)
                except Exception:
                    pass
            raise
        finally:
            try:
                a.spawner.despawn_all()
            except AdapterError as e:
                log.warning("despawn_all failed: %s", e)

    def _stage_a(self, spec: FrameSpec, snap: FrameSnapshot, scene: SceneSpec, fi: int, kind: FrameKind,
                 rep: SessionReport, records: list[FrameRecord]):
        """Cheap synchronous checks. Returns a reject reason, a _Pending (quality validation running), or None
        (accepted and committed right away when there is no quality pipeline)."""
        t = time.perf_counter()
        probe_reason = self._check_probes(snap, rep)
        self._timed("probe_check", t)
        if probe_reason:
            return probe_reason
        probe_err = round(self._last_probe.max_err, 3) if (self._last_probe is not None and snap.probes) else None

        t = time.perf_counter()
        built = self.builder.build(snap)
        self._timed("annotate", t)

        t = time.perf_counter()
        reason = next((r for v in self.validators if (r := v.check(spec, snap, built))), None)
        self._timed("validate", t)
        if reason:
            return reason

        if self.quality is None:
            self._commit(spec, snap, scene, fi, kind, built, None, probe_err, rep, records)
            return None

        from ..quality.pipeline import QualityItem
        t = time.perf_counter()
        pixels = self.adapter.capture.peek_pixels(snap, self.quality.cfg.max_side)
        self._timed("peek_pixels", t)
        item = QualityItem(spec.frame_id, built.annotations, snap.entities, kind is FrameKind.NEGATIVE,
                           built.ignored_boxes, pixels, rep.attempts)
        return _Pending(spec, snap, scene, fi, kind, built, self.quality.submit(item), probe_err)

    def _drain(self, inflight: "deque[_Pending]", rep: SessionReport, records: list[FrameRecord], wait: bool) -> None:
        while inflight and (wait or inflight[0].future.done()):
            self._finish(inflight.popleft(), rep, records)

    def _finish(self, p: _Pending, rep: SessionReport, records: list[FrameRecord]) -> None:
        """Apply the quality verdict of a frame (in capture order): commit it, or drop it and keep the evidence."""
        from ..quality.pipeline import EvaluatorFailure
        from ..quality.types import Verdict as QualityVerdictClass
        t = time.perf_counter()
        try:
            outcome = p.future.result()
        except EvaluatorFailure as e:
            self.adapter.capture.discard(p.snap)
            raise SessionAborted(f"the quality evaluator keeps failing: {e}") from e
        self._timed("quality_wait", t)
        v, pixels = outcome.verdict, outcome.pixels
        try:
            if v.verdict is QualityVerdictClass.QUARANTINE:
                self._commit(p.spec, p.snap, p.scene, p.fi, p.kind, p.built, v, p.probe_err, rep, records,
                             quarantine=True)
            elif v.tier.is_drop:
                reason = f"quality_{v.tier.value}"
                rep.rejects[reason] = rep.rejects.get(reason, 0) + 1
                assert self.rejects is not None
                self.rejects.handle(p.spec.frame_id, v, pixels.array if pixels is not None else None)
                self.adapter.capture.discard(p.snap)
            else:
                self._commit(p.spec, p.snap, p.scene, p.fi, p.kind, p.built, v, p.probe_err, rep, records)
        finally:
            if pixels is not None:
                pixels.release()
        if self.feedback is not None:
            self.feedback.observe(p.scene, p.spec, v)

    def _commit(self, spec: FrameSpec, snap: FrameSnapshot, scene: SceneSpec, fi: int, kind: FrameKind,
                built: BuildResult, verdict, probe_err: Optional[float], rep: SessionReport,
                records: list[FrameRecord], quarantine: bool = False) -> None:
        t = time.perf_counter()
        rel = Path("images") / scene.split / f"{spec.frame_id}.{self.cfg.image_ext}"
        base = self.qroot if quarantine else self.cfg.out_dir      # file_name is relative to the (sub)dataset root
        self.adapter.capture.commit(snap, base / rel)
        self._timed("commit_image", t)
        meta: dict[str, Any] = {"seed": spec.seed, "tick": snap.tick, "environment": scene.environment,
                                "camera": asdict(spec.camera), "warnings": built.warnings,
                                "actors": scene.actors, "actor_frame": spec.actor_frame,
                                "units": {"scene": scene.units, "frame": spec.units}}
        if probe_err is not None:
            meta["probe_max_err_px"] = probe_err
        if verdict is not None:
            meta["quality"] = verdict.to_dict()
            feats = {pf.entity_id: pf for pf in verdict.features.persons}
            matches = {m.entity_id: m for m in verdict.metrics.persons}
            for ann in built.annotations:
                pf, m = feats.get(ann.entity_id), matches.get(ann.entity_id)
                diff = None
                if pf is not None:
                    diff = pf.difficulty
                    ann.meta.update({"occlusion_index": round(pf.occlusion_index, 4),
                                     "contrast_rate": round(pf.contrast_rate, 4),
                                     "perceptibility": round(pf.perceptibility, 4)})
                if m is not None and verdict.metrics.evaluated:
                    ann.meta["oks"] = ann.meta["oks_score"] = round(m.oks, 4)
                    if diff is not None:
                        diff = 0.5 * diff + 0.5 * (1.0 - m.oks)      # same blend as the frame-level difficulty
                if diff is not None:
                    ann.meta["difficulty_score"] = round(diff, 4)
            rep.quality.setdefault("tiers", {})
            rep.quality["tiers"][verdict.tier.value] = rep.quality["tiers"].get(verdict.tier.value, 0) + 1
        rec = FrameRecord(spec.frame_id, scene.scene_index, fi, scene.split, rel.as_posix(),
                          snap.camera.width, snap.camera.height, kind, built.annotations, meta)
        if quarantine:
            self._q_records.append(rec)
            rep.quarantined += 1
        else:
            records.append(rec)
            rep.accepted += 1

    def _flush_scene(self, scene: SceneSpec, records: list[FrameRecord]) -> None:
        t = time.perf_counter()
        if records:
            self.store.append(scene.split, records)
            if "yolo" in self.cfg.formats:
                for r in records:
                    write_yolo_label(r, self.cfg.out_dir / "labels")
        if self._q_records:
            for split in sorted({r.split for r in self._q_records}):
                self.qstore.append(split, [r for r in self._q_records if r.split == split])
            if "yolo" in self.cfg.formats:
                for r in self._q_records:
                    write_yolo_label(r, self.qroot / "labels")
            self._q_records = []
        if self.feedback is not None and hasattr(self.feedback, "save"):
            self.feedback.save(self.cfg.out_dir / "feedback_state.json")
        self._timed("export", t)

    def _finalize(self, rep: SessionReport, wall_s: float) -> None:
        root = self.cfg.out_dir
        splits = self.store.splits()
        if "coco" in self.cfg.formats:
            for s in splits:
                write_coco(self.store.load(s), self.schema, root / "annotations" / f"coco_{s}.json")
        if "yolo" in self.cfg.formats:
            write_yolo_yaml(root, self.schema, splits)
        qsplits = self.qstore.splits()
        if "coco" in self.cfg.formats:
            for s in qsplits:
                write_coco(self.qstore.load(s), self.schema, self.qroot / "annotations" / f"coco_{s}.json")
        rep.wall_seconds = wall_s
        rep.stage_seconds = {k: round(v, 3) for k, v in self._t.items()}
        if self.quality is not None or self.feedback is not None:
            from ..quality.report import write_closed_loop_report
            stats = self.quality.stats() if self.quality is not None else None
            rep.quality = {**rep.quality, **({"pipeline": stats} if stats else {})}
            write_closed_loop_report(root, stats, dict(self.rejects.total) if self.rejects else {},
                                     self.feedback.report() if self.feedback is not None else None,
                                     (r for s_ in splits for r in self.store.load(s_)),
                                     [(s_, r) for s_ in qsplits for r in self.qstore.load(s_)])
        (root / "report.json").write_text(json.dumps(asdict(rep), indent=2))
        write_dataset_card(root, self.adapter, self.cfg, rep, self.schema, self.store)


def randomizer_feedback(randomizer) -> Optional[Any]:
    """The randomizer, if it can learn from verdicts (duck-typed: no import of the quality package needed)."""
    return randomizer if hasattr(randomizer, "observe") and hasattr(randomizer, "state") else None
