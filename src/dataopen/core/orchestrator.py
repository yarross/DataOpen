"""Dataset Orchestrator: session -> scenes -> frames.

Two-level loop on purpose: scene-level changes (level, time of day, weather, actor
population) are expensive; frame-level changes (camera, animation phase) are cheap.
Collecting many frames per scene is the single biggest throughput lever.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from .annotation import AnnotationBuilder, AnnotationConfig
from .export import CanonicalStore, write_coco, write_yolo_label, write_yolo_yaml
from .interfaces import AdapterError, IGameAdapter
from .models import CaptureRequest, FrameKind, FrameRecord, SceneSpec
from .randomization import DomainRandomizationController, IDomainRandomizer, derive_seed
from .validation import IFrameValidator, default_validators

log = logging.getLogger("dataopen")


@dataclass
class SessionConfig:
    out_dir: Path
    seed: int = 0
    target_frames: int = 1000            # accepted frames (positives + negatives)
    frames_per_scene: int = 50
    negative_ratio: float = 0.1
    max_wall_time_s: Optional[float] = None
    max_attempt_factor: float = 3.0      # hard cap on attempts = factor * target
    max_consecutive_failures: int = 25
    adapter_restart_after: int = 3       # consecutive AdapterErrors before a hard restart
    image_ext: str = "png"
    resume: bool = False
    formats: Sequence[str] = ("yolo", "coco")


@dataclass
class SessionReport:
    accepted: int = 0
    attempts: int = 0
    scenes: int = 0
    adapter_errors: int = 0
    rejects: dict[str, int] = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0
    stop_reason: str = ""

    @property
    def fps(self) -> float:
        return self.accepted / self.wall_seconds if self.wall_seconds else 0.0


class DatasetOrchestrator:
    def __init__(
        self,
        adapter: IGameAdapter,
        config: SessionConfig,
        randomizer: Optional[IDomainRandomizer] = None,
        builder: Optional[AnnotationBuilder] = None,
        validators: Optional[Sequence[IFrameValidator]] = None,
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

    # ---- manifest (scene-granular checkpoint) ----
    def _load_manifest(self) -> dict:
        if self.cfg.resume and self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        existing = self.cfg.out_dir.exists() and any(self.cfg.out_dir.iterdir())
        if existing:
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
        scene_index = m["scenes_done"]
        t0 = time.perf_counter()
        consecutive_fail = consecutive_adapter_err = 0
        max_attempts = int(cfg.target_frames * cfg.max_attempt_factor) + rep.attempts

        self.adapter.connect()
        try:
            while rep.accepted < cfg.target_frames:
                stop = self._stop_reason(rep, t0, max_attempts, consecutive_fail)
                if stop:
                    rep.stop_reason = stop
                    break
                scene = self.randomizer.sample_scene(scene_index)
                try:
                    records = self._run_scene(scene, rep, t0, max_attempts)
                    consecutive_adapter_err = 0
                except AdapterError as e:
                    rep.adapter_errors += 1
                    consecutive_adapter_err += 1
                    log.warning("scene %d adapter error: %s", scene_index, e)
                    if consecutive_adapter_err >= cfg.adapter_restart_after:
                        self.adapter.restart()
                        consecutive_adapter_err = 0
                    scene_index += 1
                    continue
                consecutive_fail = consecutive_fail + 1 if not records else 0
                self._flush_scene(scene, records)
                scene_index += 1
                rep.scenes += 1
                self._save_manifest(rep, scene_index)
            else:
                rep.stop_reason = "target_reached"
        finally:
            self.adapter.close()
        self._finalize()
        rep.wall_seconds = time.perf_counter() - t0
        rep.stage_seconds = {k: round(v, 3) for k, v in self._t.items()}
        (cfg.out_dir / "report.json").write_text(json.dumps(asdict(rep), indent=2))
        return rep

    def _stop_reason(self, rep, t0, max_attempts, consecutive_fail) -> str:
        if self.cfg.max_wall_time_s and time.perf_counter() - t0 > self.cfg.max_wall_time_s:
            return "wall_time_limit"
        if rep.attempts >= max_attempts:
            return "attempt_limit"
        if consecutive_fail >= self.cfg.max_consecutive_failures:
            return "too_many_empty_scenes"
        return ""

    def _timed(self, stage: str, t_start: float) -> None:
        self._t[stage] += time.perf_counter() - t_start

    def _run_scene(self, scene: SceneSpec, rep: SessionReport, t0: float, max_attempts: int) -> list[FrameRecord]:
        a = self.adapter
        t = time.perf_counter()
        a.environment.apply(scene)
        handles = a.spawner.spawn(scene)
        self._timed("scene_setup", t)
        records: list[FrameRecord] = []
        try:
            for fi in range(self.cfg.frames_per_scene):
                if rep.accepted >= self.cfg.target_frames or rep.attempts >= max_attempts:
                    break
                if self._stop_reason(rep, t0, max_attempts, 0) in ("wall_time_limit",):
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

                t = time.perf_counter()
                built = self.builder.build(snap)
                self._timed("annotate", t)

                t = time.perf_counter()
                reason = next((r for v in self.validators if (r := v.check(spec, snap, built))), None)
                self._timed("validate", t)

                if reason:
                    rep.rejects[reason] = rep.rejects.get(reason, 0) + 1
                    a.capture.discard(snap)
                    continue

                t = time.perf_counter()
                rel = Path("images") / scene.split / f"{spec.frame_id}.{self.cfg.image_ext}"
                a.capture.commit(snap, self.cfg.out_dir / rel)
                self._timed("commit_image", t)
                records.append(FrameRecord(
                    spec.frame_id, scene.scene_index, fi, scene.split, rel.as_posix(), w, h, kind,
                    built.annotations,
                    {"seed": spec.seed, "tick": snap.tick, "environment": scene.environment,
                     "camera": asdict(spec.camera), "warnings": built.warnings}))
                rep.accepted += 1
        finally:
            a.spawner.despawn_all()
        return records

    def _flush_scene(self, scene: SceneSpec, records: list[FrameRecord]) -> None:
        t = time.perf_counter()
        if records:
            self.store.append(scene.split, records)
            if "yolo" in self.cfg.formats:
                for r in records:
                    write_yolo_label(r, self.cfg.out_dir / "labels")
        self._timed("export", t)

    def _finalize(self) -> None:
        root = self.cfg.out_dir
        splits = self.store.splits()
        if "coco" in self.cfg.formats:
            for s in splits:
                write_coco(self.store.load(s), self.schema, root / "annotations" / f"coco_{s}.json")
        if "yolo" in self.cfg.formats:
            write_yolo_yaml(root, self.schema, splits)
