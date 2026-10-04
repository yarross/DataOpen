"""`dataopen doctor`: a ladder of checks that tells you what is wrong with a game integration
and how to fix it, before you collect a single training image.

Each step PASSes, WARNs or FAILs with a concrete hint, and the run leaves overlay images in
<out>/doctor/ so a human can confirm at a glance that the skeleton sits on the characters.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .annotation import AnnotationBuilder
from .calibration import check_probes, diagnose
from .imageio import read_image, write_png
from .interfaces import AdapterError, IGameAdapter
from .models import CaptureRequest, FrameKind
from .projection import project
from .randomization import DomainRandomizationController
from .viz import draw_annotations

# (joint a, joint b, min m, max m): typical adult segment lengths, used to catch wrong bone mapping
_SEGMENTS = (("l_shoulder", "l_elbow", 0.15, 0.45), ("l_elbow", "l_wrist", 0.15, 0.42),
             ("pelvis", "l_knee", 0.28, 0.65), ("l_knee", "l_ankle", 0.28, 0.62), ("head", "neck", 0.06, 0.40))


@dataclass
class Check:
    name: str
    status: str            # PASS | WARN | FAIL | INFO
    detail: str = ""
    hint: str = ""


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.checks.append(Check(name, status, detail, hint))

    @property
    def ok(self) -> bool:
        return not any(c.status == "FAIL" for c in self.checks)

    def render(self) -> str:
        out = []
        for c in self.checks:
            out.append(f"[{c.status:<4}] {c.name}: {c.detail}")
            if c.hint and c.status in ("WARN", "FAIL"):
                out.append(f"         -> {c.hint}")
        out.append("")
        out.append("RESULT: " + ("ready to collect" if self.ok else "NOT ready: fix the FAIL items above"))
        return "\n".join(out)


def _span(skel: np.ndarray, valid: np.ndarray) -> float:
    pts = skel[valid]
    if len(pts) < 2:
        return 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return float(d.max())


def run_doctor(adapter: IGameAdapter, out_dir: Path, frames: int = 6, tol_px: float = 3.0, seed: int = 0,
               builder: Optional[AnnotationBuilder] = None, image_ext: str = "png") -> DoctorReport:
    rep = DoctorReport()
    out = Path(out_dir) / "doctor"
    out.mkdir(parents=True, exist_ok=True)

    # 1. connect
    try:
        adapter.connect()
        info = adapter.info
    except AdapterError as e:
        rep.add("connect", "FAIL", str(e), "start the game with the DataOpen mod loaded and check the mailbox path "
                "matches on both sides (profile mailbox_dir / the mod's data folder)")
        return _finish(rep, out)
    hello = getattr(adapter, "hello", {}) or {}
    rep.add("connect", "PASS", f"{info.name} ({info.engine}) game_version={hello.get('game_version')} "
            f"mod_version={hello.get('mod_version')} capture_mode={getattr(adapter, 'capture_mode', 'direct')} "
            f"image={info.image_size[0]}x{info.image_size[1]}")
    rep.add("capabilities", "INFO", ", ".join(sorted(c.value for c in info.capabilities)) or "none declared")
    schema = info.schema
    builder = builder or AnnotationBuilder(schema)

    # 2. mod-side self tests
    if hasattr(adapter, "selftest"):
        try:
            for c in adapter.selftest():
                st = "PASS" if c.get("ok") else "FAIL"
                rep.add(f"mod:{c.get('name')}", st, str(c.get("detail", "")), str(c.get("hint", "")))
                unmapped = (c.get("data") or {}).get("unmapped")
                if unmapped:
                    found = (c.get("data") or {}).get("bones_found", [])
                    rep.add("bone_mapping", "FAIL", f"unmapped keypoints: {unmapped}",
                            f"override them in the profile's [bones] table. Bones found on the model: {found[:60]}")
        except AdapterError as e:
            rep.add("mod:selftest", "WARN", str(e), "the mod does not implement selftest")

    # 3. scene + spawn
    rz = DomainRandomizationController(seed, adapter.parameter_space())
    scene = rz.sample_scene(0)
    handles: list = []
    try:
        adapter.environment.apply(scene)
        handles = adapter.spawner.spawn(scene)
    except AdapterError as e:
        rep.add("begin_scene", "FAIL", str(e), "population/environment binding failed: see the mod log; try "
                "mod_options.population_mode = 'observe' to use existing NPCs")
        return _finish(rep, out)
    rep.add("begin_scene", "PASS" if handles else "WARN", f"{len(handles)} actors, environment={scene.environment}",
            "" if handles else "no actors spawned: fine in observe mode, otherwise the spawn binding is broken")

    probe_bad = probe_total = 0
    lr_ok = lr_bad = 0
    spans: list[float] = []
    valid_frac: list[float] = []
    invalid_names: dict[str, int] = {}
    seg_vals: dict[tuple, list[float]] = {}
    verdicts: dict[str, int] = {}
    cap_times: list[float] = []
    accepted_frames = 0
    problems: list[Check] = []
    try:
        for i in range(frames):
            spec = rz.sample_frame(scene, i, FrameKind.POSITIVE)
            adapter.spawner.set_active(handles, True)
            adapter.spawner.update_actors(handles, spec)
            w, h = info.image_size
            t = time.perf_counter()
            try:
                snap = adapter.capture.capture(CaptureRequest(spec.frame_id, spec, w, h))
            except AdapterError as e:
                rep.add(f"capture[{i}]", "FAIL", str(e), "the capture binding failed: check the mod log")
                break
            cap_times.append(time.perf_counter() - t)
            cam = snap.camera
            pr = check_probes(snap)
            if pr is None:
                if i == 0:
                    rep.add("probes", "WARN", "the mod sends no probes",
                            "without probes the projection cannot be verified; implement `probes` in capture_frame")
            else:
                probe_total += 1
                if not pr.ok(tol_px):
                    probe_bad += 1
                    problems.append(Check("projection", "FAIL",
                                          f"frame {i}: max error {pr.max_err:.1f}px (compared {pr.compared}, "
                                          f"missing {pr.missing})", diagnose(pr, cam.width, cam.height, tol_px)))
            cam_fwd = cam.world_to_camera[2, :3]
            built = builder.build(snap)
            for v in built.verdicts.values():
                verdicts[v.value] = verdicts.get(v.value, 0) + 1
            for e in snap.entities:
                spans.append(_span(e.skeleton_world, e.joint_valid))
                valid_frac.append(float(e.joint_valid.mean()))
                for j in np.where(~e.joint_valid)[0]:
                    invalid_names[schema.keypoints[j]] = invalid_names.get(schema.keypoints[j], 0) + 1
                for a, b, lo, hi in _SEGMENTS:
                    if a in schema.keypoints and b in schema.keypoints:
                        ia, ib = schema.index(a), schema.index(b)
                        if e.joint_valid[ia] and e.joint_valid[ib]:
                            seg_vals.setdefault((a, b, lo, hi), []).append(
                                float(np.linalg.norm(e.skeleton_world[ia] - e.skeleton_world[ib])))
                fwd = e.meta.get("forward")
                lr = [schema.index("l_shoulder"), schema.index("r_shoulder")] if "l_shoulder" in schema.keypoints else None
                if fwd is not None and lr is not None and e.joint_valid[lr].all():
                    d = float(np.dot(np.asarray(fwd, dtype=float), cam_fwd))
                    uv, z = project(e.skeleton_world[lr], cam)
                    if abs(d) > 0.3 and (z > cam.near).all() and abs(uv[0, 0] - uv[1, 0]) > 4:
                        left_is_right_of_screen = uv[0, 0] > uv[1, 0]
                        if left_is_right_of_screen == (d < 0):   # facing the camera => left appears on screen-right
                            lr_ok += 1
                        else:
                            lr_bad += 1
            # image round trip + overlay
            dest = out / f"frame_{i}.{image_ext}"
            try:
                adapter.capture.commit(snap, dest)
            except AdapterError as e:
                rep.add(f"commit[{i}]", "FAIL", str(e), "the mod could not write the image: check the path is "
                        "writable for the game process")
                break
            try:
                img = read_image(dest)
                if img.shape[:2] != (cam.height, cam.width):
                    problems.append(Check("image_size", "FAIL",
                                          f"image is {img.shape[1]}x{img.shape[0]}, camera says {cam.width}x{cam.height}",
                                          "render size and reported camera size must match exactly"))
                elif float(img.std()) < 1.5:
                    problems.append(Check("image_content", "WARN", f"frame {i} looks blank (std={img.std():.2f})",
                                          "black/flat frame: loading screen, menu, HUD or wrong capture region"))
                write_png(out / f"overlay_{i}.png", draw_annotations(img, built.annotations, schema))
            except (RuntimeError, ValueError, OSError) as e:
                problems.append(Check("overlay", "WARN", f"could not read back {dest.name}: {e}",
                                      "install Pillow (pip install 'dataopen[capture]') for JPEG/odd PNG files"))
            if built.annotations:
                accepted_frames += 1

        # negative frame
        try:
            spec = rz.sample_frame(scene, frames, FrameKind.NEGATIVE)
            adapter.spawner.set_active(handles, False)
            snap = adapter.capture.capture(CaptureRequest(spec.frame_id, spec, *info.image_size))
            built = builder.build(snap)
            present = sum(1 for v in built.verdicts.values() if v.value != "absent")
            adapter.capture.discard(snap)
            rep.add("negative_frame", "PASS" if present == 0 else "WARN",
                    "no person visible when actors are hidden" if present == 0 else f"{present} persons still visible",
                    "" if present == 0 else "set_active(false) did not hide everyone; in observe mode negatives are "
                    "found by placement and filtered by the validator")
        except AdapterError as e:
            rep.add("negative_frame", "WARN", str(e))
    finally:
        try:
            adapter.spawner.despawn_all()
        except AdapterError:
            pass

    # ---- evaluate collected evidence ----
    if probe_total:
        if probe_bad == 0:
            rep.add("projection", "PASS", f"engine and core projections agree within {tol_px}px on {probe_total} frames")
        else:
            rep.checks.append(problems[0] if problems and problems[0].name == "projection" else
                              Check("projection", "FAIL", f"{probe_bad}/{probe_total} frames disagree"))
    rep.checks.extend(p for p in problems if p.name != "projection")
    if spans:
        med = float(np.median(spans))
        if 0.8 <= med <= 2.6:
            rep.add("units", "PASS", f"median skeleton extent {med:.2f} m")
        elif 80 <= med <= 260:
            rep.add("units", "FAIL", f"skeleton extent {med:.0f}: looks like CENTIMETERS",
                    "convert to meters in the mod (multiply positions by 0.01), including the camera position")
        else:
            rep.add("units", "WARN", f"unusual skeleton extent {med:.2f} m",
                    "check units and that the mapped bones belong to one humanoid")
        vf = float(np.mean(valid_frac))
        rep.add("joints", "PASS" if vf > 0.95 else "WARN", f"{vf:.0%} of joints resolved",
                "" if vf > 0.95 else f"frequently missing: {sorted(invalid_names, key=invalid_names.get, reverse=True)[:6]}; "
                "fix with the profile's [bones] overrides")
        bad_seg = [f"{a}->{b} median {np.median(v):.2f}m (expected {lo}-{hi})" for (a, b, lo, hi), v in seg_vals.items()
                   if v and not lo <= float(np.median(v)) <= hi]
        if 0.8 <= med <= 2.6:
            rep.add("bone_lengths", "PASS" if not bad_seg else "WARN", "segment lengths plausible" if not bad_seg
                    else "; ".join(bad_seg), "" if not bad_seg else "a keypoint is probably mapped to the wrong bone")
    else:
        rep.add("entities", "WARN", "no persons captured in any frame",
                "nothing to label: spawn failed, camera never sees the actors, or observe mode found nobody")
    if lr_ok + lr_bad:
        if lr_bad == 0:
            rep.add("left_right", "PASS", f"left/right consistent with the image on {lr_ok} persons")
        elif lr_ok == 0:
            rep.add("left_right", "FAIL", f"left and right look SWAPPED on {lr_bad} persons",
                    "swap l_/r_ in the bone mapping, or the camera 'right' vector sign is mirrored")
        else:
            rep.add("left_right", "WARN", f"mixed: {lr_ok} consistent, {lr_bad} inconsistent",
                    "some rigs map left/right differently: check per-rig bone names")
    else:
        rep.add("left_right", "INFO", "skipped (entities carry no 'forward' vector)")
    if verdicts:
        rep.add("annotation_verdicts", "INFO", str(verdicts))
    if cap_times:
        mean = float(np.mean(cap_times))
        rate = 1.0 / mean if mean > 0 else float("inf")
        acc = accepted_frames / max(1, frames)
        rep.extra["capture_seconds_mean"] = mean
        rep.add("throughput", "INFO", f"{mean * 1000:.0f} ms/capture ~ {rate:.1f} fps ~ "
                f"{3600 * rate * max(acc, 0.1):,.0f} accepted frames/hour per instance (acc. rate ~{acc:.0%})")
    return _finish(rep, out)


def _finish(rep: DoctorReport, out: Path) -> DoctorReport:
    (out / "doctor_report.json").write_text(json.dumps(
        {"ok": rep.ok, "checks": [asdict(c) for c in rep.checks], "extra": rep.extra}, indent=2))
    return rep
