"""dataopen: command line.

  dataopen games                                   list game profiles
  dataopen doctor  --game gmod [--mailbox DIR]     check a game integration before collecting
  dataopen collect --game gmod --out DIR --frames N
  dataopen eval-image --model M.onnx --image I.png   check an ONNX model + keypoint mapping on one image
  dataopen verify  DIR                             check a collected dataset, write qa_report.{json,md}
  dataopen preview DIR                             contact sheet with labels drawn on top
  dataopen merge   OUT SRC1 SRC2 ...               merge shards
  dataopen install --game gmod|valheim|rust --dir GAME_DIR   install the in-game mod
  dataopen serve-mock --mailbox DIR                a fake game that speaks the protocol (try everything offline)
  dataopen detector train|eval|export|calib|quantize|bench   the NPU keypoint detector (docs/DETECTOR.md)
  dataopen agreement A.json B.json                 IoU agreement between two annotation files
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import signal
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, Sequence

from .core.doctor import run_doctor
from .core.interfaces import AdapterError
from .core.merge import MergeError, merge_datasets
from .core.orchestrator import CalibrationError, DatasetOrchestrator, SessionAborted, SessionConfig
from .core.preview import make_preview
from .core.qa import verify_dataset, write_reports
from .factory import build_adapter
from .core.annotation import AnnotationBuilder, AnnotationConfig
from .core.schema_io import SchemaFileError, load_schema, resolve_target
from .quality.factory import QualityConfigError, build_quality, build_randomizer
from .profiles import ProfileError, list_profiles, load_profile

EXIT_OK, EXIT_FAILED_CHECK, EXIT_INCOMPLETE, EXIT_ABORTED, EXIT_USAGE = 0, 1, 2, 3, 4


def _common_game_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--game", required=True, help="profile id (see `dataopen games`) or path to a .toml profile")
    p.add_argument("--mailbox", help="mailbox directory shared with the game mod (overrides the profile)")


def _cmd_games(a) -> int:
    for p in list_profiles():
        print(f"{p.id:<10} {p.name}\n{'':<10} engine={p.engine}  archetype={p.archetype}\n"
              f"{'':<10} status: {p.status.get('verified', 'unknown')}")
    return EXIT_OK


def _cmd_doctor(a) -> int:
    prof = load_profile(a.game)
    adapter = build_adapter(prof, a.mailbox, connect=False)
    rep = run_doctor(adapter, Path(a.out), frames=a.frames, tol_px=a.tolerance, seed=a.seed,
                     target=a.schema or prof.schema.get("target"), target_params=prof.schema.get("params"))
    print(rep.render())
    print(f"\nOverlays and doctor_report.json: {Path(a.out) / 'doctor'}")
    return EXIT_OK if rep.ok else EXIT_FAILED_CHECK


def _quality_spec(profile_quality: dict, a) -> dict:
    """Profile [quality] table overridden by CLI flags."""
    spec = {**profile_quality}
    if a.quality_model:
        spec.update(enabled=True, evaluator="onnx", model=a.quality_model)
    if a.quality_format:
        spec["format"] = a.quality_format
    if a.quality_device:
        spec["device"] = a.quality_device
    if a.quality_sim:
        spec.update(enabled=True, evaluator="simulated")
    if getattr(a, "quality_runtime", False):
        spec["runtime"] = True
    if a.quality_static:
        spec.update(enabled=True, evaluator="none")
    if a.quality_inflight is not None:
        spec["max_inflight"] = a.quality_inflight
    if a.quality_every is not None:
        spec["sample_every"] = a.quality_every
    if getattr(a, "quality_policy", None):
        spec["policy_file"] = a.quality_policy
    return spec


def _cmd_eval_image(a) -> int:
    """Run the configured evaluator on ONE image and draw what it sees: validates a model + keypoint mapping."""
    import numpy as np
    from .core.imageio import read_image, write_png
    from .core.models import Annotation
    from .core.schema import HUMAN_13
    from .core.viz import draw_annotations
    from .quality.factory import QualityConfigError, build_evaluator
    bundle, schema = None, HUMAN_13
    if a.schema:
        try:
            bundle = load_schema(a.schema)
        except SchemaFileError as e:
            print(f"schema error: {e}")
            return EXIT_USAGE
        schema = bundle.schema
    spec = {"evaluator": "onnx", "model": a.model, "format": a.format, "device": a.device,
            "keypoint_map": a.keypoint_map or ("identity" if a.format == "table" else "coco17"), "conf_thr": a.conf}
    if a.layout:
        spec["layout"] = a.layout.split(",")
    if a.coords:
        spec["coords"] = a.coords
    try:
        ev = build_evaluator(spec, schema, bundle)
    except (QualityConfigError, ValueError) as e:
        print(f"cannot build the evaluator: {e}")
        return EXIT_USAGE
    img = read_image(Path(a.image))
    preds = ev.predict([img])[0]
    print(f"{ev.name}: {len(preds)} detections on {img.shape[1]}x{img.shape[0]} ({schema.name}, {schema.num_keypoints} points)")
    anns = []
    for i, p in enumerate(preds):
        cls = "" if p.class_id is None else f" class={schema.classes[p.class_id] if p.class_id < len(schema.classes) else p.class_id}"
        print(f"  #{i} score={p.score:.2f}{cls} bbox={tuple(round(v) for v in p.bbox)} "
              f"keypoints={'yes' if p.keypoints is not None else 'no (box-only model)'}")
        if p.keypoints is not None:
            kp = p.keypoints.copy()
            kp[:, 2] = np.where(kp[:, 2] > 0.3, 2, 0)
            for j in schema.primary_idx():                      # the aim point: print where the model puts it
                print(f"      {schema.keypoints[j]}: ({p.keypoints[j, 0]:.1f}, {p.keypoints[j, 1]:.1f}) conf={p.keypoints[j, 2]:.2f}")
            anns.append(Annotation(i, kp, p.bbox))
        else:
            anns.append(Annotation(i, np.zeros((schema.num_keypoints, 3)), p.bbox))
    out = Path(a.out or "eval_overlay.png")
    write_png(out, draw_annotations(img, anns, schema))
    print(f"overlay: {out}")
    return EXIT_OK


def _cmd_collect(a) -> int:
    prof = load_profile(a.game)
    out = Path(a.out)
    adapter = build_adapter(prof, a.mailbox, connect=False)
    if not a.no_doctor:
        tmp = Path(tempfile.mkdtemp(prefix="dataopen-doctor-"))
        rep = run_doctor(adapter, tmp, frames=3, seed=a.seed, target=a.schema or prof.schema.get("target"),
                         target_params=prof.schema.get("params"))
        if not rep.ok:
            print(rep.render())
            print(f"\nRefusing to collect: the integration check failed. Fix the FAIL items "
                  f"(details: {tmp / 'doctor'}) or pass --no-doctor.")
            return EXIT_FAILED_CHECK
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        adapter.connect()  # idempotent; the orchestrator needs adapter.info before run()
    except AdapterError as e:
        print(f"cannot connect to the game: {e}")
        return EXIT_FAILED_CHECK
    s = {k: v for k, v in prof.session.items() if k in {f.name for f in dataclasses.fields(SessionConfig)}}
    if a.frames_per_scene:
        s["frames_per_scene"] = a.frames_per_scene
    if a.negative_ratio is not None:
        s["negative_ratio"] = a.negative_ratio
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    cfg = SessionConfig(out_dir=out, seed=a.seed, target_frames=a.frames, resume=a.resume, shard_index=shard_i,
                        shard_count=shard_n, image_ext=a.image_ext,
                        provenance=dict(prof.provenance) | ({"note": a.provenance_note} if a.provenance_note else {}),
                        **{k: v for k, v in s.items() if k not in {"provenance"}})
    quality_spec = _quality_spec(prof.quality, a)
    try:
        bundle, mapping = resolve_target(a.schema or prof.schema.get("target"), adapter.info.schema,
                                         prof.schema.get("params"))
    except SchemaFileError as e:
        print(f"schema error: {e}")
        return EXIT_USAGE
    target_schema = mapping.target if mapping is not None else adapter.info.schema
    quality_spec.setdefault("policy_file", str(out / "quality_policy.toml"))   # may be created while the run is going
    builder = None
    if quality_spec.get("min_person_px") is not None:
        mp = float(quality_spec["min_person_px"])
        builder = AnnotationBuilder(target_schema, AnnotationConfig(min_bbox_height_px=mp,
                                                                         negligible_height_px=min(12.0, mp)))
    try:
        quality = build_quality(quality_spec, target_schema, rig_schema=adapter.info.schema, bundle=bundle)
        randomizer = build_randomizer(adapter, a.seed, a.adaptive, quality_spec.get("feedback"),
                                      quality_spec.get("balance"))
    except (QualityConfigError, ImportError, OSError, RuntimeError) as e:
        print(f"cannot start the quality subsystem: {e}")
        return EXIT_USAGE
    orch = DatasetOrchestrator(adapter, cfg, randomizer=randomizer, builder=builder, quality=quality, target=mapping)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: (print("\nstopping after the current frame..."), orch.request_stop()))
    try:
        rep = orch.run()
    except CalibrationError as e:
        print(f"\nCALIBRATION ERROR: {e}\nRun `dataopen doctor --game {a.game}` for the full diagnosis.")
        return EXIT_ABORTED
    except (SessionAborted, FileExistsError) as e:
        print(f"\nABORTED: {e}")
        return EXIT_ABORTED
    print(json.dumps({"accepted": rep.accepted, "attempts": rep.attempts, "fps": round(rep.fps, 2),
                      "rejects": rep.rejects, "stages_s": rep.stage_seconds, "stop": rep.stop_reason}, indent=2))
    return EXIT_OK if rep.stop_reason in ("target_reached", "interrupted", "wall_time_limit") else EXIT_INCOMPLETE


def _cmd_verify(a) -> int:
    rep = verify_dataset(Path(a.dataset), check_images=not a.no_images)
    write_reports(rep, Path(a.dataset))
    st = rep.stats
    print(f"frames={st.get('frames')} persons={st.get('persons')} scenes={st.get('scenes')} "
          f"errors={len(rep.errors)} warnings={len(rep.issues) - len(rep.errors)}")
    for i in rep.issues[:25]:
        print(f"  {i.severity:<7} {i.code:<28} {i.where}: {i.message}")
    print(f"report: {Path(a.dataset) / 'qa_report.md'}")
    return EXIT_OK if rep.ok else EXIT_FAILED_CHECK


def _cmd_requalify(a) -> int:
    from .quality.requalify import load_policy_spec, requalify
    try:
        res = requalify(Path(a.dataset), load_policy_spec(Path(a.policy)))
    except (OSError, ValueError) as e:
        print(f"requalify failed: {e}")
        return EXIT_USAGE
    print(f"frames={res['frames']} verdicts={res['verdicts']} excluded={res['excluded']}")
    for k, v in sorted(res["transitions"].items()):
        print(f"  {k}: {v}")
    print(f"written: {Path(a.dataset) / 'quality_requalified.csv'}, {Path(a.dataset) / 'excluded_frames.txt'}")
    return EXIT_OK


def _cmd_preview(a) -> int:
    out = make_preview(Path(a.dataset), n=a.n, seed=a.seed, out=Path(a.out) if a.out else None)
    print(out)
    return EXIT_OK


def _cmd_merge(a) -> int:
    try:
        card = merge_datasets([Path(s) for s in a.sources], Path(a.out))
    except MergeError as e:
        print(f"merge failed: {e}")
        return EXIT_FAILED_CHECK
    print(f"merged {len(a.sources)} datasets into {a.out}: {card['counts']['frames']} frames")
    return EXIT_OK


def _coerce(kind, value: str):
    kind = kind if isinstance(kind, str) else kind.__name__
    if kind == "bool":
        return True if value == "" else value.lower() not in ("0", "false", "no")
    return float(value) if kind == "float" else value


def _cmd_install(a) -> int:
    from .installer import NEXT_STEPS, InstallError, install
    try:
        paths = install(a.game, Path(a.dir))
    except InstallError as e:
        print(f"install failed: {e}")
        return EXIT_FAILED_CHECK
    for p in paths:
        print(f"installed: {p}")
    print("\nNext steps:\n" + NEXT_STEPS[a.game].format(dir=a.dir))
    return EXIT_OK


def _cmd_serve_mock(a) -> int:
    from .adapters.mock.server import MockServerOptions, serve_mock
    kw = {}
    types = {f.name: f.type for f in dataclasses.fields(MockServerOptions)}
    for item in a.option or []:
        k, _, v = item.partition("=")
        if k not in types:
            print(f"unknown option {k!r}; known: {', '.join(types)}")
            return EXIT_USAGE
        kw[k] = _coerce(types[k], v)
    print(f"mock game serving {a.mailbox} (Ctrl+C to stop) options={kw}")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    serve_mock(Path(a.mailbox), MockServerOptions(**kw), stop)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dataopen", description="Synthetic human-pose dataset collection")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("games", help="list game profiles").set_defaults(fn=_cmd_games)

    d = sub.add_parser("doctor", help="check a game integration")
    _common_game_args(d)
    d.add_argument("--out", default="doctor_out")
    d.add_argument("--frames", type=int, default=6)
    d.add_argument("--tolerance", type=float, default=3.0, help="projection probe tolerance, pixels")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--schema", help="also check this target keypoint schema (derived points, head geometry, team): shooter12 or a .toml")
    d.set_defaults(fn=_cmd_doctor)

    c = sub.add_parser("collect", help="collect a dataset")
    _common_game_args(c)
    c.add_argument("--out", required=True)
    c.add_argument("--frames", type=int, default=1000)
    c.add_argument("--frames-per-scene", type=int)
    c.add_argument("--negative-ratio", type=float)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--resume", action="store_true")
    c.add_argument("--shard", default="0/1", help="i/n: take every n-th scene (run n game instances)")
    c.add_argument("--image-ext", default="png", choices=["png", "jpg"])
    c.add_argument("--provenance-note", help="origin/licence of the assets, stored in DATASET_CARD.json")
    c.add_argument("--schema", help="target keypoint schema: a built-in name (shooter12) or a .toml file (docs/SCHEMAS.md)")
    c.add_argument("--no-doctor", action="store_true", help="skip the pre-flight integration check")
    q = c.add_argument_group("closed-loop quality validation (docs/QUALITY.md)")
    q.add_argument("--quality-model", help="ONNX model (YOLOv8-pose or D-FINE): validate every frame in memory")
    q.add_argument("--quality-format", choices=["yolov8_pose", "dfine", "table", "apollo"])
    q.add_argument("--quality-device", choices=["cpu", "cuda", "tensorrt", "directml"])
    q.add_argument("--quality-sim", action="store_true", help="simulated detector (test double; peeks at the labels)")
    q.add_argument("--quality-runtime", action="store_true",
                   help="score frames through the production inference runtime (docs/RUNTIME.md), not the research path")
    q.add_argument("--quality-static", action="store_true", help="cheap gates only: no model")
    q.add_argument("--quality-inflight", type=int, help="frames validated while the engine renders the next ones")
    q.add_argument("--quality-every", type=int, help="run the model on every N-th frame")
    q.add_argument("--quality-policy", help="TOML with filtering thresholds, re-read while running (default: "
                                            "<out>/quality_policy.toml if present)")
    q.add_argument("--adaptive", action="store_true", help="feedback-driven domain randomization")
    c.set_defaults(fn=_cmd_collect)

    v = sub.add_parser("verify", help="verify a dataset and write qa_report")
    v.add_argument("dataset")
    v.add_argument("--no-images", action="store_true", help="skip image file checks")
    v.set_defaults(fn=_cmd_verify)

    rq = sub.add_parser("requalify", help="re-run new quality thresholds on a finished dataset (no pixels needed)")
    rq.add_argument("dataset")
    rq.add_argument("--policy", required=True, help="TOML with PolicyConfig keys (see docs/QUALITY.md)")
    rq.set_defaults(fn=_cmd_requalify)

    pv = sub.add_parser("preview", help="contact sheet with labels")
    pv.add_argument("dataset")
    pv.add_argument("--n", type=int, default=12)
    pv.add_argument("--seed", type=int, default=0)
    pv.add_argument("--out")
    pv.set_defaults(fn=_cmd_preview)

    m = sub.add_parser("merge", help="merge shard datasets")
    m.add_argument("out")
    m.add_argument("sources", nargs="+")
    m.set_defaults(fn=_cmd_merge)

    e = sub.add_parser("eval-image", help="run a model on one image and draw its detections (checks model + mapping)")
    e.add_argument("--model", required=True)
    e.add_argument("--image", required=True)
    e.add_argument("--format", default="yolov8_pose", choices=["yolov8_pose", "dfine", "table"])
    e.add_argument("--device", default="cpu")
    e.add_argument("--keypoint-map", default=None, help="identity (the model emits the schema) | coco17 (mapped by the schema file)")
    e.add_argument("--schema", help="target schema (shooter12 or a .toml); default: the 13-point human13")
    e.add_argument("--layout", help="table format: comma-separated columns, e.g. xyxy,score,class,kp:12,vis:12")
    e.add_argument("--coords", choices=["pixels", "normalized"])
    e.add_argument("--conf", type=float, default=0.25)
    e.add_argument("--out")
    e.set_defaults(fn=_cmd_eval_image)

    i = sub.add_parser("install", help="install the in-game mod")
    i.add_argument("--game", required=True, choices=["gmod", "valheim", "rust"])
    i.add_argument("--dir", required=True, help="the game's folder")
    i.set_defaults(fn=_cmd_install)

    s = sub.add_parser("serve-mock", help="fake game that speaks the protocol")
    s.add_argument("--mailbox", required=True)
    s.add_argument("--option", action="append", help="e.g. flip_probe_y, unit_scale=100, swap_lr, engine_images=false")
    s.set_defaults(fn=_cmd_serve_mock)

    from .detector.cli import register as register_detector
    register_detector(sub)
    from .runtime.cli import register as register_runtime
    register_runtime(sub)
    from .bioprofile.cli import register as register_bioprofile
    register_bioprofile(sub)
    from .assist.cli import register as register_assist
    register_assist(sub)
    from .bridge.cli import register as register_bridge
    register_bridge(sub)
    from .video.cli import register as register_video
    register_video(sub)
    from .ui.cli import register as register_ui
    register_ui(sub)
    from .ctl.cli import register as register_ctl
    register_ctl(sub)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        code = args.fn(args)
    except (ProfileError, AdapterError, ValueError) as e:
        print(f"error: {e}")
        code = EXIT_USAGE
    sys.exit(code)


if __name__ == "__main__":
    main()
