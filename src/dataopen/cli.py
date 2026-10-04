"""dataopen: command line.

  dataopen games                                   list game profiles
  dataopen doctor  --game gmod [--mailbox DIR]     check a game integration before collecting
  dataopen collect --game gmod --out DIR --frames N
  dataopen verify  DIR                             check a collected dataset, write qa_report.{json,md}
  dataopen preview DIR                             contact sheet with labels drawn on top
  dataopen merge   OUT SRC1 SRC2 ...               merge shards
  dataopen serve-mock --mailbox DIR                a fake game that speaks the protocol (try everything offline)
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
    rep = run_doctor(adapter, Path(a.out), frames=a.frames, tol_px=a.tolerance, seed=a.seed)
    print(rep.render())
    print(f"\nOverlays and doctor_report.json: {Path(a.out) / 'doctor'}")
    return EXIT_OK if rep.ok else EXIT_FAILED_CHECK


def _cmd_collect(a) -> int:
    prof = load_profile(a.game)
    out = Path(a.out)
    adapter = build_adapter(prof, a.mailbox, connect=False)
    if not a.no_doctor:
        tmp = Path(tempfile.mkdtemp(prefix="dataopen-doctor-"))
        rep = run_doctor(adapter, tmp, frames=3, seed=a.seed)
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
    orch = DatasetOrchestrator(adapter, cfg)
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
    c.add_argument("--no-doctor", action="store_true", help="skip the pre-flight integration check")
    c.set_defaults(fn=_cmd_collect)

    v = sub.add_parser("verify", help="verify a dataset and write qa_report")
    v.add_argument("dataset")
    v.add_argument("--no-images", action="store_true", help="skip image file checks")
    v.set_defaults(fn=_cmd_verify)

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

    s = sub.add_parser("serve-mock", help="fake game that speaks the protocol")
    s.add_argument("--mailbox", required=True)
    s.add_argument("--option", action="append", help="e.g. flip_probe_y, unit_scale=100, swap_lr, engine_images=false")
    s.set_defaults(fn=_cmd_serve_mock)
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
