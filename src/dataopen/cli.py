"""python -m dataopen.cli --adapter mock --frames 500 --out out/run1"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .core.orchestrator import DatasetOrchestrator, SessionConfig


def _make_adapter(name: str):
    if name == "mock":
        from .adapters.mock import MockGameAdapter
        return MockGameAdapter()
    raise SystemExit(f"unknown adapter {name!r}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Collect a synthetic human-pose dataset")
    p.add_argument("--adapter", default="mock")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--frames-per-scene", type=int, default=25)
    p.add_argument("--negative-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = SessionConfig(a.out, a.seed, a.frames, a.frames_per_scene, a.negative_ratio, resume=a.resume)
    rep = DatasetOrchestrator(_make_adapter(a.adapter), cfg).run()
    print(json.dumps({"accepted": rep.accepted, "attempts": rep.attempts, "fps": round(rep.fps, 1),
                      "rejects": rep.rejects, "stages_s": rep.stage_seconds, "stop": rep.stop_reason}, indent=2))


if __name__ == "__main__":
    main()
