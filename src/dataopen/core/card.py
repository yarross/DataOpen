"""DATASET_CARD.json: provenance and reproducibility record written next to every dataset."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from .export import CanonicalStore
from .schema import SkeletonSchema


def config_hash(cfg_dict: dict[str, Any]) -> str:
    blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_dataset_card(root: Path, adapter, cfg, rep, schema: SkeletonSchema, store: CanonicalStore) -> dict[str, Any]:
    per_split: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    persons = 0
    for split in store.splits():
        for r in store.load(split):
            per_split[split] += 1
            kinds[r.kind.value] += 1
            persons += len(r.annotations)
    hello = getattr(adapter, "hello", None) or {}
    try:
        info = adapter.info
        game = {"name": info.name, "engine": info.engine, "image_size": list(info.image_size),
                "capture_schema": info.schema.name,
                "capabilities": sorted(c.value for c in info.capabilities)}
    except Exception:  # adapter never connected
        game = {}
    game.update({"game_version": hello.get("game_version"), "mod_version": hello.get("mod_version")})
    cfg_dict = asdict(cfg)
    card = {
        "dataopen_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game": game,
        "skeleton": {"name": schema.name, "keypoints": list(schema.keypoints), "flip_idx": schema.flip_idx(),
                     "edges": [list(e) for e in schema.edges], "flip_pairs": [list(p) for p in schema.flip_pairs],
                     "sigmas": list(schema.oks_sigmas()), "weights": list(schema.oks_weights()),
                     "derived": list(schema.derived), "primary": list(schema.primary),
                     "groups": {g: list(m) for g, m in schema.groups}, "roles": dict(schema.roles),
                     "classes": list(schema.classes), "class_key": schema.class_key},
        "visibility_flags": {"0": "outside frame / behind camera / bone missing", "1": "in frame but occluded",
                             "2": "visible"},
        "session": {**cfg_dict, "config_hash": config_hash(cfg_dict)},
        "counts": {"frames": sum(per_split.values()), "persons": persons, "by_split": dict(per_split),
                   "by_kind": dict(kinds), "attempts": rep.attempts, "scenes": rep.scenes,
                   "rejects": rep.rejects, "stop_reason": rep.stop_reason},
        "provenance": cfg.provenance or {
            "assets": "UNSPECIFIED: fill in the origin and licence of the game assets before distributing this "
                      "dataset (--provenance-note / profile [provenance])"},
    }
    (root / "DATASET_CARD.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    return card


def schema_from_card_dict(sk: dict[str, Any]) -> SkeletonSchema:
    return SkeletonSchema(sk["name"], tuple(sk["keypoints"]), tuple(tuple(e) for e in sk.get("edges", [])),
                          tuple(tuple(p) for p in sk.get("flip_pairs", [])), tuple(sk.get("sigmas", ())),
                          tuple(sk.get("weights", ())), tuple(sk.get("derived", ())), tuple(sk.get("primary", ())),
                          tuple((g, tuple(m)) for g, m in sk.get("groups", {}).items()),
                          tuple(sk.get("roles", {}).items()), tuple(sk.get("classes", ["person"])),
                          sk.get("class_key", ""))


def schema_for_dir(root: Path, default: SkeletonSchema) -> SkeletonSchema:
    """The skeleton a dataset directory was labeled with (its card, or its parent's: the quarantine is a sub-dataset)."""
    for d in (Path(root), Path(root).parent):
        card = d / "DATASET_CARD.json"
        if card.exists():
            try:
                return schema_from_card_dict(json.loads(card.read_text())["skeleton"])
            except (KeyError, ValueError):
                break
    return default
