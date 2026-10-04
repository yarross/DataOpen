"""Merge datasets collected by parallel shards into one (`dataopen merge`)."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .export import CanonicalStore, write_coco, write_yolo_label, write_yolo_yaml
from .card import schema_from_card_dict


class MergeError(RuntimeError):
    pass


def merge_datasets(sources: Sequence[Path], out: Path) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise MergeError(f"{out} is not empty")
    if len(sources) < 2:
        raise MergeError("need at least two source datasets")
    cards = []
    for s in sources:
        c = Path(s) / "DATASET_CARD.json"
        if not c.exists():
            raise MergeError(f"{s} has no DATASET_CARD.json (not a DataOpen dataset?)")
        cards.append(json.loads(c.read_text()))
    first = cards[0]
    for s, c in zip(sources[1:], cards[1:]):
        if (c["skeleton"]["keypoints"] != first["skeleton"]["keypoints"]
                or c["skeleton"].get("classes", ["person"]) != first["skeleton"].get("classes", ["person"])):
            raise MergeError(f"{s} uses a different skeleton than {sources[0]}")

    store = CanonicalStore(out)
    seen: dict[str, Path] = {}
    per_split: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    persons = 0
    for src in map(Path, sources):
        src_store = CanonicalStore(src)
        for split in src_store.splits():
            batch = list(src_store.load(split))
            for r in batch:
                if r.frame_id in seen:
                    raise MergeError(f"frame {r.frame_id} exists in both {seen[r.frame_id]} and {src}: shards must "
                                     f"use distinct --shard i/n values (same seed and different scenes)")
                seen[r.frame_id] = src
                dst = out / r.file_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / r.file_name, dst)
                write_yolo_label(r, out / "labels")
                per_split[split] += 1
                kinds[r.kind.value] += 1
                persons += len(r.annotations)
            store.append(split, batch)

    sk = first["skeleton"]
    schema = schema_from_card_dict(sk)
    for split in store.splits():
        write_coco(store.load(split), schema, out / "annotations" / f"coco_{split}.json")
    write_yolo_yaml(out, schema, store.splits())
    card = dict(first)
    card["counts"] = {**first.get("counts", {}), "frames": sum(per_split.values()), "persons": persons,
                      "by_split": dict(per_split), "by_kind": dict(kinds)}
    card["merged_from"] = [str(s) for s in sources]
    card["merged_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "DATASET_CARD.json").write_text(json.dumps(card, indent=2, default=str))
    return card
