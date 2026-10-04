"""`dataopen preview`: contact sheet of random frames with labels drawn on top, for eyeballing."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .card import schema_for_dir
from .export import CanonicalStore
from .imageio import read_image, write_png
from .models import FrameKind, FrameRecord
from .schema import HUMAN_13, SkeletonSchema
from .viz import contact_sheet, draw_annotations


def make_preview(root: Path, n: int = 12, seed: int = 0, out: Optional[Path] = None, split: Optional[str] = None,
                 cols: int = 4, tile_width: int = 480, include_negatives: bool = True,
                 schema: Optional[SkeletonSchema] = None) -> Path:
    root = Path(root)
    schema = schema or schema_for_dir(root, HUMAN_13)
    store = CanonicalStore(root)
    recs: list[FrameRecord] = []
    for s in store.splits():
        if split in (None, s):
            recs += list(store.load(s))
    if not recs:
        raise ValueError(f"no records found in {root}")
    rng = np.random.default_rng(seed)
    pos = [r for r in recs if r.kind is FrameKind.POSITIVE]
    neg = [r for r in recs if r.kind is FrameKind.NEGATIVE]
    pick = list(rng.permutation(len(pos))[: max(1, n - (min(2, len(neg)) if include_negatives else 0))])
    chosen = [pos[i] for i in pick]
    if include_negatives and neg:
        chosen += [neg[i] for i in rng.permutation(len(neg))[:2]]
    tiles = [draw_annotations(read_image(root / r.file_name), r.annotations, schema) for r in chosen]
    out = Path(out) if out else root / "preview.png"
    write_png(out, contact_sheet(tiles, cols=cols, tile_width=tile_width))
    return out
