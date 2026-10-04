"""Post-processing of the network's dense outputs into a `KeypointArray`: a C implementation (compiled on demand, releases the GIL,
fills the struct directly) and the numpy reference it is tested against. `make_postprocessor("auto")` picks C when a compiler is
available and falls back to numpy otherwise, reporting which one is in use."""
from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..detector import structs
from ..detector.layout import DecodeConfig, HeadLayout
from ..detector.postprocess import decode_dense

log = logging.getLogger("dataopen.runtime")
CSRC = Path(__file__).resolve().parents[1] / "detector" / "csrc"


class _CCfg(ctypes.Structure):
    _fields_ = [("n_cls", ctypes.c_int), ("n_kpt", ctypes.c_int), ("n_levels", ctypes.c_int), ("strides", ctypes.c_int * 4),
                ("offset_scale", ctypes.c_float), ("refine_stride", ctypes.c_int), ("n_primary", ctypes.c_int),
                ("primary", ctypes.c_int * 4), ("max_det", ctypes.c_int), ("conf_thr", ctypes.c_float),
                ("refine_radius_px", ctypes.c_float)]


def build_c_library(out_dir: Optional[str | Path] = None, cc: Optional[str] = None) -> Path:
    """Compile detector/csrc/apollo_post.c into a shared library (cached by source hash). Raises RuntimeError without a compiler."""
    compiler = cc or os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        raise RuntimeError("no C compiler found (set CC, or install gcc/clang); the numpy post-processor is used instead")
    src = CSRC / "apollo_post.c"
    h = hashlib.blake2b(src.read_bytes() + (CSRC / "apollo_detection.h").read_bytes(), digest_size=6).hexdigest()
    out = Path(out_dir or os.environ.get("APOLLO_POST_DIR") or Path(tempfile.gettempdir()) / "dataopen-post")
    out.mkdir(parents=True, exist_ok=True)
    so = out / f"apollo_post_{h}.so"
    if not so.exists():
        tmp = out / f"{so.name}.{os.getpid()}.tmp"
        r = subprocess.run([compiler, "-O2", "-std=c99", "-shared", "-fPIC", f"-I{CSRC}", "-o", str(tmp), str(src), "-lm"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compiling apollo_post.c failed:\n{r.stderr[:800]}")
        tmp.replace(so)
    return so


class PostProcessor:
    name = "post"

    def decode(self, levels: Sequence[np.ndarray], refine: Optional[np.ndarray]):
        """-> filled `KeypointArray` (detections in model-input pixels; the caller sets frame_id / timestamps / flags)."""
        raise NotImplementedError


class NumpyPostProcessor(PostProcessor):
    name = "numpy"

    def __init__(self, layout: HeadLayout, cfg: DecodeConfig) -> None:
        self.lay, self.cfg = layout, cfg

    def decode(self, levels, refine):
        dets = decode_dense([lv for lv in levels], self.lay, self.cfg, refine)
        return structs.pack(dets, 0, 0, 0, keypoints=self.lay.keypoints, classes=self.lay.classes, n_kpt=self.lay.n_kpt)


class CPostProcessor(PostProcessor):
    name = "c"

    def __init__(self, layout: HeadLayout, cfg: DecodeConfig, library: Optional[str | Path] = None) -> None:
        if layout.n_kpt != 12:
            raise RuntimeError("the C post-processor is built for 12 keypoints")
        self.lib = ctypes.CDLL(str(library or build_c_library()))
        self.lib.apollo_decode.restype = ctypes.c_int
        self.lay = layout
        prim = list(layout.primary)[:4] + [0] * (4 - len(layout.primary[:4]))
        self.cfg_c = _CCfg(layout.n_cls, layout.n_kpt, len(layout.strides), (ctypes.c_int * 4)(*(list(layout.strides) + [0] * 4)[:4]),
                           layout.offset_scale, layout.refine_stride, len(layout.primary), (ctypes.c_int * 4)(*prim),
                           min(cfg.max_det, layout.max_det), cfg.conf_thr, cfg.refine_radius_px)
        self.use_refine = cfg.refine
        self.hash = structs.schema_hash(layout.keypoints, layout.classes) if layout.keypoints else 0

    def decode(self, levels, refine):
        n = len(levels)
        arrs = [np.ascontiguousarray(lv, dtype=np.float32) for lv in levels]
        ptrs = (ctypes.POINTER(ctypes.c_float) * n)(*[a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in arrs])
        h = (ctypes.c_int * n)(*[a.shape[-2] for a in arrs])
        w = (ctypes.c_int * n)(*[a.shape[-1] for a in arrs])
        out = structs.KeypointArray()
        ref, rh, rw = None, 0, 0
        if refine is not None and self.use_refine and self.lay.refine_stride:
            ref = np.ascontiguousarray(refine, dtype=np.float32)
            rh, rw = ref.shape[-2], ref.shape[-1]
        rptr = ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if ref is not None else None
        got = self.lib.apollo_decode(ctypes.byref(self.cfg_c), ptrs, h, w, rptr, rh, rw, ctypes.byref(out))
        if got < 0:
            raise RuntimeError("apollo_decode rejected its arguments")
        out.schema_hash = self.hash
        return out


def make_postprocessor(layout: HeadLayout, cfg: Optional[DecodeConfig] = None, mode: str = "auto") -> PostProcessor:
    cfg = cfg or DecodeConfig(max_det=layout.max_det)
    if mode not in ("auto", "c", "numpy"):
        raise ValueError("post must be 'auto', 'c' or 'numpy'")
    if mode != "numpy":
        try:
            return CPostProcessor(layout, cfg)
        except (RuntimeError, OSError) as e:
            if mode == "c":
                raise
            log.warning("C post-processor unavailable (%s): using numpy (slower)", e)
    return NumpyPostProcessor(layout, cfg)
