"""Python side of the frame preparation: the C99 streaming core through ctypes, the geometry that maps detector coordinates back to screen
pixels, and a numpy reference the C core is held to."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

CSRC = Path(__file__).resolve().parent / "csrc"
SOURCES = [CSRC / "video_prep.c", CSRC / "video_prep_host.c"]
FMT = {"rgb24": 0, "bgr24": 1, "uyvy": 2, "yuyv": 3}
BT709, BT601 = 0, 1
PAD = 114


class VpCfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint16) for n in ("src_w", "src_h", "crop_x", "crop_y", "crop_w", "crop_h", "out_w", "out_h")] + [
        (n, ctypes.c_uint8) for n in ("fmt", "limited", "matrix", "pad")
    ]


def build_library(out_dir: Optional[str | Path] = None, cc: Optional[str] = None) -> Path:
    compiler = cc or os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        raise RuntimeError("no C compiler found (set CC, or install gcc/clang)")
    h = hashlib.blake2b(b"".join(p.read_bytes() for p in (*SOURCES, CSRC / "video_prep.h")), digest_size=6).hexdigest()
    out = Path(out_dir or Path(tempfile.gettempdir()) / "dataopen-video")
    out.mkdir(parents=True, exist_ok=True)
    so = out / f"video_prep_{h}.so"
    if not so.exists():
        tmp = out / f"{so.name}.{os.getpid()}.tmp"
        r = subprocess.run(
            [compiler, "-O2", "-std=gnu99", "-Wall", "-Wextra", "-shared", "-fPIC", f"-I{CSRC}", "-o", str(tmp), *map(str, SOURCES)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"compiling the video prep core failed:\n{r.stderr[:800]}")
        tmp.replace(so)
    return so


_LIB: Optional[ctypes.CDLL] = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        c = ctypes
        L = c.CDLL(str(build_library()))
        u8p = c.POINTER(c.c_uint8)
        L.vp_sizeof.restype = c.c_size_t
        L.vp_init.argtypes = [c.c_void_p, c.POINTER(VpCfg)]
        L.vp_begin.argtypes = [c.c_void_p, u8p]
        L.vp_line.argtypes = [c.c_void_p, c.c_int, u8p]
        L.vp_complete.argtypes = [c.c_void_p]
        L.vp_last_row.argtypes = [c.c_void_p]
        L.vp_geometry.argtypes = [c.c_void_p, c.POINTER(c.c_int32)]
        L.vp_frame.argtypes = [c.c_void_p, u8p, c.c_uint32, u8p, c.POINTER(c.c_uint64), c.POINTER(c.c_uint64)]
        _LIB = L
    return _LIB


@dataclass(frozen=True)
class Geometry:
    """How a source frame sits in the detector's input: crop rectangle on the screen, scale, and where the scaled crop is placed."""

    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    content_w: int
    content_h: int
    pad_x: int
    pad_y: int
    out_w: int
    out_h: int

    @property
    def scale(self) -> float:
        return self.content_w / self.crop_w

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Detector input coordinates -> screen pixel coordinates."""
        return (
            self.crop_x + (x - self.pad_x) * self.crop_w / self.content_w,
            self.crop_y + (y - self.pad_y) * self.crop_h / self.content_h,
        )

    def to_input(self, sx: float, sy: float) -> tuple[float, float]:
        return (
            self.pad_x + (sx - self.crop_x) * self.content_w / self.crop_w,
            self.pad_y + (sy - self.crop_y) * self.content_h / self.crop_h,
        )


class FramePrep:
    """One configured preparation pipeline (one source mode, one crop, one output size). Not thread-safe: one stream, one owner."""

    def __init__(
        self,
        src_w: int,
        src_h: int,
        fmt: str = "rgb24",
        crop: Optional[tuple[int, int, int, int]] = None,
        out: int = 640,
        limited: bool = False,
        matrix: int = BT709,
    ) -> None:
        crop = crop or (0, 0, src_w, src_h)
        self.cfg = VpCfg(src_w, src_h, crop[0], crop[1], crop[2], crop[3], out, out, FMT[fmt], int(limited), matrix, PAD)
        self.fmt, self.bytes_per_px = fmt, 3 if fmt in ("rgb24", "bgr24") else 2
        self.L = lib()
        self.mem = ctypes.create_string_buffer(self.L.vp_sizeof())
        self.p = ctypes.cast(self.mem, ctypes.c_void_p)
        r = self.L.vp_init(self.p, ctypes.byref(self.cfg))
        if r != 0:
            raise ValueError("bad crop/size arguments" if r == -1 else "the crop would need upscaling")
        g = (ctypes.c_int32 * 8)()
        self.L.vp_geometry(self.p, g)
        self.geometry = Geometry(crop[0], crop[1], crop[2], crop[3], g[0], g[1], g[2], g[3], out, out)
        self.step = (g[4] / 65536, g[5] / 65536)

    @property
    def last_row(self) -> int:
        return self.L.vp_last_row(self.p)

    def frame(self, src: np.ndarray, out: Optional[np.ndarray] = None) -> tuple[np.ndarray, int, int]:
        """Whole frame; returns (output, ns total, ns of the last needed line). `src` is (H, W, 3) uint8 for rgb24/bgr24 or (H, 2W) uint8
        for the packed 4:2:2 formats."""
        src = np.ascontiguousarray(src)
        o = out if out is not None else np.empty((self.cfg.out_h, self.cfg.out_w, 3), np.uint8)
        assert o.flags.c_contiguous and o.dtype == np.uint8
        stride = src.strides[0]
        tot, last = ctypes.c_uint64(), ctypes.c_uint64()
        r = self.L.vp_frame(
            self.p,
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            stride,
            o.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(tot),
            ctypes.byref(last),
        )
        if r != 0:
            raise RuntimeError("frame incomplete")
        return o, tot.value, last.value

    # streaming interface (what the DMA/ISR side does)
    def begin(self, out: np.ndarray) -> None:
        self._out = out
        self.L.vp_begin(self.p, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

    def line(self, y: int, data: np.ndarray) -> None:
        self.L.vp_line(self.p, y, data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

    @property
    def complete(self) -> bool:
        return bool(self.L.vp_complete(self.p))


# ---------------------------------------------------------------- numpy reference
def yuv_to_rgb_ref(src: np.ndarray, fmt: str, limited: bool, matrix: int) -> np.ndarray:
    h = src.shape[0]
    w = src.shape[1] // 2
    p = src.reshape(h, w // 2, 4).astype(np.float64)
    if fmt == "uyvy":
        u, y0, v, y1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    else:
        y0, u, y1, v = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    y = np.empty((h, w))
    y[:, 0::2], y[:, 1::2] = y0, y1
    u, v = np.repeat(u, 2, axis=1) - 128, np.repeat(v, 2, axis=1) - 128
    if limited:
        ys = (y - 16) * 1.1644
        k = (1.7927, 0.2132, 0.5329, 2.1124) if matrix == BT709 else (1.5960, 0.3918, 0.8130, 2.0172)
    else:
        ys = y
        k = (1.5748, 0.1873, 0.4681, 1.8556) if matrix == BT709 else (1.4020, 0.3441, 0.7141, 1.7720)
    rgb = np.stack([ys + k[0] * v, ys - k[1] * u - k[2] * v, ys + k[3] * u], axis=-1)
    return np.clip(np.round(rgb), 0, 255)


def prep_reference(src: np.ndarray, cfg: FramePrep) -> np.ndarray:
    """Float64 reference: colour conversion, crop, exact area average over the same footprints, letterbox. The C core is within a couple of
    LSBs of this (it rounds to 8 bits after colour conversion, as hardware does)."""
    c = cfg.cfg
    if cfg.fmt in ("uyvy", "yuyv"):
        rgb = yuv_to_rgb_ref(src, cfg.fmt, bool(c.limited), c.matrix)
    else:
        rgb = src.astype(np.float64)
        if cfg.fmt == "bgr24":
            rgb = rgb[..., ::-1]
        if c.limited:
            rgb = np.clip(np.round((rgb - 16) * 255 / 219), 0, 255)
    crop = rgb[c.crop_y : c.crop_y + c.crop_h, c.crop_x : c.crop_x + c.crop_w]
    g = cfg.geometry
    sx, sy = c.crop_w / g.content_w, c.crop_h / g.content_h

    def weights(n_out: int, n_in: int, step: float) -> np.ndarray:
        w = np.zeros((n_out, n_in))
        for k in range(n_out):
            a, b = k * step, min((k + 1) * step, n_in)
            for i in range(int(a), min(int(np.ceil(b)), n_in)):
                w[k, i] = max(0.0, min(b, i + 1) - max(a, i))
        return w / w.sum(axis=1, keepdims=True)

    wx, wy = weights(g.content_w, c.crop_w, sx), weights(g.content_h, c.crop_h, sy)
    small = np.tensordot(np.tensordot(wy, crop, axes=(1, 0)), wx, axes=(1, 1)).transpose(0, 2, 1)  # (J, W, 3) x (K, W) -> (J, K, 3)
    out = np.full((c.out_h, c.out_w, 3), PAD, np.uint8)
    out[g.pad_y : g.pad_y + g.content_h, g.pad_x : g.pad_x + g.content_w] = np.clip(np.round(small), 0, 255).astype(np.uint8)
    return out
