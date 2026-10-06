"""Simulation of the capture side: a synthetic screen (rectangles at known places, like UI elements), the packing a bridge chip would send
(RGB or 4:2:2), the preparation pipeline, the frame timeline, and delivery to the existing runtime through its FrameSource interface.

Nothing here detects anything. The detector, and what it is allowed to look for, is a separate decision (docs/VIDEO.md, section 0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..runtime.frames import QueueSource
from .prep import BT709, FramePrep, Geometry
from .timing import Roi, VideoMode, data_age_us


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int
    color: tuple[int, int, int]

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


class SyntheticScreen:
    def __init__(self, w: int, h: int, rects: list[Rect], background: tuple[int, int, int] = (24, 28, 36)) -> None:
        self.w, self.h, self.rects, self.bg = w, h, rects, background

    def render(self) -> np.ndarray:
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = self.bg
        for r in self.rects:
            img[max(r.y, 0) : r.y + r.h, max(r.x, 0) : r.x + r.w] = r.color
        return img


def rgb_to_packed422(rgb: np.ndarray, fmt: str = "uyvy", limited: bool = True, matrix: int = BT709) -> np.ndarray:
    """What an HDMI-to-CSI bridge sends in YUV 4:2:2 mode: chroma averaged over each horizontal pair. Returns (H, 2W) uint8."""
    f = rgb.astype(np.float64)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    kr, kb = (0.2126, 0.0722) if matrix == BT709 else (0.299, 0.114)
    y = kr * r + (1 - kr - kb) * g + kb * b
    u, v = (b - y) / (2 * (1 - kb)), (r - y) / (2 * (1 - kr))
    if limited:
        y, u, v = 16 + y * 219 / 255, 128 + u * 224 / 255, 128 + v * 224 / 255
    else:
        u, v = 128 + u, 128 + v
    h, w = y.shape
    uu = (u[:, 0::2] + u[:, 1::2]) / 2
    vv = (v[:, 0::2] + v[:, 1::2]) / 2
    out = np.empty((h, w // 2, 4), np.uint8)
    q = lambda a: np.clip(np.round(a), 0, 255).astype(np.uint8)  # noqa: E731
    if fmt == "uyvy":
        out[..., 0], out[..., 1], out[..., 2], out[..., 3] = q(uu), q(y[:, 0::2]), q(vv), q(y[:, 1::2])
    else:
        out[..., 0], out[..., 1], out[..., 2], out[..., 3] = q(y[:, 0::2]), q(uu), q(y[:, 1::2]), q(vv)
    return out.reshape(h, w * 2)


class VideoPath:
    """Capture side of one display mode: prepares frames and hands them to a FrameSource the runtime reads.

    `ts_us` of a delivered frame is the time the MIDDLE of the region of interest was on the wire (scan-out), not the time the frame
    became ready: that is the instant the picture content belongs to, and the one a scene timestamp for the bridge needs
    (`display_latency_us` adds the monitor's own pixel latency if it is known)."""

    def __init__(
        self,
        mode: VideoMode,
        fmt: str = "rgb24",
        crop: Optional[tuple[int, int, int, int]] = None,
        out: int = 640,
        limited: bool = False,
        matrix: int = BT709,
        capacity: int = 4,
        display_latency_us: int = 0,
    ) -> None:
        self.mode, self.fmt = mode, fmt
        self.prep = FramePrep(mode.h_active, mode.v_active, fmt, crop, out, limited, matrix)
        self.roi = Roi(*(crop or (0, 0, mode.h_active, mode.v_active)))
        self.source = QueueSource(capacity)
        self.display_latency_us = display_latency_us
        self.n = 0
        self.tail_ns: list[int] = []

    @property
    def geometry(self) -> Geometry:
        return self.prep.geometry

    def push(self, frame: np.ndarray, t_vsync_us: int) -> bool:
        """One frame whose active video started at `t_vsync_us` (monotonic microseconds). Never blocks: a full queue drops the frame."""
        out = np.empty((self.prep.cfg.out_h, self.prep.cfg.out_w, 3), np.uint8)
        _, total_ns, last_ns = self.prep.frame(frame, out)
        self.tail_ns.append(last_ns)
        ages = data_age_us(self.mode, self.roi, tail_us=last_ns / 1000.0)
        ts = int(t_vsync_us + ages["mid_row_us"] + self.display_latency_us)
        meta = {
            "geometry": self.geometry,
            "t_vsync_us": t_vsync_us,
            "ready_us": t_vsync_us + ages["ready_us"],
            "age_of_content_us": ages["age_of_content_us"],
            "mode": self.mode.name,
        }
        ok = self.source.push(out, self.n, ts, meta)
        self.n += 1
        return ok
