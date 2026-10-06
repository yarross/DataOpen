"""Closed-loop check of the whole assistive chain with the UI scene in it: the detector is replaced by a stand-in with a controllable error
model (box jitter, misses, one-frame false alarms, latency, frame rate), the scene goes through the tracker and the BridgeLink SCENE frames
into the real bridge core, and the simulated person reaches for a button. Compare with the ideal scene (the object known exactly, every tick)."""  # noqa: E501

from __future__ import annotations

import numpy as np

from ..assist.types import ObjectOfInterest
from ..bridge import protocol as P
from ..bridge.sim import BridgeAssist
from ..video.prep import Geometry
from .scene import Det, SceneConfig, UiSceneBuilder
from .taxonomy import NAMES

GEOM_1080P = Geometry(0, 0, 1920, 1080, 640, 360, 0, 140, 640, 640)


class StandInDetector:
    """What a trained detector would hand over, with an error model. Boxes are given in SCREEN pixels and returned in detector-input pixels."""  # noqa: E501

    def __init__(self, geometry: Geometry, jitter_px: float = 1.5, miss: float = 0.05, false_alarm: float = 0.1, seed: int = 0) -> None:
        self.g, self.jitter, self.miss, self.fa = geometry, jitter_px, miss, false_alarm
        self.rng = np.random.default_rng(seed)

    def detect(self, targets: list[tuple[str, tuple]], cursor_xy: tuple[float, float]) -> list[Det]:
        out = []
        j = self.jitter
        for cls, box in targets + [("cursor", (cursor_xy[0], cursor_xy[1], cursor_xy[0] + 11, cursor_xy[1] + 19))]:
            if cls != "cursor" and self.rng.random() < self.miss:
                continue
            b = np.asarray(box, float) + self.rng.normal(0, j, 4)
            x0, y0 = self.g.to_input(b[0], b[1])
            x1, y1 = self.g.to_input(b[2], b[3])
            out.append(Det(cls, float(np.clip(self.rng.normal(0.85, 0.05), 0.5, 0.99)), (x0, y0, x1, y1)))
        if self.rng.random() < self.fa:  # a one-frame false alarm somewhere
            x, y = self.rng.uniform(0, 1700), self.rng.uniform(0, 900)
            a = self.g.to_input(x, y)
            b = self.g.to_input(x + 90, y + 40)
            out.append(Det(str(self.rng.choice(NAMES[:7])), 0.55, (*a, *b)))
        return out


class UiBridgeAssist(BridgeAssist):
    """`tick(t, dx, dy, px, py, obj)` as the person simulator calls it, but `obj` is only the ground truth the stand-in detector looks at:
    the bridge gets what the detector + tracker made of it, `fps` times a second, `latency_ms` after it was captured."""

    def __init__(
        self,
        asc,
        tremor,
        target_box,
        target_cls="button",
        fps: float = 60.0,
        latency_ms: float = 20.0,
        detector: StandInDetector | None = None,
        scene_cfg: SceneConfig | None = None,
        **kw,
    ) -> None:
        super().__init__(asc, tremor, **kw)
        self.builder = UiSceneBuilder(GEOM_1080P, scene_cfg or SceneConfig(warmup_frames=0))
        self.det = detector or StandInDetector(GEOM_1080P)
        self.target = (target_cls, tuple(target_box))
        self.period_us, self.lat_us = int(1e6 / fps), int(latency_ms * 1000)
        self.next_cap = 0
        self.hist: dict[int, tuple[float, float]] = {}
        self.scenes_sent = 0

    def tick(self, t_us: int, dx: int, dy: int, px: float, py: float, obj):
        t = t_us + self.t0
        if self.last_ms is None:
            self.last_ms = t
        while self.last_ms + 1000 < t:
            self.last_ms += 1000
            self.b.poll(self.last_ms)
        self.last_ms = t
        self.hist[t_us // 1000] = (px, py)
        if t >= self.next_keepalive:
            self.next_keepalive = t + 100_000
            self._link(t, [P.Frame(P.LK_HELLO)])
            if t % 1_000_000 < 100_000:
                self._link(t, self.rig.module.param_frames())
        self.b.poll(t)
        if t_us >= self.next_cap + self.lat_us:  # a frame captured `latency` ago has just been analysed
            cap_us = max(0, t_us - self.lat_us)
            cx, cy = self.hist.get(cap_us // 1000, (px, py))
            self.next_cap = t_us - self.lat_us + self.period_us
            snap = self.builder.update(cap_us + self.t0, self.det.detect([self.target], (cx, cy)))
            tcap, objs = self.builder.bridge_scene(snap)
            if objs:
                self._link(t, [P.scene_frame(tcap, objs)])
                self.scenes_sent += 1
        ox, oy = 0, 0
        if dx or dy:
            out, _ = self.b.mouse_in(t, 0x81, self.mouse.pack(0, dx, dy))
            ox, oy = self.mouse_xy(out)
        s = self.b.status(t)
        from ..bridge.sim import _Out

        return _Out(ox, oy, s.k_q16 / 65536)


def ideal_object(box) -> ObjectOfInterest:
    x0, y0, x1, y1 = box
    return ObjectOfInterest(1, (x0 + x1) / 2, (y0 + y1) / 2, 0.35 * min(x1 - x0, y1 - y0), t_appear_us=0)
