"""From detector output to objects of interest for the ASC.

    detections (UI elements + the pointer, in detector-input pixels)
      -> screen pixels (video.prep.Geometry.to_screen)
      -> tracker: stable ids, smoothing, when an element APPEARED (the ASC's pre-reaction guard needs that)
      -> the pointer position and, per target, the nearest point of its box, relative to the pointer at capture time
      -> ObjectOfInterest (float ASC / ObjectProvider)  or  BridgeLink SceneObject (the HID bridge, with the capture time in bridge time).

The wall: `UiSceneBuilder` only accepts detections whose class is in the UI taxonomy and refuses a model that is not one."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from ..assist.types import ObjectOfInterest
from ..bridge.protocol import SceneObject
from ..video.prep import Geometry
from .taxonomy import NAMES, TARGET_NAMES, require_ui_layout


@dataclass(frozen=True)
class Det:
    """One detection in DETECTOR-INPUT pixels (the 640x640 letterboxed frame)."""

    cls: str
    conf: float
    box: tuple[float, float, float, float]


@dataclass
class Track:
    id: int
    cls: str
    box: tuple[float, float, float, float]  # smoothed, screen pixels
    conf: float
    first_seen_us: int
    hits: int = 1
    misses: int = 0
    appeared_us: Optional[int] = None  # None: it was there before the scene started to be watched


@dataclass(frozen=True)
class Target:
    id: int
    cls: str
    box: tuple[float, float, float, float]
    conf: float
    appeared_us: Optional[int]


@dataclass(frozen=True)
class Snapshot:
    t_capture_us: int
    cursor: Optional[tuple[float, float]]  # the pointer's hotspot, screen pixels
    targets: tuple[Target, ...]

    def nearest_point(self, t: Target, at: Optional[tuple[float, float]] = None) -> tuple[float, float]:
        cx, cy = at or self.cursor
        x0, y0, x1, y1 = t.box
        return min(max(cx, x0), x1), min(max(cy, y0), y1)

    def distance(self, t: Target, at: Optional[tuple[float, float]] = None) -> float:
        px, py = self.nearest_point(t, at)
        cx, cy = at or self.cursor
        return math.hypot(px - cx, py - cy)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class SceneConfig:
    conf_target: float = 0.6  # minimum confidence of an element to be offered as a target (measured trade-off: docs/UIDET.md)
    conf_cursor: float = 0.4
    min_side_px: float = 8.0  # smaller than this on the screen: not a plausible element
    match_iou: float = 0.3
    smooth: float = 0.5  # weight of the NEW box in the track (1 = no smoothing; lower = steadier, laggier)
    confirm_hits: int = 2  # frames an element must be seen before it is a target (kills one-frame false alarms)
    drop_misses: int = 6  # frames an element may be missing before the track is dropped
    warmup_frames: int = 5  # elements first seen during the first frames were "already there", not "appeared"
    radius_frac: float = 0.35  # ASC radius = this x the smaller side of the box
    radius_min_px: float = 4.0
    radius_max_px: float = 60.0
    max_scene: int = 5  # the bridge keeps at most 5 objects
    reach_px: float = 1200.0  # elements further than this from the pointer are not sent


class UiSceneBuilder:
    def __init__(
        self, geometry: Geometry, cfg: Optional[SceneConfig] = None, class_names: Iterable[str] = NAMES, n_keypoints: int = 0
    ) -> None:
        require_ui_layout(class_names, n_keypoints)  # a pose / people model is refused here
        self.g, self.cfg = geometry, cfg or SceneConfig()
        self.class_names = tuple(class_names)
        self.tracks: dict[int, Track] = {}
        self._next = 1
        self.frames = 0

    def set_geometry(self, g: Geometry) -> None:
        self.g = g
        self.tracks.clear()
        self.frames = 0

    def _to_screen(self, d: Det) -> tuple[float, float, float, float]:
        x0, y0 = self.g.to_screen(d.box[0], d.box[1])
        x1, y1 = self.g.to_screen(d.box[2], d.box[3])
        return (x0, y0, x1, y1)

    def update(self, t_capture_us: int, dets: Iterable[Det]) -> Snapshot:
        c = self.cfg
        self.frames += 1
        cursor_det, targets_in = None, []
        for d in dets:
            if d.cls not in self.class_names:
                raise ValueError(f"detection of class {d.cls!r}: not a class of this model")
            b = self._to_screen(d)
            if d.cls == "cursor":
                if d.conf >= c.conf_cursor and (cursor_det is None or d.conf > cursor_det[0]):
                    cursor_det = (d.conf, b)
            elif d.cls in TARGET_NAMES and d.conf >= c.conf_target and min(b[2] - b[0], b[3] - b[1]) >= c.min_side_px:
                targets_in.append((d.cls, d.conf, b))
        # greedy IoU matching within a class, best confidence first
        for t in self.tracks.values():
            t.misses += 1
        for cls, conf, b in sorted(targets_in, key=lambda x: -x[1]):
            best, best_iou = None, c.match_iou
            for t in self.tracks.values():
                if t.cls == cls and t.misses > 0:  # not yet matched in this frame (all were counted as missed above)
                    v = iou(t.box, b)
                    if v > best_iou:
                        best, best_iou = t, v
            if best is not None:
                s = c.smooth
                best.box = tuple(s * n + (1 - s) * o for n, o in zip(b, best.box))
                best.conf, best.misses, best.hits = conf, 0, best.hits + 1
            else:
                appeared = t_capture_us if self.frames > c.warmup_frames else None
                self.tracks[self._next] = Track(self._next, cls, b, conf, t_capture_us, 1, 0, appeared)
                self._next += 1
        for k in [k for k, t in self.tracks.items() if t.misses > c.drop_misses]:
            del self.tracks[k]
        cur = None
        if cursor_det is not None:
            b = cursor_det[1]
            cur = (b[0], b[1])  # the arrow's hotspot is the top-left of its box
        tg = tuple(
            Target(t.id, t.cls, t.box, t.conf, t.appeared_us) for t in self.tracks.values() if t.hits >= c.confirm_hits and t.misses <= 1
        )
        return Snapshot(t_capture_us, cur, tg)

    # ---- ASC (float, absolute screen pixels)
    def radius(self, t: Target) -> float:
        c = self.cfg
        return min(c.radius_max_px, max(c.radius_min_px, c.radius_frac * min(t.box[2] - t.box[0], t.box[3] - t.box[1])))

    def objects(self, snap: Snapshot, at: Optional[tuple[float, float]] = None) -> list[ObjectOfInterest]:
        """Objects of interest nearest first, in absolute screen pixels. Each target is represented by the point of its box nearest to the
        pointer and a radius proportional to its size: a wide field is 'reached' at its edge, not at its centre."""
        cur = at or snap.cursor
        if cur is None:
            return []
        out = []
        for t in sorted(snap.targets, key=lambda t: snap.distance(t, cur)):
            if snap.distance(t, cur) > self.cfg.reach_px:
                continue
            px, py = snap.nearest_point(t, cur)
            out.append(ObjectOfInterest(t.id, px, py, self.radius(t), t.appeared_us))
        return out

    def nearest(self, snap: Snapshot, x: float, y: float) -> Optional[ObjectOfInterest]:
        o = self.objects(snap, (x, y))
        return o[0] if o else None

    # ---- the HID bridge (relative to the pointer at capture time)
    def bridge_scene(self, snap: Snapshot, to_bridge_time=lambda t: t) -> tuple[int, list[SceneObject]]:
        """(t_capture in bridge time, up to 5 objects relative to the pointer). No pointer -> no scene: the bridge then gives no help (K = 1)."""  # noqa: E501
        if snap.cursor is None:
            return to_bridge_time(snap.t_capture_us), []
        objs = []
        cx, cy = snap.cursor
        for o in self.objects(snap)[: self.cfg.max_scene]:
            ta = None if o.t_appear_us is None else int(to_bridge_time(o.t_appear_us))
            objs.append(SceneObject(o.id, o.x - cx, o.y - cy, o.radius, ta))
        return int(to_bridge_time(snap.t_capture_us)), objs


class BridgeScenePublisher:
    """`on_snapshot` for `UiService`: sends the scene to the HID bridge as a BridgeLink SCENE frame (`send` writes one frame to the link).
    The capture time is converted with `clock` (a `ClockSync` fed by TSYNC replies) when the module's clock is not the bridge's.

    What it does NOT send, on purpose (docs/LATENCY.md):
      * a scene that is late (`health`, ui/health.py): the bridge would use it at 80-100 ms of age, or not at all, and an age that wanders
        around its 100 ms TTL makes the help blink. While the scene help is off nothing is sent; the bridge lets its last scene expire.
    What it sends that the old version did not:
      * ONE empty scene when the targets are gone (after `empty_after` empty snapshots in a row) or when the help is switched off: the
        bridge then drops the old scene now, not up to 100 ms later (a one-frame phantom lives one cycle, not the whole TTL).
    `gate=False` (and no `health`): no gating at all (a test of the bare publisher)."""

    def __init__(self, builder: UiSceneBuilder, send, clock=None, health=None, now_us=None, gate: bool = True,
                 empty_after: int = 2) -> None:
        from ..runtime.frames import now_us as _now
        from .health import SceneHealth

        self.builder, self.send, self.clock = builder, send, clock
        self.health = health if health is not None else (SceneHealth() if gate else None)
        self.now_us = now_us or _now
        self.empty_after = empty_after
        self.sent = self.retracted = 0
        self._had = False
        self._empty_run = 0

    def _send_empty(self, tcap: int) -> None:
        from ..bridge.protocol import scene_frame

        self.send(scene_frame(tcap, []))
        self.retracted += 1
        self._had = False

    def __call__(self, snap: Snapshot, frame=None) -> None:
        to_bridge = self.clock.to_bridge if self.clock is not None else (lambda t: t)
        tcap, objs = self.builder.bridge_scene(snap, to_bridge)
        if self.health is not None:
            now = self.now_us()
            ok = self.health.scene((now - snap.t_capture_us) / 1000.0, now / 1000.0)
            if self.health.take_announcement():          # the help has just been switched off: end it now
                if self._had:
                    self._send_empty(tcap)
                return
            if not ok:
                return
        if objs:
            from ..bridge.protocol import scene_frame

            self.send(scene_frame(tcap, objs))
            self.sent += 1
            self._had, self._empty_run = True, 0
        elif self._had:
            self._empty_run += 1
            if self._empty_run >= self.empty_after:
                self._send_empty(tcap)
