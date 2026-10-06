"""Synthetic screens with labelled interface elements, for training and evaluating the assistive UI detector.

A screen is composed from a scene (application window, dialog, game-style menu, desktop, settings page, web-style page, pop-up menu) drawn
in a theme. Themes come in two families: TRAIN families (light, dark, high contrast, colourful, neon game) and HELD-OUT families (retro bevel,
soft glass) that training never sees, so the evaluation can say how much of the skill survives a look the model has not met.

What this does not do: it is not the real world. Real screenshots (and real games) are the actual test; the numbers it gives are a floor for
how well the pipeline works, not a promise about any application. Text is Latin only (the bundled font)."""  # noqa: E501

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .taxonomy import ID

WORDS = (
    "File Edit View Help Open Save Close Cancel OK Apply Next Back Search Settings Options Play Pause Stop Start Exit Quit Load New Delete Add Remove "  # noqa: E501
    "Yes No Continue Retry Send Share Download Upload Login Logout Profile Home Menu Inventory Map Quests Skills Shop Audio Video Controls Language "  # noqa: E501
    "Graphics Resolution Volume Brightness Name Email Password Address City Country Notes Filter Sort Select All None Undo Redo Copy Paste Cut Print "  # noqa: E501
    "Preview Export Import Reset Default Advanced Basic General Network Account Privacy Security Update About Support"
).split()
GLYPHS = (
    "plus",
    "minus",
    "cross",
    "check",
    "arrow_l",
    "arrow_r",
    "arrow_u",
    "arrow_d",
    "play",
    "stop",
    "pause",
    "gear",
    "magnifier",
    "heart",
    "star",
    "home",
    "folder",
    "doc",
    "trash",
    "bell",
    "menu",
    "dots",
)


@lru_cache(maxsize=64)
def font(size: int):
    return ImageFont.load_default(size=max(6, int(size)))


@dataclass
class Element:
    cls: int
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in screen pixels


@dataclass
class Screen:
    image: np.ndarray  # (H, W, 3) uint8
    elements: list[Element] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def cursor(self) -> Optional[Element]:
        return next((e for e in self.elements if e.cls == ID["cursor"]), None)


def _rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0), 1), min(max(v, 0), 1))
    return int(r * 255), int(g * 255), int(b * 255)


def _mix(a, b, t):
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


@dataclass
class Theme:
    name: str
    bg: tuple
    panel: tuple
    fg: tuple
    dim: tuple
    accent: tuple
    field_bg: tuple
    border: tuple
    radius: float
    style: str  # flat | outline | bevel | neon | glass
    bg_kind: str  # flat | gradient | noise | pattern


TRAIN_FAMILIES = ("light", "dark", "contrast", "colorful", "neon")
HELDOUT_FAMILIES = ("retro", "glass")


def make_theme(rng: np.random.Generator, family: str) -> Theme:
    h = float(rng.random())
    if family == "light":
        bg = _rgb(h, 0.04, 0.96)
        return Theme(
            family,
            bg,
            _mix(bg, (255, 255, 255), 0.6),
            (30, 32, 36),
            (110, 115, 125),
            _rgb(h, 0.7, 0.85),
            (255, 255, 255),
            (190, 195, 205),
            rng.uniform(2, 10),
            rng.choice(["flat", "outline"]),
            rng.choice(["flat", "flat", "gradient"]),
        )
    if family == "dark":
        bg = _rgb(h, 0.1, 0.14)
        return Theme(
            family,
            bg,
            _mix(bg, (255, 255, 255), 0.07),
            (232, 234, 238),
            (140, 146, 156),
            _rgb(h, 0.6, 0.9),
            _mix(bg, (0, 0, 0), 0.3),
            (70, 76, 88),
            rng.uniform(2, 10),
            rng.choice(["flat", "outline"]),
            rng.choice(["flat", "gradient", "noise"]),
        )
    if family == "contrast":
        return Theme(
            family,
            (0, 0, 0),
            (0, 0, 0),
            (255, 255, 255),
            (200, 200, 120),
            (255, 230, 0),
            (0, 0, 0),
            (255, 255, 255),
            rng.uniform(0, 4),
            "outline",
            "flat",
        )
    if family == "colorful":
        bg = _rgb(h, 0.5, 0.75)
        return Theme(
            family,
            bg,
            _mix(bg, (255, 255, 255), 0.35),
            (25, 25, 40),
            (70, 70, 100),
            _rgb(h + 0.5, 0.8, 0.9),
            _mix(bg, (255, 255, 255), 0.8),
            _mix(bg, (0, 0, 0), 0.4),
            rng.uniform(4, 16),
            rng.choice(["flat", "outline"]),
            rng.choice(["gradient", "noise", "pattern"]),
        )
    if family == "neon":
        bg = _rgb(h, 0.5, 0.1)
        acc = _rgb(h + rng.choice([0.33, 0.5, 0.66]), 0.9, 1.0)
        return Theme(
            family,
            bg,
            _mix(bg, acc, 0.12),
            (225, 235, 245),
            (130, 150, 170),
            acc,
            _mix(bg, (0, 0, 0), 0.5),
            acc,
            rng.uniform(0, 8),
            "neon",
            rng.choice(["noise", "gradient", "pattern"]),
        )
    if family == "retro":
        bg = (192, 192, 192)
        return Theme(family, bg, bg, (0, 0, 0), (96, 96, 96), (0, 0, 128), (255, 255, 255), (128, 128, 128), 0, "bevel", "flat")
    if family == "glass":
        bg = _rgb(h, 0.25, 0.8)
        return Theme(
            family,
            bg,
            _mix(bg, (255, 255, 255), 0.5),
            (40, 45, 60),
            (100, 105, 120),
            _rgb(h + 0.1, 0.35, 0.9),
            _mix(bg, (255, 255, 255), 0.7),
            _mix(bg, (255, 255, 255), 0.9),
            rng.uniform(8, 20),
            "glass",
            "gradient",
        )
    raise ValueError(family)


def background(rng: np.random.Generator, w: int, h: int, t: Theme) -> Image.Image:
    """Smooth backgrounds are computed at a quarter of the resolution and scaled up (they are smooth anyway)."""
    qw, qh = max(8, w // 4), max(8, h // 4)
    base = np.empty((qh, qw, 3), np.float32)
    base[:] = np.array(t.bg, np.float32)
    resample = Image.BICUBIC
    if t.bg_kind == "gradient":
        c2 = np.array(_mix(t.bg, _rgb(float(rng.random()), 0.5, 0.5), 0.5), np.float32)
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:qh, 0:qw]
        u = ((xx * np.cos(ang) + yy * np.sin(ang)) / max(qw, qh)).clip(0, 1)[..., None]
        base = base * (1 - u) + c2 * u
    elif t.bg_kind == "noise":  # soft blotches, like a game scene or a photo wallpaper
        acc = np.zeros((qh, qw), np.float32)
        for octave, amp in ((6, 1.0), (16, 0.5), (48, 0.25)):
            g = rng.random((octave, int(octave * w / h) + 1)).astype(np.float32)
            acc += amp * np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((qw, qh), Image.BICUBIC), np.float32) / 255
        acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
        c2 = np.array(_mix(t.bg, _rgb(float(rng.random()), 0.6, 0.7), 0.6), np.float32)
        base = base * (1 - acc[..., None]) + c2 * acc[..., None]
    elif t.bg_kind == "pattern":
        s = int(rng.integers(6, 24))
        yy, xx = np.mgrid[0:qh, 0:qw]
        m = (((xx // s) + (yy // s)) % 2).astype(np.float32)
        base = base * (1 - 0.12 * m[..., None]) + 12 * m[..., None]
        resample = Image.NEAREST
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    return img if (qw, qh) == (w, h) else img.resize((w, h), resample)


class Painter:
    def __init__(self, rng: np.random.Generator, w: int, h: int, theme: Theme, k: float) -> None:
        self.rng, self.w, self.h, self.t, self.k = rng, w, h, theme, k
        self.img = background(rng, w, h, theme)
        self.d = ImageDraw.Draw(self.img, "RGBA")
        self.els: list[Element] = []

    # ---- helpers
    def word(self, n: int = 1) -> str:
        return " ".join(self.rng.choice(WORDS, n))

    def tsize(self, s: str, size: float) -> tuple[int, int]:
        b = self.d.textbbox((0, 0), s, font=font(size))
        return b[2] - b[0], b[3] - b[1]

    def text(self, xy, s: str, size: float, fill, anchor="la") -> None:
        self.d.text(xy, s, font=font(size), fill=fill, anchor=anchor)

    def add(self, cls: str, box) -> tuple:
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(self.w), x1), min(float(self.h), y1)
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            self.els.append(Element(ID[cls], (x0, y0, x1, y1)))
        return (x0, y0, x1, y1)

    def shade(self, c, f):
        return tuple(int(min(255, max(0, v * f))) for v in c)

    def rect(self, box, fill=None, outline=None, width=1, radius=None) -> None:
        r = self.t.radius * self.k if radius is None else radius
        x0, y0, x1, y1 = box
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        if r >= 1:
            self.d.rounded_rectangle([x0, y0, x1, y1], r, fill=fill, outline=outline, width=width)
        else:
            self.d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width)

    # ---- interactive elements (annotated)
    def button(self, x, y, w, h, label=None, state="normal", cls="button", accent=None, annotate=True):
        t, k = self.t, self.k
        acc = accent or t.accent
        label = label if label is not None else self.word(int(self.rng.integers(1, 3)))
        box = (x, y, x + w, y + h)
        hov = 1.12 if state == "hover" else 1.0
        if t.style == "bevel":
            self.d.rectangle(box, fill=t.panel)
            self.d.line([(x, y), (x + w, y)], fill=(255, 255, 255), width=2)
            self.d.line([(x, y), (x, y + h)], fill=(255, 255, 255), width=2)
            self.d.line([(x, y + h), (x + w, y + h)], fill=(64, 64, 64), width=2)
            self.d.line([(x + w, y), (x + w, y + h)], fill=(64, 64, 64), width=2)
            tc = t.fg
        elif t.style == "neon":
            self.rect(box, fill=self.shade(t.panel, 1.0 + 0.5 * (hov - 1)) + (255,), outline=acc, width=max(1, int(2 * k)))
            self.rect((x - 2 * k, y - 2 * k, x + w + 2 * k, y + h + 2 * k), outline=acc + (70,), width=max(1, int(2 * k)))
            tc = acc
        elif t.style == "glass":
            self.rect(box, fill=(255, 255, 255, 120 if state == "normal" else 170), outline=(255, 255, 255, 220), width=max(1, int(k)))
            tc = t.fg
        elif t.style == "outline" or (t.style == "flat" and self.rng.random() < 0.25):
            self.rect(
                box, fill=(t.bg + (0,)) if state == "normal" else self.shade(acc, 0.3) + (90,), outline=acc, width=max(1, int(1.5 * k))
            )
            tc = acc if t.name != "light" else self.shade(acc, 0.7)
        else:
            self.rect(box, fill=self.shade(acc, hov))
            lum = 0.3 * acc[0] + 0.59 * acc[1] + 0.11 * acc[2]
            tc = (255, 255, 255) if lum < 150 else (20, 20, 20)
        size = max(8, h * 0.42)
        tw, _ = self.tsize(label, size)
        while tw > w - 8 and size > 7:
            size -= 1
            tw, _ = self.tsize(label, size)
        self.text((x + w / 2, y + h / 2), label, size, tc, "mm")
        return self.add(cls, box) if annotate else box

    def icon(self, x, y, s, glyph=None, caption=None, backing=None):
        t = self.t
        glyph = glyph or str(self.rng.choice(GLYPHS))
        col = t.fg if self.rng.random() < 0.6 else t.accent
        if backing == "circle":
            self.d.ellipse([x, y, x + s, y + s], fill=_mix(t.panel, t.fg, 0.1) + (255,), outline=t.border)
        elif backing == "square":
            self.rect((x, y, x + s, y + s), fill=_mix(t.panel, t.fg, 0.08) + (255,), radius=s * 0.2)
        self.glyph(glyph, x + s * 0.2, y + s * 0.2, s * 0.6, col)
        box = [x, y, x + s, y + s]
        if caption:
            size = max(8, s * 0.22)
            tw, th = self.tsize(caption, size)
            self.text((x + s / 2, y + s + 4 * self.k), caption, size, t.fg, "ma")
            box = [min(x, x + s / 2 - tw / 2), y, max(x + s, x + s / 2 + tw / 2), y + s + 4 * self.k + th + 4]
        return self.add("icon", box)

    def glyph(self, g, x, y, s, col, wd=None) -> None:
        d = self.d
        wd = wd or max(2, int(s * 0.12))
        cx, cy, r = x + s / 2, y + s / 2, s / 2
        if g == "plus":
            d.line([(x, cy), (x + s, cy)], col, wd)
            d.line([(cx, y), (cx, y + s)], col, wd)
        elif g == "minus":
            d.line([(x, cy), (x + s, cy)], col, wd)
        elif g == "cross":
            d.line([(x, y), (x + s, y + s)], col, wd)
            d.line([(x, y + s), (x + s, y)], col, wd)
        elif g == "check":
            d.line([(x, cy), (x + s * 0.4, y + s), (x + s, y)], col, wd)
        elif g.startswith("arrow"):
            dx, dy = {"arrow_l": (-1, 0), "arrow_r": (1, 0), "arrow_u": (0, -1), "arrow_d": (0, 1)}[g]
            tip, tail = (cx + dx * r, cy + dy * r), (cx - dx * r, cy - dy * r)
            d.line([tail, tip], col, wd)
            px, py = -dy, dx
            d.line([tip, (tip[0] - dx * r * 0.6 + px * r * 0.5, tip[1] - dy * r * 0.6 + py * r * 0.5)], col, wd)
            d.line([tip, (tip[0] - dx * r * 0.6 - px * r * 0.5, tip[1] - dy * r * 0.6 - py * r * 0.5)], col, wd)
        elif g == "play":
            d.polygon([(x + s * 0.2, y), (x + s, cy), (x + s * 0.2, y + s)], fill=col)
        elif g == "stop":
            d.rectangle([x + s * 0.1, y + s * 0.1, x + s * 0.9, y + s * 0.9], fill=col)
        elif g == "pause":
            d.rectangle([x + s * 0.15, y, x + s * 0.4, y + s], fill=col)
            d.rectangle([x + s * 0.6, y, x + s * 0.85, y + s], fill=col)
        elif g == "gear":
            pts = []
            for i in range(16):
                a = i * np.pi / 8
                rr = r if i % 2 == 0 else r * 0.75
                pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
            d.polygon(pts, fill=col)
            d.ellipse([cx - r * 0.3, cy - r * 0.3, cx + r * 0.3, cy + r * 0.3], fill=self.t.panel)
        elif g == "magnifier":
            d.ellipse([x, y, x + s * 0.7, y + s * 0.7], outline=col, width=wd)
            d.line([(x + s * 0.6, y + s * 0.6), (x + s, y + s)], col, wd)
        elif g == "heart":
            d.ellipse([x, y, x + s * 0.55, y + s * 0.55], fill=col)
            d.ellipse([x + s * 0.45, y, x + s, y + s * 0.55], fill=col)
            d.polygon([(x + s * 0.03, y + s * 0.4), (x + s * 0.97, y + s * 0.4), (cx, y + s)], fill=col)
        elif g == "star":
            pts = [
                (cx + (r if i % 2 == 0 else r * 0.45) * np.sin(i * np.pi / 5), cy - (r if i % 2 == 0 else r * 0.45) * np.cos(i * np.pi / 5))
                for i in range(10)
            ]
            d.polygon(pts, fill=col)
        elif g == "home":
            d.polygon([(x, cy), (cx, y), (x + s, cy)], fill=col)
            d.rectangle([x + s * 0.15, cy, x + s * 0.85, y + s], fill=col)
        elif g == "folder":
            d.rectangle([x, y + s * 0.2, x + s * 0.4, y + s * 0.35], fill=col)
            d.rectangle([x, y + s * 0.3, x + s, y + s * 0.9], fill=col)
        elif g == "doc":
            d.polygon(
                [(x + s * 0.15, y), (x + s * 0.65, y), (x + s * 0.85, y + s * 0.2), (x + s * 0.85, y + s), (x + s * 0.15, y + s)],
                outline=col,
                width=wd,
            )
        elif g == "trash":
            d.rectangle([x + s * 0.2, y + s * 0.25, x + s * 0.8, y + s], outline=col, width=wd)
            d.line([(x + s * 0.1, y + s * 0.2), (x + s * 0.9, y + s * 0.2)], col, wd)
        elif g == "bell":
            d.pieslice([x + s * 0.15, y, x + s * 0.85, y + s * 0.9], 180, 360, fill=col)
            d.rectangle([x + s * 0.15, cy - s * 0.05, x + s * 0.85, y + s * 0.75], fill=col)
        elif g == "menu":
            for i in range(3):
                yy = y + s * (0.15 + 0.35 * i)
                d.line([(x, yy), (x + s, yy)], col, wd)
        else:  # dots
            for i in range(3):
                d.ellipse([cx - wd, y + s * (0.1 + 0.4 * i) - wd, cx + wd, y + s * (0.1 + 0.4 * i) + wd], fill=col)

    def field_box(self, x, y, w, h, text=None, focused=False, search=False):
        t, k = self.t, self.k
        box = (x, y, x + w, y + h)
        if t.style == "bevel":
            self.d.rectangle(box, fill=t.field_bg)
            self.d.line([(x, y), (x + w, y)], fill=(96, 96, 96), width=2)
            self.d.line([(x, y), (x, y + h)], fill=(96, 96, 96), width=2)
        elif t.style == "glass":
            self.rect(box, fill=(255, 255, 255, 150), outline=(255, 255, 255, 230))
        else:
            self.rect(
                box, fill=t.field_bg + (255,), outline=(t.accent if focused else t.border), width=max(1, int((2 if focused else 1) * k))
            )
        size = max(8, h * 0.4)
        px = x + 8 * k
        if search:
            self.glyph("magnifier", px, y + h * 0.28, h * 0.44, t.dim)
            px += h * 0.7
        if text:
            self.text((px, y + h / 2), text, size, t.fg, "lm")
        else:
            self.text((px, y + h / 2), self.word(), size, t.dim, "lm")
        if focused:
            cx = px + (self.tsize(text, size)[0] + 2 if text else 0)
            self.d.line([(cx, y + h * 0.22), (cx, y + h * 0.78)], t.fg, max(1, int(k)))
        return self.add("text_field", box)

    def menu_rows(self, x, y, w, labels, hi=None, rh=None):
        t, k = self.t, self.k
        rh = rh or 30 * k
        h = rh * len(labels)
        self.rect((x + 4 * k, y + 4 * k, x + w + 4 * k, y + h + 4 * k), fill=(0, 0, 0, 60), radius=3 * k)
        self.rect((x, y, x + w, y + h), fill=t.panel + (255,), outline=t.border, radius=3 * k)
        for i, s in enumerate(labels):
            yy = y + i * rh
            if i == hi:
                self.d.rectangle([x + 2, yy + 2, x + w - 2, yy + rh - 2], fill=t.accent + (255,))
                tc = (255, 255, 255)
            else:
                tc = t.fg
            self.text((x + 12 * k, yy + rh / 2), s, rh * 0.45, tc, "lm")
            if self.rng.random() < 0.3:
                self.text(
                    (x + w - 12 * k, yy + rh / 2),
                    "Ctrl+" + "ABCDEFGHJKLMNOPRSTUVWXYZ"[int(self.rng.integers(0, 24))],
                    rh * 0.35,
                    t.dim,
                    "rm",
                )
            self.add("menu_item", (x, yy, x + w, yy + rh))
            if self.rng.random() < 0.15 and i < len(labels) - 1:
                self.d.line([(x + 6, yy + rh), (x + w - 6, yy + rh)], t.border, 1)

    def menu_bar(self, x, y, h, labels, bar_w):
        self.d.rectangle([x, y, x + bar_w, y + h], fill=self.t.panel + (255,))
        cx = x + 6 * self.k
        for s in labels:
            size = h * 0.5
            tw, _ = self.tsize(s, size)
            self.text((cx + 10 * self.k, y + h / 2), s, size, self.t.fg, "lm")
            self.add("menu_item", (cx, y, cx + tw + 20 * self.k, y + h))
            cx += tw + 20 * self.k

    def toggle(self, x, y, kind, label, checked, size=None):
        t, k = self.t, self.k
        s = size or 22 * k
        if kind == "check":
            self.rect(
                (x, y, x + s, y + s),
                fill=(t.accent if checked else t.field_bg) + (255,),
                outline=t.accent if checked else t.border,
                width=max(1, int(1.5 * k)),
                radius=3 * k,
            )
            if checked:
                self.glyph("check", x + s * 0.2, y + s * 0.22, s * 0.6, (255, 255, 255))
            right = x + s
        elif kind == "radio":
            self.d.ellipse(
                [x, y, x + s, y + s], fill=t.field_bg + (255,), outline=t.accent if checked else t.border, width=max(1, int(1.5 * k))
            )
            if checked:
                self.d.ellipse([x + s * 0.28, y + s * 0.28, x + s * 0.72, y + s * 0.72], fill=t.accent)
            right = x + s
        else:  # switch
            sw = s * 1.8
            self.rect((x, y, x + sw, y + s), fill=(t.accent if checked else t.border) + (255,), radius=s / 2)
            kx = x + sw - s / 2 - 2 if checked else x + s / 2 + 2
            self.d.ellipse([kx - s * 0.38, y + s * 0.12, kx + s * 0.38, y + s * 0.88], fill=(255, 255, 255, 255))
            right = x + sw
        size = max(8, s * 0.7)
        tw, _ = self.tsize(label, size)
        self.text((right + 8 * k, y + s / 2), label, size, t.fg, "lm")
        return self.add("toggle", (x, y, right + 8 * k + tw, y + s))

    def tabs(self, x, y, h, labels, sel):
        t, k = self.t, self.k
        cx = x
        for i, s in enumerate(labels):
            size = h * 0.45
            tw, _ = self.tsize(s, size)
            w = tw + 28 * k
            box = (cx, y, cx + w, y + h)
            if i == sel:
                self.rect(box, fill=t.accent + (255,) if t.style != "outline" else t.panel + (255,), outline=t.accent, radius=6 * k)
                tc = (255, 255, 255) if t.style != "outline" else t.accent
            else:
                self.rect(box, fill=_mix(t.panel, t.fg, 0.06) + (255,), outline=t.border, radius=6 * k)
                tc = t.dim
            self.text((cx + w / 2, y + h / 2), s, size, tc, "mm")
            self.add("tab", box)
            cx += w + 3 * k

    def window_controls(self, x_right, y, h, style="win"):
        t, k = self.t, self.k
        w = h * 1.4
        if style == "win":
            for i, g in enumerate(("minus", "stop", "cross")):
                x = x_right - w * (3 - i)
                if g == "cross" and self.rng.random() < 0.4:
                    self.d.rectangle([x, y, x + w, y + h], fill=(200, 40, 40))
                self.glyph(g, x + w / 2 - h * 0.17, y + h / 2 - h * 0.17, h * 0.34, t.fg if g != "cross" else t.fg, wd=max(1, int(k)))
                self.add("window_control", (x, y, x + w, y + h))
        else:  # three round buttons
            d = h * 0.6
            for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
                x = x_right - h * 2.6 + i * (d + 8 * k)
                self.d.ellipse([x, y + (h - d) / 2, x + d, y + (h + d) / 2], fill=c)
                self.add("window_control", (x, y + (h - d) / 2, x + d, y + (h + d) / 2))

    def cursor(self, x, y, size, kind="arrow"):
        pts = [(0, 0), (0, 16), (4, 12.5), (7, 19), (9.5, 18), (6.5, 11.5), (11.5, 11.5)]
        s = size / 19.0
        poly = [(x + px * s, y + py * s) for px, py in pts]
        if self.rng.random() < 0.5:
            self.d.polygon(poly, fill=(255, 255, 255), outline=(0, 0, 0))
        else:
            self.d.polygon(poly, fill=(0, 0, 0), outline=(255, 255, 255))
        self.add("cursor", (x, y, x + 11.5 * s, y + 19 * s))

    # ---- things that look interactive but are not (never annotated)
    def chip(self, x, y, w, h):
        t = self.t
        self.rect((x, y, x + w, y + h), outline=t.border, fill=_mix(t.panel, t.fg, 0.04) + (255,), radius=h / 2)
        self.text((x + w / 2, y + h / 2), self.word(), h * 0.45, t.dim, "mm")

    def banner(self, x, y, w, h):
        a, b = (
            np.array(_rgb(float(self.rng.random()), 0.5, 0.8), np.float32),
            np.array(_rgb(float(self.rng.random()), 0.5, 0.5), np.float32),
        )
        u = np.linspace(0, 1, int(w))[None, :, None]
        arr = (a * (1 - u) + b * u).repeat(int(h), axis=0)
        self.img.paste(Image.fromarray(arr.astype(np.uint8)), (int(x), int(y)))
        self.text((x + w / 2, y + h / 2), self.word(2), h * 0.3, (255, 255, 255), "mm")

    def paragraph(self, x, y, w, lines, size):
        for i in range(lines):
            n = int(self.rng.integers(4, 12))
            self.text(
                (x, y + i * size * 1.4),
                " ".join(self.rng.choice(WORDS, n))[: int(w / (size * 0.55))],
                size,
                self.t.fg if i % 4 else self.t.dim,
            )

    def progress(self, x, y, w, h):
        self.rect((x, y, x + w, y + h), fill=self.t.border + (255,), radius=h / 2)
        self.rect((x, y, x + w * float(self.rng.uniform(0.2, 0.9)), y + h), fill=self.t.accent + (255,), radius=h / 2)

    def label_box(self, x, y, w, h):
        self.rect((x, y, x + w, y + h), outline=self.t.border, width=max(1, int(self.k)), radius=self.t.radius * self.k)
        self.text((x + 8 * self.k, y + h / 2), self.word(2), h * 0.4, self.t.fg, "lm")


def _nonoverlap(boxes, new, pad=4):
    return all(new[2] + pad < b[0] or new[0] - pad > b[2] or new[3] + pad < b[1] or new[1] - pad > b[3] for b in boxes)


def scene_app(p: Painter) -> None:
    rng, k, W, H = p.rng, p.k, p.w, p.h
    x0, y0 = rng.uniform(0, W * 0.15), rng.uniform(0, H * 0.12)
    x1, y1 = W - rng.uniform(0, W * 0.15), H - rng.uniform(0, H * 0.12)
    p.rect((x0, y0, x1, y1), fill=p.t.panel + (255,), outline=p.t.border, radius=p.t.radius * k * 0.5)
    th = 34 * k
    p.d.rectangle([x0, y0, x1, y0 + th], fill=_mix(p.t.panel, p.t.fg, 0.08) + (255,))
    p.text((x0 + 14 * k, y0 + th / 2), p.word(2), th * 0.45, p.t.fg, "lm")
    p.window_controls(x1, y0, th, "win" if rng.random() < 0.7 else "mac")
    y = y0 + th
    if rng.random() < 0.8:
        p.menu_bar(x0, y, 28 * k, [p.word() for _ in range(int(rng.integers(3, 8)))], x1 - x0)
        y += 28 * k
    if rng.random() < 0.7:  # toolbar of icon buttons
        x = x0 + 10 * k
        s = rng.uniform(28, 44) * k
        for _ in range(int(rng.integers(4, 12))):
            if x + s > x1 - 10 * k:
                break
            p.icon(x, y + 6 * k, s, backing=rng.choice([None, "square", "circle"]))
            x += s + rng.uniform(6, 16) * k
        y += s + 14 * k
    if rng.random() < 0.6:
        p.tabs(x0 + 10 * k, y + 4 * k, 32 * k, [p.word() for _ in range(int(rng.integers(2, 6)))], int(rng.integers(0, 2)))
        y += 44 * k
    cy = y + 16 * k
    boxes: list = []
    for _ in range(int(rng.integers(4, 14))):
        kind = rng.choice(["field", "field", "button", "toggle", "label", "chip", "icon", "banner", "para"])
        w = rng.uniform(140, 420) * k
        h = rng.uniform(30, 54) * k
        px, py = rng.uniform(x0 + 16 * k, max(x0 + 17 * k, x1 - w - 16 * k)), rng.uniform(cy, max(cy + 1, y1 - h - 16 * k))
        if py + h > y1 - 8 * k or not _nonoverlap(boxes, (px, py, px + w, py + h)):
            continue
        boxes.append((px, py, px + w, py + h))
        if kind == "field":
            p.field_box(px, py, w, h, text=p.word(2) if rng.random() < 0.5 else None, focused=rng.random() < 0.2, search=rng.random() < 0.2)
        elif kind == "button":
            p.button(px, py, min(w, 220 * k), h, state="hover" if rng.random() < 0.15 else "normal")
        elif kind == "toggle":
            p.toggle(px, py, rng.choice(["check", "radio", "switch"]), p.word(), bool(rng.random() < 0.5))
        elif kind == "label":
            p.label_box(px, py, w, h)
        elif kind == "chip":
            p.chip(px, py, min(w, 160 * k), h * 0.7)
        elif kind == "icon":
            p.icon(px, py, h, backing=rng.choice([None, "square"]))
        elif kind == "banner":
            p.banner(px, py, w, h)
        else:
            p.paragraph(px, py, w, 3, h * 0.3)


def scene_dialog(p: Painter) -> None:
    rng, k, W, H = p.rng, p.k, p.w, p.h
    w, h = rng.uniform(420, 760) * k, rng.uniform(200, 380) * k
    x, y = (W - w) / 2 + rng.uniform(-0.2, 0.2) * W, (H - h) / 2 + rng.uniform(-0.15, 0.15) * H
    p.rect((x + 8 * k, y + 8 * k, x + w + 8 * k, y + h + 8 * k), fill=(0, 0, 0, 80))
    p.rect((x, y, x + w, y + h), fill=p.t.panel + (255,), outline=p.t.border, radius=p.t.radius * k)
    th = 36 * k
    p.text((x + 16 * k, y + th / 2), p.word(2), th * 0.45, p.t.fg, "lm")
    if rng.random() < 0.6:
        p.window_controls(x + w, y, th)
    p.paragraph(x + 20 * k, y + th + 20 * k, w - 40 * k, int(rng.integers(2, 5)), 18 * k)
    if rng.random() < 0.4:
        p.field_box(x + 20 * k, y + h * 0.5, w - 40 * k, 40 * k, focused=rng.random() < 0.5)
    n = int(rng.integers(1, 4))
    bw, bh = rng.uniform(100, 170) * k, rng.uniform(36, 52) * k
    bx = x + w - 20 * k - n * (bw + 10 * k)
    for i in range(n):
        p.button(
            bx + i * (bw + 10 * k),
            y + h - bh - 20 * k,
            bw,
            bh,
            label=("OK", "Cancel", "Apply")[i % 3] if rng.random() < 0.6 else None,
            state="hover" if rng.random() < 0.15 else "normal",
        )


def scene_game_menu(p: Painter) -> None:
    rng, k, W, H = p.rng, p.k, p.w, p.h
    p.text((W / 2, H * 0.14), p.word(2).upper(), 64 * k, p.t.fg, "mm")
    n = int(rng.integers(3, 8))
    bw, bh = rng.uniform(260, 520) * k, rng.uniform(46, 80) * k
    gap = rng.uniform(10, 26) * k
    x = rng.choice([(W - bw) / 2, W * 0.08, W - bw - W * 0.08])
    y = (H - n * (bh + gap)) / 2 + H * 0.06
    hov = int(rng.integers(-1, n))
    for i in range(n):
        p.button(x, y + i * (bh + gap), bw, bh, state="hover" if i == hov else "normal")
    if rng.random() < 0.7:  # corner icons (settings, sound, ...)
        s = rng.uniform(36, 64) * k
        for i in range(int(rng.integers(1, 4))):
            p.icon(W - (i + 1) * (s + 14 * k), 14 * k, s, backing=rng.choice(["circle", "square", None]))
    if rng.random() < 0.5:
        p.toggle(W * 0.1, H * 0.9, "switch", p.word(), bool(rng.random() < 0.5))


def scene_desktop(p: Painter) -> None:
    rng, k, W, H = p.rng, p.k, p.w, p.h
    s = rng.uniform(48, 84) * k
    cols = int((W * 0.7) / (s * 2.2))
    for c in range(cols):
        for r in range(int((H * 0.8) / (s * 2.4))):
            if rng.random() < 0.55:
                p.icon(24 * k + c * s * 2.2, 24 * k + r * s * 2.4, s, caption=p.word(), backing=rng.choice([None, "square"]))
    th = 44 * k
    p.d.rectangle([0, H - th, W, H], fill=_mix(p.t.panel, p.t.fg, 0.1) + (255,))
    x = 8 * k
    for _ in range(int(rng.integers(3, 9))):
        p.icon(x, H - th + 6 * k, th - 12 * k, backing="square")
        x += th
    if rng.random() < 0.6:
        p.menu_rows(
            rng.uniform(0, W * 0.6),
            rng.uniform(H * 0.1, H * 0.6),
            rng.uniform(180, 300) * k,
            [p.word() for _ in range(int(rng.integers(3, 9)))],
            hi=int(rng.integers(0, 3)),
        )


def scene_settings(p: Painter) -> None:
    rng, k, W, H = p.rng, p.k, p.w, p.h
    x, y = W * rng.uniform(0.08, 0.3), H * 0.1
    rh = rng.uniform(48, 70) * k
    for i in range(int((H * 0.8) / rh)):
        yy = y + i * rh
        p.text((x, yy + rh / 2), p.word(2), rh * 0.36, p.t.fg, "lm")
        kind = rng.choice(["toggle", "field", "button", "none"])
        xr = x + W * 0.4
        if kind == "toggle":
            p.toggle(xr, yy + rh * 0.2, rng.choice(["check", "switch", "radio"]), "", bool(rng.random() < 0.5))
        elif kind == "field":
            p.field_box(xr, yy + rh * 0.1, 220 * k, rh * 0.8, text=p.word())
        elif kind == "button":
            p.button(xr, yy + rh * 0.1, 140 * k, rh * 0.8)


def scene_popup(p: Painter) -> None:
    rng, k, W, H = p.rng, p.k, p.w, p.h
    p.paragraph(40 * k, 60 * k, W * 0.6, int(rng.integers(8, 20)), 18 * k)
    for _ in range(int(rng.integers(1, 3))):
        p.menu_rows(
            rng.uniform(0, W * 0.7),
            rng.uniform(0, H * 0.6),
            rng.uniform(160, 320) * k,
            [p.word() for _ in range(int(rng.integers(3, 10)))],
            hi=int(rng.integers(0, 3)),
        )
    if rng.random() < 0.6:
        p.field_box(40 * k, H - 100 * k, W * 0.4, 44 * k, search=True)


SCENES = (scene_app, scene_app, scene_dialog, scene_game_menu, scene_game_menu, scene_desktop, scene_settings, scene_popup)
SIZES = ((1920, 1080), (1920, 1080), (2560, 1440), (1280, 720))


def render_screen(
    rng: np.random.Generator,
    size: Optional[tuple[int, int]] = None,
    families=TRAIN_FAMILIES,
    scale: Optional[float] = None,
    cursor: bool = True,
    blur: float = 0.0,
) -> Screen:
    w, h = size or SIZES[int(rng.integers(0, len(SIZES)))]
    fam = str(rng.choice(families))
    t = make_theme(rng, fam)
    k = scale if scale is not None else float(rng.uniform(0.75, 1.8)) * h / 1080
    p = Painter(rng, w, h, t, k)
    scene = SCENES[int(rng.integers(0, len(SCENES)))]
    scene(p)
    if cursor and rng.random() < 0.85:
        if p.els and rng.random() < 0.6:  # on or near an element, as it usually is
            e = p.els[int(rng.integers(0, len(p.els)))]
            cx = rng.uniform(e.box[0], e.box[2]) + rng.normal(0, 6)
            cy = rng.uniform(e.box[1], e.box[3]) + rng.normal(0, 6)
        else:
            cx, cy = rng.uniform(0, w - 30), rng.uniform(0, h - 40)
        p.cursor(float(np.clip(cx, 0, w - 20)), float(np.clip(cy, 0, h - 30)), float(rng.uniform(14, 40)) * max(k, 0.8))
    img = p.img
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return Screen(
        np.asarray(img.convert("RGB")).copy(), p.els, {"family": fam, "scale": k, "scene": scene.__name__, "size": (w, h), "style": t.style}
    )
