"""Minimal image I/O: dependency-free PNG writer, header-only size reader for PNG/JPEG,
and a JPEG path that uses Pillow when it is installed (`pip install dataopen[capture]`)."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, img: np.ndarray, level: int = 3) -> None:
    """RGB uint8 (H, W, 3) -> PNG."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8, got {img.shape}")
    h, w, _ = img.shape
    # filter byte 0 prepended to every scanline, built in one vectorized step
    raw = np.concatenate([np.zeros((h, 1), dtype=np.uint8), img.reshape(h, w * 3)], axis=1).tobytes()
    data = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw, level)) + _chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_image(path: Path, img: np.ndarray, jpeg_quality: int = 95) -> None:
    """Write by extension: .png always works; .jpg/.jpeg needs Pillow."""
    ext = path.suffix.lower()
    if ext == ".png":
        write_png(path, img)
    elif ext in (".jpg", ".jpeg"):
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError("JPEG output needs Pillow: pip install 'dataopen[capture]'") from e
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).save(path, quality=jpeg_quality)
    else:
        raise ValueError(f"unsupported image extension {ext!r}")


def read_image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the file header only; PNG and JPEG."""
    with path.open("rb") as f:
        head = f.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        if head[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                b = f.read(1)
                if not b:
                    break
                if b != b"\xff":
                    continue
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    break
                m = marker[0]
                if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                    continue
                seglen = struct.unpack(">H", f.read(2))[0]
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    _, h, w = struct.unpack(">BHH", f.read(5))
                    return int(w), int(h)
                f.seek(seglen - 2, 1)
    raise ValueError(f"cannot read image size of {path}")
