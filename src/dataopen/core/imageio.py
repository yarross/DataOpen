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


def _unfilter_png(raw: bytes, w: int, h: int, bpp: int) -> np.ndarray:
    stride = w * bpp
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    pos = 0
    for y in range(h):
        ft = raw[pos]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos + 1).astype(np.int32)
        pos += 1 + stride
        if ft == 0:
            cur = line
        elif ft == 2:                                   # Up
            cur = (line + prev) & 0xFF
        else:                                           # Sub / Average / Paeth are sequential per byte
            cur = line.copy()
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                if ft == 1:
                    cur[i] = (cur[i] + a) & 0xFF
                elif ft == 3:
                    cur[i] = (cur[i] + ((a + b) >> 1)) & 0xFF
                elif ft == 4:
                    c = prev[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    cur[i] = (cur[i] + pred) & 0xFF
                else:
                    raise ValueError(f"bad PNG filter {ft}")
        out[y] = cur
        prev = cur
    return out


def _read_png(data: bytes) -> np.ndarray:
    pos, idat, w = 8, [], 0
    h = depth = ctype = interlace = 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        pos += 12 + n
    if depth != 8 or ctype not in (0, 2, 6) or interlace:
        raise ValueError("pure-Python PNG reader supports 8-bit gray/RGB/RGBA, non-interlaced; "
                         "install Pillow for other formats")
    ch = {0: 1, 2: 3, 6: 4}[ctype]
    arr = _unfilter_png(zlib.decompress(b"".join(idat)), w, h, ch).reshape(h, w, ch)
    if ch == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr[:, :, :3])


def read_image(path: Path) -> np.ndarray:
    """RGB uint8 (H, W, 3). Uses Pillow when installed, otherwise a pure-Python PNG decoder."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except ImportError:
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError("reading JPEG needs Pillow: pip install 'dataopen[capture]'")
        return _read_png(data)
