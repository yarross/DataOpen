import numpy as np
import pytest

from dataopen.core.imageio import _read_png, read_image, read_image_size, write_png
from dataopen.core.models import Annotation
from dataopen.core.schema import HUMAN_13
from dataopen.core.viz import contact_sheet, draw_annotations


def test_png_write_read_roundtrip_exact(tmp_path):
    img = np.random.default_rng(1).integers(0, 256, (31, 47, 3), dtype=np.uint8)
    write_png(tmp_path / "a.png", img)
    assert np.array_equal(read_image(tmp_path / "a.png"), img)
    assert read_image_size(tmp_path / "a.png") == (47, 31)


@pytest.mark.parametrize("ftype", [1, 2, 3, 4])
def test_png_decoder_handles_every_filter_type(ftype):
    """Encode a PNG with a given scanline filter by hand and decode it with the pure reader."""
    import struct
    import zlib
    rng = np.random.default_rng(ftype)
    img = rng.integers(0, 256, (6, 5, 3), dtype=np.uint8)
    bpp, stride = 3, 15
    rows, prev = [], np.zeros(stride, dtype=np.int32)
    for y in range(6):
        cur = img[y].reshape(-1).astype(np.int32)
        f = np.zeros(stride, dtype=np.int32)
        for i in range(stride):
            a = cur[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if ftype == 1:
                pred = a
            elif ftype == 2:
                pred = b
            elif ftype == 3:
                pred = (a + b) >> 1
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            f[i] = (cur[i] - pred) & 0xFF
        rows.append(bytes([ftype]) + f.astype(np.uint8).tobytes())
        prev = cur

    def chunk(tag, d):
        return struct.pack(">I", len(d)) + tag + d + struct.pack(">I", zlib.crc32(tag + d) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 5, 6, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))
    assert np.array_equal(_read_png(png), img)


def test_draw_annotations_marks_visible_filled_and_occluded_hollow_without_mutating():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    kp = np.zeros((13, 3))
    kp[HUMAN_13.index("l_shoulder")] = (40, 30, 2)
    kp[HUMAN_13.index("r_shoulder")] = (100, 30, 1)
    out = draw_annotations(img, [Annotation(0, kp, (20.0, 10.0, 100.0, 90.0))], HUMAN_13)
    assert img.sum() == 0                                              # input untouched
    assert tuple(out[30, 40]) == (60, 120, 255)                        # visible left joint: filled blue
    assert tuple(out[30, 100]) != (255, 70, 70)                        # occluded right joint: hollow centre
    assert (out == (255, 220, 0)).all(axis=2).any()                    # bbox drawn


def test_contact_sheet_grid_shape():
    tiles = [np.full((100, 200, 3), i * 40, dtype=np.uint8) for i in range(5)]
    sheet = contact_sheet(tiles, cols=3, tile_width=100)
    assert sheet.shape == (100, 300, 3)  # 5 tiles of 100x50 laid out in 2 rows of 3
