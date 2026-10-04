"""The binary result the NPU host hands to its consumer: `KeypointArray` of up to 20 `ApolloDetection`s.

Improvements over the first draft (and why):
  * `vis_mask` is 16 bits: 12 keypoints do not fit the draft's uint8 bit mask;
  * coordinates are Q12.4 fixed point (1/16 px) instead of whole pixels: the aim point is a head a few pixels across, whole-pixel
    rounding alone costs up to 0.7 px of aim error;
  * the header carries the letterbox (scale, pad, source size) so coordinates map back to the camera frame without side channels,
    a schema hash so a consumer built for one keypoint set refuses another, a magic and a version;
  * structs are packed (no compiler-dependent padding) and their sizes are asserted in C and in Python.
The C header is generated from this file (`header_text()`), and a test compiles it and compares layouts.
"""
from __future__ import annotations

import ctypes
import hashlib
from typing import Optional, Sequence

import numpy as np

MAX_DET = 20
Q = 16.0                       # fixed point: value * 16
MAGIC = 0x4F4C5041             # 'APLO' little endian
VERSION = 2


def schema_hash(keypoints: Sequence[str], classes: Sequence[str]) -> int:
    h = hashlib.blake2b("|".join([*keypoints, "#", *classes]).encode(), digest_size=4).digest()
    return int.from_bytes(h, "little")


def make_structs(n_kpt: int = 12, max_det: int = MAX_DET):
    class ApolloDetection(ctypes.LittleEndianStructure):
        _pack_ = 1
        _fields_ = [("class_id", ctypes.c_uint8), ("confidence", ctypes.c_uint8), ("vis_mask", ctypes.c_uint16),
                    ("track_id", ctypes.c_uint16),                        # 0 = untracked
                    ("bbox", ctypes.c_uint16 * 4),                        # x, y, w, h  (Q12.4, model-input pixels)
                    ("keypoints", (ctypes.c_int16 * 2) * n_kpt),          # x, y        (Q12.4, model-input pixels)
                    ("kp_confidence", ctypes.c_uint8 * n_kpt)]            # 0..255

    class KeypointArray(ctypes.LittleEndianStructure):
        _pack_ = 1
        _fields_ = [("magic", ctypes.c_uint32), ("version", ctypes.c_uint16), ("n_keypoints", ctypes.c_uint8),
                    ("max_detections", ctypes.c_uint8), ("schema_hash", ctypes.c_uint32),
                    ("frame_id", ctypes.c_uint32), ("timestamp_us", ctypes.c_uint32),
                    ("detection_count", ctypes.c_uint8), ("avg_scene_brightness", ctypes.c_uint8),
                    ("flags", ctypes.c_uint8), ("reserved", ctypes.c_uint8),
                    ("letterbox_scale", ctypes.c_float), ("pad_x", ctypes.c_int16), ("pad_y", ctypes.c_int16),
                    ("src_w", ctypes.c_uint16), ("src_h", ctypes.c_uint16),
                    ("detections", ApolloDetection * max_det)]

    return ApolloDetection, KeypointArray


ApolloDetection, KeypointArray = make_structs()


def _q(v: float, lo: float, hi: float) -> int:
    return int(np.clip(round(v * Q), lo, hi))


def pack(dets: Sequence[dict], frame_id: int, timestamp_us: int, brightness: int, letterbox=None, flags: int = 0,
         keypoints: Sequence[str] = (), classes: Sequence[str] = (), n_kpt: int = 12, track_ids: Optional[Sequence[int]] = None):
    """Detections from `postprocess.decode_dense` (model-input pixels) -> a filled KeypointArray."""
    Det, Arr = make_structs(n_kpt)
    a = Arr()
    a.magic, a.version, a.n_keypoints, a.max_detections = MAGIC, VERSION, n_kpt, MAX_DET
    a.schema_hash = schema_hash(keypoints, classes) if keypoints else 0
    a.frame_id, a.timestamp_us = frame_id & 0xFFFFFFFF, timestamp_us & 0xFFFFFFFF
    a.avg_scene_brightness, a.flags = int(np.clip(brightness, 0, 255)), flags
    if letterbox is not None:
        a.letterbox_scale, a.pad_x, a.pad_y = float(letterbox.scale), int(letterbox.pad_x), int(letterbox.pad_y)
        a.src_w, a.src_h = int(letterbox.orig_w), int(letterbox.orig_h)
    n = min(len(dets), MAX_DET)
    a.detection_count = n
    for i, d in enumerate(dets[:n]):
        o = a.detections[i]
        o.class_id = d["cls"]
        o.confidence = int(np.clip(round(d["score"] * 255), 0, 255))
        o.track_id = 0 if track_ids is None else int(track_ids[i]) & 0xFFFF
        x1, y1, x2, y2 = d["box"]
        o.bbox[0], o.bbox[1] = _q(max(x1, 0), 0, 65535), _q(max(y1, 0), 0, 65535)
        o.bbox[2], o.bbox[3] = _q(max(x2 - x1, 0), 0, 65535), _q(max(y2 - y1, 0), 0, 65535)
        mask = 0
        for k in range(n_kpt):
            o.keypoints[k][0], o.keypoints[k][1] = _q(d["kxy"][k, 0], -32768, 32767), _q(d["kxy"][k, 1], -32768, 32767)
            o.kp_confidence[k] = int(np.clip(round(d["kscore"][k] * 255), 0, 255))
            if d["kvis"][k] > 0.5:
                mask |= 1 << k
        o.vis_mask = mask
    return a


def unpack(a, n_kpt: int = 12) -> list[dict]:
    out = []
    for i in range(a.detection_count):
        o = a.detections[i]
        x, y, w, h = (o.bbox[j] / Q for j in range(4))
        kp = np.array([[o.keypoints[k][0] / Q, o.keypoints[k][1] / Q] for k in range(n_kpt)])
        out.append({"cls": o.class_id, "score": o.confidence / 255.0, "box": np.array([x, y, x + w, y + h]), "kxy": kp,
                    "kscore": np.array([o.kp_confidence[k] / 255.0 for k in range(n_kpt)]),
                    "kvis": np.array([(o.vis_mask >> k) & 1 for k in range(n_kpt)], dtype=float), "track_id": o.track_id})
    return out


def validate(buf: bytes, keypoints: Sequence[str] = (), classes: Sequence[str] = (), n_kpt: int = 12) -> ctypes.Structure:
    """What a consumer does first: size, magic, version and schema hash are checked before any field is trusted."""
    _, Arr = make_structs(n_kpt)
    if len(buf) != ctypes.sizeof(Arr):
        raise ValueError(f"KeypointArray is {ctypes.sizeof(Arr)} bytes, got {len(buf)}")
    a = Arr.from_buffer_copy(buf)
    if a.magic != MAGIC or a.version != VERSION:
        raise ValueError(f"bad magic/version: {a.magic:#x} v{a.version}")
    if keypoints and a.schema_hash != schema_hash(keypoints, classes):
        raise ValueError("schema hash mismatch: this array was produced for another keypoint set")
    if a.detection_count > a.max_detections:
        raise ValueError("detection_count exceeds max_detections")
    return a


def header_text(n_kpt: int = 12, max_det: int = MAX_DET) -> str:
    Det, Arr = make_structs(n_kpt, max_det)
    return f"""/* Generated by dataopen.detector.structs: do not edit. Little endian, packed. */
#ifndef APOLLO_DETECTION_H
#define APOLLO_DETECTION_H
#include <stdint.h>

#define APOLLO_MAGIC 0x{MAGIC:08X}u /* 'APLO' */
#define APOLLO_VERSION {VERSION}
#define APOLLO_NUM_KEYPOINTS {n_kpt}
#define APOLLO_MAX_DETECTIONS {max_det}
#define APOLLO_Q 16 /* fixed point: pixels = value / 16 */

#pragma pack(push, 1)
typedef struct {{
    uint8_t  class_id;                       /* index into the schema's classes (0 = player_ct, 1 = player_t) */
    uint8_t  confidence;                     /* score * 255 */
    uint16_t vis_mask;                       /* bit i set = keypoint i visible (v == 2) */
    uint16_t track_id;                       /* 0 = untracked */
    uint16_t bbox[4];                        /* x, y, w, h in Q12.4 pixels (model-input space) */
    int16_t  keypoints[APOLLO_NUM_KEYPOINTS][2]; /* x, y in Q12.4 pixels (model-input space) */
    uint8_t  kp_confidence[APOLLO_NUM_KEYPOINTS]; /* calibrated keypoint score * 255 */
}} ApolloDetection;

typedef struct {{
    uint32_t magic;                          /* APOLLO_MAGIC */
    uint16_t version;                        /* APOLLO_VERSION */
    uint8_t  n_keypoints;
    uint8_t  max_detections;
    uint32_t schema_hash;                    /* blake2b-32 of keypoint and class names: consumers refuse a mismatch */
    uint32_t frame_id;
    uint32_t timestamp_us;
    uint8_t  detection_count;                /* <= APOLLO_MAX_DETECTIONS */
    uint8_t  avg_scene_brightness;           /* mean luma 0..255 of the input frame */
    uint8_t  flags;
    uint8_t  reserved;
    float    letterbox_scale;                /* camera px * scale + pad = model-input px */
    int16_t  pad_x, pad_y;
    uint16_t src_w, src_h;                   /* camera frame size */
    ApolloDetection detections[APOLLO_MAX_DETECTIONS];
}} KeypointArray;
#pragma pack(pop)

_Static_assert(sizeof(ApolloDetection) == {ctypes.sizeof(Det)}, "ApolloDetection layout");
_Static_assert(sizeof(KeypointArray) == {ctypes.sizeof(Arr)}, "KeypointArray layout");
#endif
"""
