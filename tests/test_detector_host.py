"""Torch-free parts of the detector: head layout, NMS-free decoding, the KeypointArray struct and its C header, the C
post-processor (compiled and compared with the Python reference), annotation agreement and the calibration-set builder."""
import ctypes
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dataopen.detector import structs
from dataopen.detector.agreement import compare
from dataopen.detector.layout import DecodeConfig, HeadLayout
from dataopen.detector.postprocess import decode_dense, decode_level, to_predictions
from dataopen.quality.evaluators.decode import Letterbox

CSRC = Path(__file__).resolve().parents[1] / "src" / "dataopen" / "detector" / "csrc"
LAY = HeadLayout(n_cls=2, n_kpt=12, strides=(8, 16, 32), refine_stride=4, primary=(1,), input_size=(128, 128),
                 classes=("player_ct", "player_t"), keypoints=tuple(f"k{i}" for i in range(12)))


def blank_levels(size=128, bias=-6.0):
    out = []
    for s in LAY.strides:
        t = np.zeros((1, LAY.channels, size // s, size // s), np.float32)
        t[0, :LAY.n_cls] = bias                                            # no class fires
        t[0, LAY.n_cls + 4 + 3 * 12:] = 0.0
        out.append(t)
    return out


def plant(levels, level, gx, gy, cls=0, box=(10.0, 20.0, 12.0, 30.0), kp_off=None, score=6.0):
    """Write one object into the dense output of `level` at cell (gx, gy) the way the network would."""
    s = LAY.strides[level]
    t = levels[level][0]
    t[cls, gy, gx] = score
    t[LAY.n_cls:LAY.n_cls + 4, gy, gx] = np.log(np.array(box) / s)          # l, t, r, b distances from the anchor
    kp = np.zeros((12, 2)) if kp_off is None else kp_off
    t[LAY.n_cls + 4:LAY.n_cls + 4 + 24, gy, gx] = (kp / (LAY.offset_scale * s)).reshape(-1)
    t[LAY.n_cls + 4 + 24:LAY.n_cls + 4 + 36, gy, gx] = 3.0                   # keypoint score logits
    t[LAY.n_cls + 4 + 36:, gy, gx] = np.where(np.arange(12) % 2 == 0, 4.0, -4.0)    # vis logits: even points visible


def test_layout_roundtrips_through_json_and_describes_the_channels():
    assert HeadLayout.from_json(LAY.to_json()) == LAY
    assert LAY.channels == 2 + 4 + 48 and LAY.slices["kp"] == slice(6, 30) and LAY.slices["kp_vis"] == slice(42, 54)


def test_decode_finds_the_planted_objects_without_nms_and_orders_by_score():
    lv = blank_levels()
    plant(lv, 0, 5, 6, cls=0, score=3.0)                                   # a person on the stride-8 grid
    plant(lv, 1, 4, 3, cls=1, box=(20.0, 40.0, 22.0, 41.0), score=5.0)     # a bigger one, other team, stride 16
    dets = decode_dense(lv, LAY, DecodeConfig(conf_thr=0.5, refine=False))
    assert [d["cls"] for d in dets] == [1, 0] and dets[0]["score"] > dets[1]["score"]
    ax, ay = (5 + 0.5) * 8, (6 + 0.5) * 8
    assert dets[1]["box"] == pytest.approx([ax - 10, ay - 20, ax + 12, ay + 30], abs=1e-3)
    assert dets[1]["kxy"][3] == pytest.approx([ax, ay], abs=1e-3)           # zero offsets: the keypoints sit on the anchor
    assert (dets[1]["kvis"][::2] > 0.5).all() and (dets[1]["kvis"][1::2] < 0.5).all()
    assert dets[1]["kscore"][0] == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-5)


def test_neighbouring_cells_of_the_same_object_are_NOT_merged_unless_asked_and_max_det_is_enforced():
    lv = blank_levels()
    plant(lv, 0, 5, 6, score=4.0)
    plant(lv, 0, 6, 6, score=3.5, box=(18.0, 20.0, 4.0, 30.0))              # a duplicate cell covering the same person
    both = decode_dense(lv, LAY, DecodeConfig(conf_thr=0.5, refine=False))
    assert len(both) == 2                                                    # the one-to-one head is trained not to do this
    nms = decode_dense(lv, LAY, DecodeConfig(conf_thr=0.5, refine=False, nms_iou=0.5))
    assert len(nms) == 1 and nms[0]["score"] > 0.97
    many = blank_levels()
    for i in range(16):
        plant(many, 0, i, 2, score=2.0 + i * 0.1)
    for i in range(16):
        plant(many, 0, i, 5, score=1.0 + i * 0.05)
    got = decode_dense(many, LAY, DecodeConfig(conf_thr=0.5, refine=False))
    assert len(got) == 20 and all(got[i]["score"] >= got[i + 1]["score"] for i in range(19))
    assert decode_dense(blank_levels(), LAY) == []
    with pytest.raises(ValueError, match="channels"):
        decode_level(np.zeros((1, 10, 4, 4), np.float32), 8, LAY)
    with pytest.raises(ValueError, match="level tensors"):
        decode_dense(blank_levels()[:2], LAY)


def test_aim_refinement_moves_the_head_to_the_heatmap_peak_but_only_within_its_radius():
    lv = blank_levels()
    plant(lv, 0, 5, 6, score=5.0)
    ref = np.full((3, 32, 32), -8.0, np.float32)                             # stride-4 heatmap logits + offsets
    ax, ay = (5 + 0.5) * 8, (6 + 0.5) * 8                                    # = 44, 52 -> cell (11, 13) at stride 4
    ref[0, 13, 11] = 8.0
    ref[1, 13, 11], ref[2, 13, 11] = 0.25, -0.25                              # sub-cell offsets
    d = decode_dense(lv, LAY, DecodeConfig(conf_thr=0.5), ref)[0]
    assert d["kxy"][1] == pytest.approx([(11 + 0.5 + 0.25) * 4, (13 + 0.5 - 0.25) * 4], abs=0.05)     # refined
    assert d["kxy"][0] == pytest.approx([ax, ay], abs=1e-3)                    # other points untouched
    far = ref.copy()
    far[0, 13, 11] = -8.0
    far[0, 2, 2] = 8.0                                                         # a peak far from the regressed point
    d2 = decode_dense(lv, LAY, DecodeConfig(conf_thr=0.5), far)[0]
    assert d2["kxy"][1] == pytest.approx([ax, ay], abs=1e-3)                   # rejected: outside refine_radius_px


def test_predictions_are_mapped_back_through_the_letterbox():
    lv = blank_levels()
    plant(lv, 0, 5, 6, score=5.0)
    d = decode_dense(lv, LAY, DecodeConfig(conf_thr=0.5, refine=False))
    lb = Letterbox(0.1, 0.0, 14.0, 128, 128, 1280, 720)                       # 1280x720 letterboxed into 128x128
    p = to_predictions(d, lb, LAY)[0]
    x1 = (44 - 10 - 0) / 0.1
    assert p.bbox[0] == pytest.approx(x1, abs=1e-2) and p.keypoints.shape == (12, 3) and p.class_id == 0
    assert p.keypoints[3, 0] == pytest.approx((44 - 0) / 0.1, abs=1e-2) and p.keypoints[3, 1] == pytest.approx((52 - 14) / 0.1, abs=1e-2)
    assert p.visibility.shape == (12,)


# ---- the binary result ----

def test_struct_sizes_and_roundtrip_with_subpixel_precision_and_a_16_bit_visibility_mask():
    assert ctypes.sizeof(structs.ApolloDetection) == 74 and ctypes.sizeof(structs.KeypointArray) == 1516
    kx = np.array([[100.03 + i, 50.97 - i] for i in range(12)])
    det = {"cls": 1, "score": 0.83, "box": np.array([90.4, 40.2, 130.7, 190.9]), "kxy": kx, "kscore": np.linspace(0, 1, 12),
           "kvis": np.array([1.0] * 12)}
    kw = dict(keypoints=LAY.keypoints, classes=LAY.classes)
    lb = Letterbox(0.5, 0, 140, 640, 640, 1280, 720)
    a = structs.pack([det], frame_id=77, timestamp_us=123456, brightness=300, letterbox=lb, **kw)
    assert a.detection_count == 1 and a.avg_scene_brightness == 255 and a.detections[0].vis_mask == 0xFFF   # 12 bits, not 8
    out = structs.unpack(structs.validate(bytes(a), **kw))[0]
    assert out["cls"] == 1 and out["score"] == pytest.approx(0.83, abs=1 / 255)
    assert np.abs(out["kxy"] - kx).max() <= 1 / 32 + 1e-9                    # Q12.4: half an LSB (0.03 px)
    assert np.abs(out["box"] - det["box"]).max() <= 1 / 16
    assert (out["kvis"] == 1).all() and a.src_w == 1280 and a.pad_y == 140 and a.letterbox_scale == pytest.approx(0.5)
    big = structs.pack([det] * 25, 1, 1, 0, **kw)
    assert big.detection_count == structs.MAX_DET                             # never more than 20


def test_consumer_validation_rejects_wrong_size_magic_version_and_schema():
    kw = dict(keypoints=LAY.keypoints, classes=LAY.classes)
    good = bytes(structs.pack([], 1, 2, 3, **kw))
    structs.validate(good, **kw)
    with pytest.raises(ValueError, match="bytes"):
        structs.validate(good[:-1], **kw)
    with pytest.raises(ValueError, match="magic"):
        structs.validate(b"\0" * len(good), **kw)
    with pytest.raises(ValueError, match="schema hash"):
        structs.validate(good, keypoints=("a", "b"), classes=("x",))
    bad = bytearray(good)
    bad[4:6] = (99).to_bytes(2, "little")
    with pytest.raises(ValueError, match="version"):
        structs.validate(bytes(bad), **kw)
    assert structs.schema_hash(["a", "b"], ["c"]) != structs.schema_hash(["a", "b"], ["d"])


needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")


@needs_gcc
def test_the_generated_c_header_matches_the_python_layout_and_its_static_asserts_hold(tmp_path):
    committed = (CSRC / "apollo_detection.h").read_text()
    assert committed == structs.header_text(), "regenerate: dataopen detector header --out src/dataopen/detector/csrc/apollo_detection.h"
    src = tmp_path / "t.c"
    src.write_text('#include "apollo_detection.h"\n#include <stddef.h>\n#include <stdio.h>\n'
                   'int main(void){printf("%zu %zu %zu", sizeof(ApolloDetection), sizeof(KeypointArray),\n'
                   '  offsetof(KeypointArray, detections));return 0;}\n')
    exe = tmp_path / "t"
    subprocess.run(["gcc", "-std=c11", "-Wall", "-Werror", f"-I{CSRC}", str(src), "-o", str(exe)], check=True)
    sizes = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    assert [int(x) for x in sizes] == [74, 1516, ctypes.sizeof(structs.KeypointArray) - 20 * 74]


class CCfg(ctypes.Structure):
    _fields_ = [("n_cls", ctypes.c_int), ("n_kpt", ctypes.c_int), ("n_levels", ctypes.c_int), ("strides", ctypes.c_int * 4),
                ("offset_scale", ctypes.c_float), ("refine_stride", ctypes.c_int), ("n_primary", ctypes.c_int),
                ("primary", ctypes.c_int * 4), ("max_det", ctypes.c_int), ("conf_thr", ctypes.c_float),
                ("refine_radius_px", ctypes.c_float)]


@needs_gcc
def test_the_c_postprocessor_matches_the_python_reference_on_random_tensors(tmp_path):
    so = tmp_path / "apollo_post.so"
    subprocess.run(["gcc", "-O2", "-Wall", "-Wextra", "-Werror", "-std=c99", "-shared", "-fPIC", f"-I{CSRC}", "-o", str(so),
                    str(CSRC / "apollo_post.c"), "-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    lib.apollo_decode.restype = ctypes.c_int
    rng = np.random.default_rng(3)
    lv = [rng.normal(0, 1.5, (1, LAY.channels, 128 // s, 128 // s)).astype(np.float32) for s in LAY.strides]
    for t in lv:
        t[0, :2] -= 2.0
        t[0, 2:6] = rng.normal(1.0, 0.3, t[0, 2:6].shape)
        t[0, 6:30] = rng.normal(0, 0.4, t[0, 6:30].shape)
    ref = rng.normal(0, 2.0, (3, 32, 32)).astype(np.float32)
    ref[1:] *= 0.1
    cfg = DecodeConfig(conf_thr=0.5, max_det=20)
    py = decode_dense([t for t in lv], LAY, cfg, ref)
    assert len(py) > 3
    c = CCfg(2, 12, 3, (8, 16, 32, 0), LAY.offset_scale, 4, 1, (1, 0, 0, 0), 20, cfg.conf_thr, cfg.refine_radius_px)
    ptrs = (ctypes.POINTER(ctypes.c_float) * 3)(*[t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for t in lv])
    H = (ctypes.c_int * 3)(*[t.shape[2] for t in lv])
    W = (ctypes.c_int * 3)(*[t.shape[3] for t in lv])
    out = structs.KeypointArray()
    n = lib.apollo_decode(ctypes.byref(c), ptrs, H, W, ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 32, 32, ctypes.byref(out))
    assert n == len(py) == out.detection_count
    ref_pack = structs.pack(py, 0, 0, 0)
    for i in range(n):
        a, b = out.detections[i], ref_pack.detections[i]
        assert a.class_id == b.class_id and abs(a.confidence - b.confidence) <= 1 and a.vis_mask == b.vis_mask, i
        assert max(abs(int(a.bbox[j]) - int(b.bbox[j])) for j in range(4)) <= 1
        assert max(abs(int(a.keypoints[k][j]) - int(b.keypoints[k][j])) for k in range(12) for j in range(2)) <= 1, i
        assert max(abs(int(a.kp_confidence[k]) - int(b.kp_confidence[k])) for k in range(12)) <= 1
    assert out.magic == structs.MAGIC and out.version == structs.VERSION
    bad = CCfg(2, 7, 3, (8, 16, 32, 0), 4.0, 0, 0, (0, 0, 0, 0), 20, 0.5, 6.0)
    assert lib.apollo_decode(ctypes.byref(bad), ptrs, H, W, None, 0, 0, ctypes.byref(out)) == -1


# ---- agreement and calibration set ----

def coco(boxes_by_image):
    return {name: [{"bbox": b, "cls": 0} for b in bs] for name, bs in boxes_by_image.items()}


def test_agreement_counts_iou_misses_unmatched_and_class_conflicts():
    a = coco({"i1": [[10, 10, 50, 100]], "i2": [[0, 0, 40, 80], [100, 0, 40, 80]], "i3": [[5, 5, 30, 60]]})
    b = coco({"i1": [[11, 10, 50, 100]], "i2": [[0, 0, 40, 80]], "i3": [[40, 40, 30, 60]]})
    r = compare(a, b, 0.85)
    assert r["images_compared"] == 3 and r["matched_persons"] == 2 and r["unmatched_in_a"] == 2 and r["unmatched_in_b"] == 1
    assert not r["passes"] and r["share_iou_ge_thr"] == 1.0 and {x["image"] for x in r["images_to_review"]} == {"i2", "i3"}
    same = compare(a, a, 0.85)
    assert same["passes"] and same["mean_iou"] == 1.0 and same["unmatched_in_a"] == 0
    b2 = coco({"i1": [[10, 10, 50, 100]]})
    b2["i1"][0]["cls"] = 1
    assert compare(coco({"i1": [[10, 10, 50, 100]]}), b2)["class_mismatch"] == 1


def test_average_precision_and_target_checks():
    from dataopen.detector.evaluate import average_precision, check_targets
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([0.9, 0.8]), np.array([False, False]), 2) == 0.0
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)    # recall 0.5 @ precision 1, 1.0 @ 2/3
    assert ap == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101, abs=0.01)
    assert np.isnan(average_precision(np.zeros(0), np.zeros(0, bool), 0))
    assert check_targets({"map50": 0.95, "kp_ap_oks50": 0.9}) == [] and len(check_targets({"map50": 0.9, "kp_ap_oks50": 0.5})) == 2


@pytest.fixture(scope="module")
def shooter_dataset(tmp_path_factory):
    from dataopen.cli import main
    out = tmp_path_factory.mktemp("calib_src") / "ds"
    with pytest.raises(SystemExit) as e:
        main(["collect", "--game", "mock_shooter", "--frames", "70", "--out", str(out), "--quality-sim", "--no-doctor", "--seed", "11"])
    assert e.value.code == 0
    return out


def test_calibration_set_is_stratified_oversamples_hard_frames_and_writes_the_rknn_listing(shooter_dataset, tmp_path):
    from dataopen.detector.calibration_set import build_calibration_set
    rep = build_calibration_set([shooter_dataset], tmp_path / "calib", n=30, size=96, hard_share=0.5)
    assert rep["images"] == 30 and rep["strata_covered"] >= 3
    assert rep["by_verdict"].get("keep_hard", 0) >= 10                       # the quota favours hard frames over their natural share
    lines = (tmp_path / "calib" / "dataset.txt").read_text().split()
    assert len(lines) == 30 and (tmp_path / "calib" / lines[0]).exists()
    from dataopen.core.imageio import read_image
    assert read_image(tmp_path / "calib" / lines[0]).shape == (96, 96, 3)
    small = build_calibration_set([shooter_dataset], tmp_path / "c2", n=5000, size=64)
    assert "warning" in small and small["images"] < 5000                     # asking for more than exists is reported, not hidden
    json.loads((tmp_path / "calib" / "calibration_report.json").read_text())
    with pytest.raises(ValueError, match="no records"):
        build_calibration_set([shooter_dataset], tmp_path / "c3", split="nonexistent")
