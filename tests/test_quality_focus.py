"""Aim-point (primary keypoint) metrics, verdicts and the table-detection decoder for the 12-point schema."""
import numpy as np
import pytest

from dataopen.core.models import Annotation
from dataopen.core.schema_io import load_schema
from dataopen.quality.evaluators.decode import KeypointMap, Letterbox, decode_table, parse_layout
from dataopen.quality.features import compute_features
from dataopen.quality.metrics import DefaultMetricCalculator, oks
from dataopen.quality.policy import DefaultQualityPolicy, PolicyConfig
from dataopen.quality.types import Prediction, QualityMetrics, Tier, Verdict
from tests.test_schema_shooter12 import coco_person

B = load_schema("shooter12")
S = B.schema
HEAD, SPINE = S.index("head_center"), S.index("spine")


def gt_person(x=100.0, y=40.0, w=70.0, h=190.0, vis=2):
    """Ground-truth annotation in the 12-point schema (a COCO-shaped person mapped through the schema rules)."""
    kp = B.model_mapping("coco17").apply2d(coco_person(x, y, w, h))
    kp[:, 2] = vis
    return Annotation(0, kp, (x, y, w, h), {}, 0)


def pred_of(a, move=None, conf=0.9, cls=None, vis=None):
    kp = a.keypoints.copy()
    kp[:, 2] = conf
    for i, (dx, dy) in (move or {}).items():
        kp[i, 0] += dx
        kp[i, 1] += dy
    return Prediction(tuple(a.bbox), 0.95, kp, cls, vis)


def test_weighted_oks_punishes_the_aim_point_more_than_derived_points():
    a = gt_person()
    err = {HEAD: (6.0, 0.0)}
    err2 = {SPINE: (6.0, 0.0)}                                                    # same pixel error, on a derived point
    plain = lambda mv: oks(a.keypoints, pred_of(a, mv).keypoints, a.area, S.oks_sigmas())    # noqa: E731
    weighted = lambda mv: oks(a.keypoints, pred_of(a, mv).keypoints, a.area, S.oks_sigmas(), S.oks_weights())   # noqa: E731
    assert plain({}) == pytest.approx(1.0) and weighted({}) == pytest.approx(1.0)
    assert weighted(err) < weighted(err2)                                         # head: tighter sigma AND bigger weight
    assert weighted(err) < plain(err)                                             # the weights make the aim point count more
    assert oks(a.keypoints, a.keypoints, a.area, S.oks_sigmas(), None) == pytest.approx(1.0)   # weights=None is plain COCO


def test_focus_metrics_class_and_visibility_agreement():
    calc = DefaultMetricCalculator(S)
    a = gt_person()
    m = calc.compute([a], [pred_of(a, cls=0, vis=np.array([1.0] * 12))], "t")
    p = m.persons[0]
    assert m.focus_evaluated and p.focus_oks == pytest.approx(1.0) and p.focus_hit and p.class_ok is True
    assert p.vis_acc == pytest.approx(1.0) and m.class_accuracy == 1.0 and m.focus_hit_rate == 1.0
    assert len(p.kp_sim) == 12
    off = calc.compute([a], [pred_of(a, {HEAD: (0.0, -0.18 * a.bbox[3])}, cls=1)], "t")       # head found 18% of the height away
    q = off.persons[0]
    assert q.focus_oks < 0.05 and not q.focus_hit and q.class_ok is False and off.class_accuracy == 0.0
    assert q.oks > 0.7                                                                         # the BODY is still found
    lost = calc.compute([a], [], "t")
    assert lost.persons[0].focus_oks == 0.0 and lost.persons[0].focus_hit is False
    box_only = calc.compute([a], [Prediction(tuple(a.bbox), 0.9, None)], "t")
    assert box_only.persons[0].focus_oks is None and not box_only.focus_evaluated             # no keypoints: nothing to judge
    hidden = gt_person()
    hidden.keypoints[HEAD] = 0                                                                  # head not labeled (v = 0)
    assert calc.compute([hidden], [pred_of(hidden)], "t").persons[0].focus_oks is None


def scene_image(head_contrast=True, size=(480, 640)):
    """Gray background, a red-limbed person, a skin-coloured head disc (or a head camouflaged to the background)."""
    img = np.repeat(np.linspace(60, 120, size[0]).astype(np.uint8)[:, None, None], size[1], axis=1).repeat(3, axis=2)
    bg = img.copy()
    a = gt_person()
    for e0, e1 in S.edges:
        if not head_contrast and "head_center" in (e0, e1):
            continue                                            # nothing but background around a camouflaged head
        p0, p1 = a.keypoints[S.index(e0), :2], a.keypoints[S.index(e1), :2]
        for t in np.linspace(0, 1, 60):
            x, y = (p0 * (1 - t) + p1 * t).astype(int)
            img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = (200, 60, 60)
    hx, hy = a.keypoints[HEAD, :2].astype(int)
    yy, xx = np.ogrid[:size[0], :size[1]]
    disc = (xx - hx) ** 2 + (yy - hy) ** 2 <= 100
    if head_contrast:
        img[disc] = (225, 175, 140)
    else:
        img[disc] = bg[disc]                                    # the head blends into the background
    return img, a


def decide(img, a, metrics, cfg=None):
    feats = compute_features(img, [a], [], S)
    return DefaultQualityPolicy(cfg or PolicyConfig()).decide(metrics, feats, False), feats


def test_a_model_that_is_blind_on_the_head_of_a_clearly_visible_person_gets_keep_hard_and_a_higher_weight():
    img, a = scene_image()
    calc = DefaultMetricCalculator(S)
    good, _ = decide(img, a, calc.compute([a], [pred_of(a)], "t"))
    blind_m = calc.compute([a], [pred_of(a, {HEAD: (0, -40.0)})], "t")
    blind, feats = decide(img, a, blind_m)
    body_only_m = calc.compute([a], [pred_of(a, {i: (40.0, 0.0) for i in range(12) if i not in (HEAD, 0)})], "t")
    body, _ = decide(img, a, body_only_m)
    assert feats.persons[0].has_focus and feats.persons[0].focus_visibility == 1.0 and feats.persons[0].focus_perceptibility > 0.3
    assert good.tier is Tier.KEEP and good.verdict is Verdict.KEEP_CLEAN
    assert blind.tier is Tier.KEEP_HARD and "model_blind_on_focus" in blind.reasons[0] and "aim_region=visible" in blind.reasons[0]
    assert blind.utility == 1.0 and blind.weight >= 1.0 + 0.85 + 1.0 - 1e-9          # difficulty + the full focus boost
    assert blind.weight > good.weight + 1.5 and blind.focus_oks is not None and blind.focus_oks < 0.15
    assert body.verdict is Verdict.KEEP_HARD and body.utility < blind.utility        # a hard BODY is worth less than a missed HEAD
    assert blind.to_dict()["focus_oks"] is not None and blind.to_dict()["focus_difficulty"] > 0


def test_boost_depends_on_why_the_aim_region_is_hard():
    calc = DefaultMetricCalculator(S)
    img, a = scene_image()
    miss = {HEAD: (0, -40.0)}
    full, _ = decide(img, a, calc.compute([a], [pred_of(a, miss)], "t"))
    partly = gt_person(vis=2)
    partly.keypoints[HEAD, 2] = 1                                                    # the aim point is occluded (cover)
    pv, _ = decide(img, partly, calc.compute([partly], [pred_of(partly, miss)], "t"))
    camo_img, ac = scene_image(head_contrast=False)
    cv, _ = decide(camo_img, ac, calc.compute([ac], [pred_of(ac, miss)], "t"))
    assert "partly_hidden" in pv.reasons[0] and "camouflaged" in cv.reasons[0]
    assert full.weight > pv.weight > cv.weight                                       # full boost > partial > none
    assert all(v.tier is Tier.KEEP_HARD for v in (full, pv, cv))                      # still kept: they are legitimate hard cases
    off = PolicyConfig(focus_priority=0.0)                                            # the focus logic can be switched off
    plain, _ = decide(img, a, calc.compute([a], [pred_of(a, miss)], "t"), off)
    assert plain.tier is Tier.KEEP and not plain.reasons                               # body found, so nothing flags the miss


def test_static_mode_still_steers_toward_hard_heads_without_any_model():
    img, a = scene_image(head_contrast=False)
    v, _ = decide(img, a, QualityMetrics())
    easy_img, ae = scene_image()
    ve, _ = decide(easy_img, ae, QualityMetrics())
    assert v.focus_difficulty > ve.focus_difficulty and v.utility > ve.utility        # a camouflaged head is "more valuable"


def test_class_confusion_is_a_hard_case():
    img, a = scene_image()
    calc = DefaultMetricCalculator(S)
    v, _ = decide(img, a, calc.compute([a], [pred_of(a, cls=1)], "t"))                # right body, wrong team
    assert v.tier is Tier.KEEP_HARD and "class_confusion" in v.reasons[0] and v.weight > 1.5


# ---- the fixed-size detection table (set-prediction head) ------------------------------------------------------------

LB = Letterbox(0.5, 0.0, 140.0, 640, 640, 1280, 720)          # a 1280x720 frame letterboxed into 640x640


def table_row(box_xyxy_orig, score, cls, kp_orig, vis, norm=False):
    to_in = lambda x, y: (x * 0.5, y * 0.5 + 140.0)                                    # noqa: E731
    x1, y1 = to_in(box_xyxy_orig[0], box_xyxy_orig[1])
    x2, y2 = to_in(box_xyxy_orig[2], box_xyxy_orig[3])
    kp = []
    for kx, ky, kc in kp_orig:
        ix, iy = to_in(kx, ky)
        kp += [ix / 640 if norm else ix, iy / 640 if norm else iy, kc]
    box = [x1 / 640, y1 / 640, x2 / 640, y2 / 640] if norm else [x1, y1, x2, y2]
    return box + [score, cls] + kp + list(vis)


def test_decode_table_pixels_normalized_logits_class_map_and_limits():
    layout = ["xyxy", "score", "class", "kp:12", "vis:12"]
    kp = [(400 + 5 * i, 200 + 12 * i, 0.9) for i in range(12)]
    rows = [table_row((400, 200, 500, 500), 0.9, 1, kp, [1.0] * 12), table_row((10, 10, 50, 60), 0.02, 0, kp, [0.0] * 12),
            table_row((600, 100, 700, 400), 0.7, 0, kp, [0.0, 1.0] * 6)]
    p = decode_table(np.array(rows, np.float32)[None], layout, LB, 0.25, "pixels", KeypointMap.identity(12))
    assert [round(d.score, 2) for d in p] == [0.9, 0.7]                                # sorted, the 0.02 row dropped
    assert p[0].bbox == pytest.approx((400.0, 200.0, 100.0, 300.0), abs=1e-3) and p[0].class_id == 1
    assert p[0].keypoints.shape == (12, 3) and p[0].keypoints[3, :2] == pytest.approx([415.0, 236.0], abs=1e-3)
    assert p[0].visibility.shape == (12,) and p[1].visibility[1] == 1.0
    norm = decode_table(np.array([table_row((400, 200, 500, 500), 0.9, 0, kp, [1.0] * 12, norm=True)], np.float32),
                        layout, LB, 0.25, "normalized", KeypointMap.identity(12))
    assert norm[0].bbox == pytest.approx((400.0, 200.0, 100.0, 300.0), abs=1e-2)
    logit = np.array(rows[:1], np.float32).copy()
    logit[0, 4] = 2.2                                                                  # raw logit score -> sigmoid
    assert decode_table(logit, layout, LB, 0.8)[0].score == pytest.approx(1 / (1 + np.exp(-2.2)), abs=1e-4)
    mapped = decode_table(np.array(rows[:1], np.float32), layout, LB, 0.25, class_map=[None, 0])
    assert mapped[0].class_id == 0
    many = np.array([table_row((10 * i, 10, 10 * i + 30, 90), 0.9 - i * 0.01, 0, kp, [1.0] * 12) for i in range(30)], np.float32)
    assert len(decode_table(many, layout, LB, 0.1, max_det=20)) == 20
    cs = decode_table(np.array([[400, 340, 500, 640, 0.1, 0.8, 0.3]], np.float32), ["xyxy", "class_scores:3"], LB, 0.5)
    assert cs[0].class_id == 1 and cs[0].keypoints is None and cs[0].score == pytest.approx(0.8)
    with pytest.raises(ValueError, match="columns"):
        decode_table(np.zeros((1, 5, 10), np.float32), layout, LB, 0.1)
    for bad in (["score"], ["xyxy"], ["xyxy", "score", "kp:0"], ["xyxy", "score", "bogus"]):
        with pytest.raises(ValueError):
            parse_layout(bad)


def test_onnx_table_model_with_uint8_nhwc_input_end_to_end(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    from dataopen.quality.evaluators.onnx import OnnxEvaluator
    layout = ["xyxy", "score", "class", "kp:12", "vis:12"]
    kp = [(400 + 5 * i, 200 + 12 * i, 0.9) for i in range(12)]
    table = np.array([table_row((400, 200, 500, 500), 0.9, 1, kp, [1.0] * 12)], np.float32)
    inp = helper.make_tensor_value_info("images", TensorProto.UINT8, [1, 640, 640, 3])      # raw uint8, NHWC
    out = helper.make_tensor_value_info("dets", TensorProto.FLOAT, list(table.shape))
    nodes = [helper.make_node("Cast", ["images"], ["f"], to=TensorProto.FLOAT),
             helper.make_node("ReduceMean", ["f"], ["m"], keepdims=0),
             helper.make_node("Mul", ["m", "zero"], ["z"]), helper.make_node("Add", ["table", "z"], ["dets"])]
    inits = [numpy_helper.from_array(np.zeros((1,), np.float32), "zero"), numpy_helper.from_array(table, "table")]
    model = helper.make_model(helper.make_graph(nodes, "g", [inp], [out], inits), opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "t.onnx"
    onnx.save(model, str(path))
    ev = OnnxEvaluator(str(path), "table", (640, 640), conf_thr=0.25, keypoint_map=KeypointMap.identity(12), layout=layout)
    assert ev.in_dtype == "uint8" and ev.in_layout == "nhwc" and ev.has_keypoints
    (preds,) = ev.predict([np.zeros((720, 1280, 3), np.uint8)])
    assert len(preds) == 1 and preds[0].class_id == 1 and preds[0].bbox == pytest.approx((400, 200, 100, 300), abs=1e-3)
    with pytest.raises(ValueError, match="layout"):
        OnnxEvaluator(str(path), "table")
