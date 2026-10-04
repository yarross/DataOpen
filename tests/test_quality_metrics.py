import numpy as np
import pytest

from dataopen.core.models import Annotation
from dataopen.core.schema import HUMAN_13
from dataopen.quality.metrics import DefaultMetricCalculator, iou_xywh, oks
from dataopen.quality.types import Prediction


def gt_person(entity_id=0, x0=100.0, y0=50.0, h=300.0):
    """A standing person whose bbox is (x0, y0, h/3, h)."""
    rel = {"head": (.5, .03), "neck": (.5, .12), "l_shoulder": (.7, .17), "r_shoulder": (.3, .17), "l_elbow": (.8, .35),
           "r_elbow": (.2, .35), "l_wrist": (.85, .5), "r_wrist": (.15, .5), "pelvis": (.5, .52), "l_knee": (.6, .74),
           "r_knee": (.4, .74), "l_ankle": (.6, .97), "r_ankle": (.4, .97)}
    w = h / 3
    kp = np.array([[x0 + rel[k][0] * w, y0 + rel[k][1] * h, 2.0] for k in HUMAN_13.keypoints])
    return Annotation(entity_id, kp, (x0, y0, w, h))


def pred_from(gt: Annotation, offset=0.0, score=0.9, conf=0.9):
    kp = gt.keypoints.copy()
    kp[:, :2] += offset
    kp[:, 2] = conf
    x, y, w, h = gt.bbox
    return Prediction((x + offset, y + offset, w, h), score, kp)


def test_iou_basics():
    assert iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xywh((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert abs(iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)) - 1 / 3) < 1e-9


def test_oks_perfect_zero_and_monotonic():
    g = gt_person()
    sig = HUMAN_13.oks_sigmas()
    assert oks(g.keypoints, g.keypoints, g.area, sig) == pytest.approx(1.0)
    vals = [oks(g.keypoints, pred_from(g, off).keypoints, g.area, sig) for off in (0, 2, 5, 10, 80)]
    assert vals == sorted(vals, reverse=True) and vals[-1] < 0.05 and vals[1] > 0.9
    # unlabeled joints are ignored: a wrong prediction on a v=0 joint costs nothing
    g2 = gt_person()
    g2.keypoints[0, 2] = 0
    p = pred_from(g2)
    p.keypoints[0, :2] += 500
    assert oks(g2.keypoints, p.keypoints, g2.area, sig) == pytest.approx(1.0)
    assert oks(np.zeros((13, 3)), g.keypoints, 100.0, sig) == 0.0                     # nothing labeled


def test_matching_perfect_miss_and_phantom():
    calc = DefaultMetricCalculator(HUMAN_13)
    g = gt_person()
    m = calc.compute([g], [pred_from(g)], "t", 3.0)
    assert m.evaluated and m.mean_oks == pytest.approx(1.0) and m.recall == 1.0 and m.n_phantom == 0 and m.latency_ms == 3.0
    miss = calc.compute([g], [], "t")
    assert miss.mean_oks == 0.0 and miss.recall == 0.0 and miss.n_gt == 1
    other = gt_person(entity_id=9, x0=600)
    ph = calc.compute([g], [pred_from(g), pred_from(other, score=0.95)], "t")
    assert ph.n_phantom == 1 and ph.mean_oks == pytest.approx(1.0)
    low = calc.compute([g], [pred_from(g), pred_from(other, score=0.3)], "t")        # not confident enough to be a phantom
    assert low.n_phantom == 0
    assert not calc.compute([g], None).evaluated


def test_phantom_is_explained_by_ignored_persons_and_each_prediction_matches_one_person():
    calc = DefaultMetricCalculator(HUMAN_13)
    g, small = gt_person(), gt_person(entity_id=5, x0=700, h=40)
    m = calc.compute([g], [pred_from(g), pred_from(small)], explained_boxes=[small.bbox])
    assert m.n_phantom == 0
    a, b = gt_person(0, 100), gt_person(1, 105)                   # two overlapping people, ONE prediction
    m2 = calc.compute([a, b], [pred_from(a)])
    assert sorted(round(p.oks, 1) for p in m2.persons)[-1] == 1.0 and m2.recall == 0.5


def test_confident_disagreement_is_detected_and_box_only_models_use_iou():
    calc = DefaultMetricCalculator(HUMAN_13)
    g = gt_person()
    bad = pred_from(g)
    bad.keypoints[:, :2] += 60                                    # skeleton placed elsewhere inside the same box
    m = calc.compute([g], [bad])
    assert m.max_disagreement_score >= 0.9 and m.mean_oks < 0.3
    box_only = Prediction(g.bbox, 0.8, None)
    mb = calc.compute([g], [box_only])
    assert mb.mean_oks == pytest.approx(1.0) and mb.recall == 1.0 and mb.max_disagreement_score == 0.0
