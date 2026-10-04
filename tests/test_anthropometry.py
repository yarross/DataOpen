import numpy as np

from dataopen.core.anthropometry import segment_issues
from dataopen.core.schema import HUMAN_13


def person():
    pos = {"head": (0, 0, 1.7), "neck": (0, 0, 1.5), "l_shoulder": (0, 0.2, 1.45), "r_shoulder": (0, -0.2, 1.45),
           "pelvis": (0, 0, 0.95), "l_elbow": (0, 0.3, 1.2), "r_elbow": (0, -0.3, 1.2), "l_wrist": (0, 0.3, 0.9),
           "r_wrist": (0, -0.3, 0.9), "l_knee": (0, 0.1, 0.5), "r_knee": (0, -0.1, 0.5), "l_ankle": (0, 0.1, 0.08),
           "r_ankle": (0, -0.1, 0.08)}
    return np.array([pos[k] for k in HUMAN_13.keypoints], dtype=float), np.ones(13, bool)


def test_normal_person_has_no_issues():
    assert segment_issues(*person(), HUMAN_13) == []


def test_stretched_leg_and_asymmetry_and_nan_are_found():
    sk, v = person()
    sk[HUMAN_13.index("r_knee")] = (0, -0.1, -1.5)               # shin+thigh stretched to ~2.5 m
    issues = segment_issues(sk, v, HUMAN_13)
    assert any("pelvis-r_knee" in i for i in issues) and any("asymmetric" in i or "r_knee-r_ankle" in i for i in issues)
    sk2, v2 = person()
    sk2[HUMAN_13.index("l_wrist")] = (np.nan, 0, 0)
    assert any("not finite" in i for i in segment_issues(sk2, v2, HUMAN_13))


def test_centimeter_scale_is_caught_and_invalid_joints_are_skipped():
    sk, v = person()
    assert segment_issues(sk * 100, v, HUMAN_13)                  # units mistake
    v[:] = False
    assert segment_issues(sk * 100, v, HUMAN_13) == []            # nothing to judge
