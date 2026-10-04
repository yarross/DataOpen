import numpy as np

from dataopen.core.models import Visibility
from dataopen.core.schema import HUMAN_13
from dataopen.core.self_occlusion import _closest_segment_params, apply_self_occlusion

V, O = int(Visibility.VISIBLE), int(Visibility.OCCLUDED)


def standing():
    """Person at the origin, +z up, shoulders along y (0.4 m apart)."""
    s = np.zeros((13, 3))
    pos = {"head": (0, 0, 1.7), "neck": (0, 0, 1.5), "l_shoulder": (0, 0.2, 1.45), "r_shoulder": (0, -0.2, 1.45),
           "pelvis": (0, 0, 0.95), "l_elbow": (0, 0.3, 1.2), "r_elbow": (0, -0.3, 1.2),
           "l_wrist": (0, 0.3, 0.9), "r_wrist": (0, -0.3, 0.9), "l_knee": (0, 0.1, 0.5),
           "r_knee": (0, -0.1, 0.5), "l_ankle": (0, 0.1, 0.08), "r_ankle": (0, -0.1, 0.08)}
    for k, v in pos.items():
        s[HUMAN_13.index(k)] = v
    return s


def flags():
    return np.full(13, V, dtype=np.int8)


def test_closest_segment_params_basic():
    s, t, d = _closest_segment_params(np.array([0.0, 0, 0]), np.array([1.0, 0, 0]),
                                      np.array([0.5, 1, 0]), np.array([0.5, 2, 0]))
    assert np.isclose(d, 1.0) and np.isclose(s, 0.5) and np.isclose(t, 0.0)


def test_wrist_behind_torso_is_occluded_and_front_or_side_is_not():
    cam = np.array([-5.0, 0.0, 1.2])
    sk = standing()
    i = HUMAN_13.index("l_wrist")
    sk[i] = (1.0, 0.0, 1.2)                      # directly behind the torso as seen from the camera
    assert apply_self_occlusion(sk, flags(), cam, HUMAN_13)[i] == O
    sk[i] = (-0.5, 0.0, 1.2)                     # in front of the torso
    assert apply_self_occlusion(sk, flags(), cam, HUMAN_13)[i] == V
    sk[i] = (0.0, 0.6, 1.2)                      # beside the torso
    assert apply_self_occlusion(sk, flags(), cam, HUMAN_13)[i] == V


def test_torso_attached_joints_and_already_hidden_joints_are_untouched():
    cam = np.array([-5.0, 0.0, 1.2])
    f = apply_self_occlusion(standing(), flags(), cam, HUMAN_13)
    for name in ("head", "neck", "pelvis", "l_shoulder", "r_shoulder"):
        assert f[HUMAN_13.index(name)] == V
    f0 = flags()
    f0[HUMAN_13.index("l_wrist")] = 0
    sk = standing()
    sk[HUMAN_13.index("l_wrist")] = (1.0, 0.0, 1.2)
    assert apply_self_occlusion(sk, f0, cam, HUMAN_13)[HUMAN_13.index("l_wrist")] == 0


def test_hand_behind_head_is_occluded():
    cam = np.array([-5.0, 0.0, 1.7])
    sk = standing()
    i = HUMAN_13.index("r_wrist")
    sk[i] = (0.6, 0.0, 1.7)                      # raised hand directly behind the head
    assert apply_self_occlusion(sk, flags(), cam, HUMAN_13)[i] == O


def test_does_not_mutate_input_and_survives_degenerate_pose():
    f = flags()
    apply_self_occlusion(standing(), f, np.array([-5.0, 0, 1.2]), HUMAN_13)
    assert (f == V).all()
    apply_self_occlusion(np.zeros((13, 3)), flags(), np.array([1.0, 0, 0]), HUMAN_13)  # no crash / NaN
