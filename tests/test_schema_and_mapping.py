import numpy as np
import pytest

from dataopen.adapters.mock.adapter import RIG_A
from dataopen.core.schema import HUMAN_13, SkeletonSchema


def test_flip_idx_is_an_involution_swapping_left_right():
    f = HUMAN_13.flip_idx()
    assert [f[f[i]] for i in range(13)] == list(range(13))
    assert f[HUMAN_13.index("l_wrist")] == HUMAN_13.index("r_wrist")
    assert f[HUMAN_13.index("head")] == HUMAN_13.index("head")


def test_schema_rejects_unknown_edge():
    with pytest.raises(ValueError):
        SkeletonSchema("bad", ("a",), (("a", "b"),), ())


def test_weighted_mapping_and_missing_bone():
    bones = {n: np.zeros(3) for pairs in RIG_A.weights.values() for n, _ in pairs}
    bones["Thigh_L"], bones["Thigh_R"] = np.array([0, 1.0, 0]), np.array([0, -1.0, 2.0])
    pos, valid = RIG_A.resolve(HUMAN_13, bones)
    assert valid.all() and np.allclose(pos[HUMAN_13.index("pelvis")], [0, 0, 1.0])
    del bones["Thigh_R"]                                  # pelvis needs both hips
    _, valid = RIG_A.resolve(HUMAN_13, bones)
    assert not valid[HUMAN_13.index("pelvis")] and valid.sum() == 12
