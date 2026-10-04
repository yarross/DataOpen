import numpy as np

from dataopen.core.models import CameraModel, Visibility
from dataopen.core.projection import project, visibility_flags


def cam(width=640, height=480, fov=90.0):
    # camera at origin looking down +x of a Z-up world (right=-y, down=-z, forward=+x)
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    M = np.eye(4)
    M[:3, :3] = R
    return CameraModel.from_vertical_fov(width, height, fov, M)


def test_point_on_axis_projects_to_principal_point():
    c = cam()
    uv, z = project(np.array([[5.0, 0, 0]]), c)
    assert np.allclose(uv[0], [320, 240]) and np.isclose(z[0], 5.0)


def test_known_pixel_offsets_fov90():
    c = cam(fov=90.0)  # fy = 240 -> 1 m up at 1 m depth is 240 px above the centre
    uv, _ = project(np.array([[1.0, 0, 1.0], [1.0, -1.0, 0]]), c)  # up; right (= -y)
    assert np.allclose(uv[0], [320, 0], atol=1e-6)
    assert np.allclose(uv[1], [320 + 240, 240], atol=1e-6)


def test_visibility_out_behind_occluded_visible():
    c = cam()
    pts = np.array([[5, 0, 0],      # visible
                    [5, 0, 0],      # occluded by depth
                    [5, -50, 0],    # far right, outside frame
                    [-5, 0, 0]],    # behind camera
                   dtype=float)
    uv, z = project(pts, c)
    depth = np.full((240, 320), np.inf, dtype=np.float32)
    depth[120, 160] = 2.0  # wall at 2 m in front of the on-axis point (depth is half-res)
    flags = visibility_flags(uv, z, c, depth=depth)
    assert list(flags) == [Visibility.OCCLUDED, Visibility.OCCLUDED, 0, 0]
    # without the wall pixel the first one is visible
    flags = visibility_flags(uv, z, c, depth=np.full((240, 320), np.inf, dtype=np.float32))
    assert flags[0] == Visibility.VISIBLE


def test_engine_visibility_overrides_depth_and_invalid_joint_is_zero():
    c = cam()
    uv, z = project(np.array([[5.0, 0, 0], [5.0, 0.1, 0]]), c)
    flags = visibility_flags(uv, z, c, joint_valid=np.array([True, False]),
                             engine_visibility=np.array([1, 2]))
    assert list(flags) == [Visibility.OCCLUDED, 0]


def test_nan_world_point_is_out_of_frame_not_crash():
    c = cam()
    uv, z = project(np.array([[np.nan, 0, 0]]), c)
    assert visibility_flags(uv, z, c, depth=np.zeros((10, 10), dtype=np.float32))[0] == 0


def test_points_within_rounding_distance_of_the_right_or_bottom_edge_are_out_of_frame():
    """Regression: 319.996 serialized with 2 decimals became 320.00 == width, a label outside the image."""
    from dataopen.core.projection import in_frame_mask
    c = cam(width=320, height=240)
    uv = np.array([[319.996, 100.0], [319.98, 100.0], [100.0, 239.997], [100.0, 239.9], [0.0, 0.0]])
    z = np.full(5, 5.0)
    assert list(in_frame_mask(uv, z, c)) == [False, True, False, True, True]
    for u in uv[in_frame_mask(uv, z, c)][:, 0]:
        assert round(float(u), 2) < c.width
