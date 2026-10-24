"""Deterministic geometry-perturbation tests."""
import numpy as np
from tess_uav.geometry.perturbations import axis_angle_to_rotation, perturb_world_cam_pose, perturb_depth

def test_zero_noise_preserves_inputs():
    rng = np.random.default_rng(42)
    p = np.eye(4, dtype=np.float32)
    d = np.array([[0.5, 3.0], [10.0, 0.0]], dtype=np.float32)
    assert np.array_equal(perturb_world_cam_pose(p, 0, 0, rng), p)
    assert np.array_equal(perturb_depth(d, 0, rng), d)

def test_rodrigues_rotation_is_orthogonal():
    r = axis_angle_to_rotation(np.array([0.1, -0.2, 0.3]))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(r) > 0

def test_seeded_pose_and_depth_repeat():
    def run():
        rng = np.random.default_rng(42)
        return (perturb_world_cam_pose(np.eye(4, dtype=np.float32), 0.01, 0.05, rng),
                perturb_depth(np.ones((3, 4), dtype=np.float32) * 10, 0.003, rng))
    first, second = run(), run()
    assert all(np.array_equal(a, b) for a, b in zip(first, second))

def test_relative_depth_noise_applied_to_full_image():
    d = np.ones((2, 3), dtype=np.float32)*10
    rng = np.random.default_rng(42)
    actual = perturb_depth(d, 0.001, rng)
    expected_rng = np.random.default_rng(42)
    expected = np.maximum(d * (1 + expected_rng.normal(0, 0.001, d.shape).astype(np.float32)), 1e-3)
    assert np.array_equal(actual, expected)
