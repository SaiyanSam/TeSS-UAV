"""Deterministic camera-pose and metric-depth perturbations."""

import numpy as np

def axis_angle_to_rotation(axis_angle):
    """
    Convert a 3D axis-angle / rotation-vector perturbation to a 3x3 rotation matrix.

    axis_angle:
        np.ndarray shape (3,), radians.

    Uses Rodrigues' formula.
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float64)

    theta = float(np.linalg.norm(axis_angle))

    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)

    axis = axis_angle / theta
    x, y, z = axis

    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )

    R = (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )

    return R.astype(np.float32)


def perturb_world_cam_pose(
    T_world_cam,
    translation_sigma,
    rotation_sigma_deg,
    rng,
):
    """
    Apply zero-mean Gaussian pose perturbations to T_world_cam.

    Translation:
        each world-frame translation component is sampled independently:
            dt_i ~ N(0, translation_sigma^2)

    Rotation:
        each component of a small rotation vector is sampled independently:
            dr_i ~ N(0, sigma_rad^2)

        The perturbation is left-multiplied onto the camera orientation:
            R_noisy = R_noise @ R_world_cam

    Notes
    -----
    - translation_sigma is in meters.
    - rotation_sigma_deg is in degrees.
    - Setting both values to 0 exactly preserves the original pose.
    """
    T = np.array(T_world_cam, dtype=np.float32, copy=True)

    if translation_sigma > 0.0:
        dt = rng.normal(
            loc=0.0,
            scale=float(translation_sigma),
            size=3,
        ).astype(np.float32)

        T[:3, 3] += dt

    if rotation_sigma_deg > 0.0:
        sigma_rad = np.deg2rad(float(rotation_sigma_deg))

        rotvec = rng.normal(
            loc=0.0,
            scale=sigma_rad,
            size=3,
        )

        R_noise = axis_angle_to_rotation(rotvec)

        T[:3, :3] = R_noise @ T[:3, :3]

    return T


def perturb_depth(
    depth,
    relative_sigma,
    rng,
):
    """
    Apply multiplicative relative Gaussian depth noise:

        D_noisy = D * (1 + epsilon)
        epsilon ~ N(0, relative_sigma^2)

    Setting relative_sigma=0 exactly preserves the original depth.

    Negative / zero perturbed depths are clipped to a small positive value;
    the normal backprojection validity mask still enforces the configured
    minimum and maximum depth thresholds.
    """
    if relative_sigma <= 0.0:
        return depth

    depth = np.asarray(depth, dtype=np.float32)

    eps = rng.normal(
        loc=0.0,
        scale=float(relative_sigma),
        size=depth.shape,
    ).astype(np.float32)

    noisy = depth * (1.0 + eps)
    noisy = np.maximum(noisy, 1e-3)

    return noisy.astype(np.float32)

