"""Camera projection and world-coordinate transformations."""
import numpy as np
def backproject(depth, mask, K, sample_stride=4, max_depth=120.0, ignore_label=0):
    H, W = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    ys, xs = np.meshgrid(
        np.arange(0, H, sample_stride),
        np.arange(0, W, sample_stride),
        indexing="ij",
    )

    z = depth[ys, xs]
    labels = mask[ys, xs]

    valid = (
        np.isfinite(z)
        & (z > 0.5)
        & (z < max_depth)
        & (labels != ignore_label)
    )

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    labels = labels[valid].astype(np.int32)

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=1)

    return points_cam, labels


def transform_points(points_cam, T_world_cam):
    if points_cam.shape[0] == 0:
        return points_cam

    ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_cam, ones], axis=1)

    points_world = (T_world_cam @ points_h.T).T[:, :3]
    return points_world.astype(np.float32)

