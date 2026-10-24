"""Metric depth decoding and validity filtering."""
import cv2
import numpy as np
def load_depth(path, depth_scale=1.0):
    """
    Decode UAVid / CARLA 24-bit depth PNG into metric depth in meters.

    OpenCV reads PNG as BGR.
    depth = (R + G*256 + B*256^2) / (256^3 - 1) * 1000
    """
    depth_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if depth_bgr is None:
        raise FileNotFoundError(f"Could not read depth file: {path}")

    b = depth_bgr[:, :, 0].astype(np.float32)
    g = depth_bgr[:, :, 1].astype(np.float32)
    r = depth_bgr[:, :, 2].astype(np.float32)

    depth_m = (r + g * 256.0 + b * 256.0**2) / (256.0**3 - 1.0) * 1000.0
    depth_m = depth_m / float(depth_scale)

    return depth_m.astype(np.float32)

