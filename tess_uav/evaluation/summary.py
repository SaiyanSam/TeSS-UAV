"""Aggregate scene-level voxel stability statistics."""
import numpy as np
def summarize(records, num_classes):
    if len(records) == 0:
        return {}

    vss = np.array(
        [r["vss"] for r in records if r["vss"] is not None],
        dtype=np.float64,
    )

    sbd = np.array(
        [r["sbd"] for r in records if r["sbd"] is not None],
        dtype=np.float64,
    )

    entropy_norm = np.array(
        [r["entropy_norm"] for r in records if r["entropy_norm"] is not None],
        dtype=np.float64,
    )

    op = np.array([r["op"] for r in records], dtype=np.float64)
    observed_frames = np.array([r["num_observed_frames"] for r in records], dtype=np.float64)

    dominant_classes = np.array([r["dominant_class"] for r in records], dtype=np.int32)

    class_counts = {}
    for c in range(num_classes):
        class_counts[str(c)] = int(np.sum(dominant_classes == c))

    return {
        "num_voxels": int(len(records)),
        "num_classes": int(num_classes),
        "mean_op": float(np.mean(op)),
        "median_op": float(np.median(op)),
        "max_op": int(np.max(op)),
        "mean_observed_frames": float(np.mean(observed_frames)),
        "median_observed_frames": float(np.median(observed_frames)),
        "max_observed_frames": int(np.max(observed_frames)),
        "mean_vss": float(np.mean(vss)) if len(vss) else None,
        "median_vss": float(np.median(vss)) if len(vss) else None,
        "stable_voxel_fraction_vss_ge_090": float(np.mean(vss >= 0.90)) if len(vss) else None,
        "stable_voxel_fraction_vss_ge_095": float(np.mean(vss >= 0.95)) if len(vss) else None,
        "mean_sbd": float(np.mean(sbd)) if len(sbd) else None,
        "median_sbd": float(np.median(sbd)) if len(sbd) else None,
        "high_drift_fraction_sbd_ge_005": float(np.mean(sbd >= 0.05)) if len(sbd) else None,
        "high_drift_fraction_sbd_ge_010": float(np.mean(sbd >= 0.10)) if len(sbd) else None,
        "mean_entropy_norm": float(np.mean(entropy_norm)) if len(entropy_norm) else None,
        "median_entropy_norm": float(np.median(entropy_norm)) if len(entropy_norm) else None,
        "class_voxel_counts": class_counts,
    }

