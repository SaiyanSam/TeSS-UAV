"""Frame-wise semantic evidence accumulation in metric voxel space."""
import numpy as np
from collections import defaultdict, Counter
from tess_uav.metrics.core import normalize,entropy_bits,normalized_entropy,js_divergence
class EvidenceVoxel:
    def __init__(self, key, num_classes, voxel_size, entropy_num_classes=None):
        self.key = tuple(int(v) for v in key)
        self.num_classes = int(num_classes)
        self.voxel_size = float(voxel_size)
        self.entropy_num_classes = (
            None if entropy_num_classes is None else int(entropy_num_classes)
        )

        self.evidence = np.zeros(self.num_classes, dtype=np.float64)

        self.observation_count = 0
        self.observed_frames = []
        self.top_label_history = []

        self.previous_belief = None
        self.js_values = []

    @property
    def center(self):
        return ((np.array(self.key, dtype=np.float32) + 0.5) * self.voxel_size).tolist()

    def add_evidence(self, semantic_vector, frame_id):
        semantic_vector = np.asarray(semantic_vector, dtype=np.float64)

        if semantic_vector.shape != (self.num_classes,):
            raise ValueError(
                f"Expected semantic vector shape {(self.num_classes,)}, "
                f"got {semantic_vector.shape}"
            )

        semantic_vector = np.clip(semantic_vector, 0.0, None)
        semantic_vector = normalize(semantic_vector)

        if semantic_vector is None:
            return False

        current_before = self.belief()

        self.evidence += semantic_vector
        self.observation_count += 1
        self.observed_frames.append(frame_id)
        self.top_label_history.append(int(np.argmax(semantic_vector)))

        current_after = self.belief()

        if current_before is not None and current_after is not None:
            self.js_values.append(js_divergence(current_before, current_after))

        self.previous_belief = current_after.copy()

        return True

    def belief(self):
        return normalize(self.evidence)

    def dominant_class(self):
        if self.observation_count == 0:
            return -1

        return int(np.argmax(self.evidence))

    def vss(self):
        total = float(np.sum(self.evidence))

        if total <= 0:
            return np.nan

        return float(np.max(self.evidence) / total)

    def sbd(self):
        if len(self.js_values) == 0:
            return np.nan

        return float(np.mean(self.js_values))

    def entropy(self):
        p = self.belief()

        if p is None:
            return np.nan

        return entropy_bits(p)

    def entropy_norm(self):
        p = self.belief()

        if p is None:
            return np.nan

        return normalized_entropy(
            p,
            num_entropy_classes=self.entropy_num_classes,
        )

    def to_record(self):
        p = self.belief()

        if p is None:
            p = np.zeros(self.num_classes, dtype=np.float64)

        return {
            "key": [int(v) for v in self.key],
            "center": [float(v) for v in self.center],
            "op": int(self.observation_count),
            "observed_frames": list(self.observed_frames),
            "num_observed_frames": int(len(set(self.observed_frames))),
            "accumulated_evidence": [float(v) for v in self.evidence.tolist()],
            "final_belief": [float(v) for v in p.tolist()],
            "dominant_class": int(self.dominant_class()),
            "vss": float(self.vss()) if np.isfinite(self.vss()) else None,
            "sbd": float(self.sbd()) if np.isfinite(self.sbd()) else None,
            "entropy": float(self.entropy()) if np.isfinite(self.entropy()) else None,
            "entropy_norm": float(self.entropy_norm()) if np.isfinite(self.entropy_norm()) else None,
            "top_label_history": [int(v) for v in self.top_label_history],
            "js_history": [float(v) for v in self.js_values],
        }


def accumulate_frame_evidence(
    points_world,
    labels,
    frame_id,
    voxel_size,
    num_classes,
    voxels,
    entropy_num_classes=None,
):
    """
    For each frame, first aggregate sampled pixels into voxel-level semantic evidence.

    This means each voxel receives at most one semantic evidence vector per frame.

    Example:
        voxel sees 10 sampled points in this frame:
            class 1: 7 points
            class 4: 3 points

        semantic vector = [0, 0.7, 0, 0, 0.3, 0, 0, 0]
    """
    voxel_indices = np.floor(points_world / voxel_size).astype(np.int32)

    frame_voxel_counts = defaultdict(Counter)

    for idx, label in zip(voxel_indices, labels):
        label = int(label)

        if label < 0 or label >= num_classes:
            continue

        key = tuple(int(v) for v in idx)
        frame_voxel_counts[key][label] += 1

    accepted = 0

    for key, counter in frame_voxel_counts.items():
        semantic_vector = np.zeros(num_classes, dtype=np.float64)

        for label, count in counter.items():
            semantic_vector[int(label)] = float(count)

        semantic_vector = normalize(semantic_vector)

        if semantic_vector is None:
            continue

        if key not in voxels:
            voxels[key] = EvidenceVoxel(
                key=key,
                num_classes=num_classes,
                voxel_size=voxel_size,
                entropy_num_classes=entropy_num_classes,
            )

        if voxels[key].add_evidence(semantic_vector, frame_id):
            accepted += 1

    return accepted

