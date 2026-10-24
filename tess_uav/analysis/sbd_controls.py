"""Temporal-order controls for cumulative semantic belief drift."""
from collections import Counter, defaultdict
import numpy as np
from tess_uav.metrics.core import normalize, js_divergence
from tess_uav.fusion.evidence import EvidenceVoxel


def cumulative_sbd(history):
    if len(history) < 2:
        return np.nan
    cumulative = np.zeros_like(history[0], dtype=np.float64)
    previous = None
    values = []
    for evidence in history:
        cumulative += evidence
        current = normalize(cumulative)
        if previous is not None:
            values.append(js_divergence(previous, current))
        previous = current
    return float(np.mean(values)) if values else np.nan


def direct_sbd(history):
    if len(history) < 2:
        return np.nan
    return float(np.mean([js_divergence(history[i - 1], history[i]) for i in range(1, len(history))]))


def shuffled_cumulative_sbd(history, num_permutations, rng):
    if len(history) < 2:
        return np.nan, np.nan
    vals = np.empty(num_permutations, dtype=np.float64)
    for i in range(num_permutations):
        order = rng.permutation(len(history))
        vals[i] = cumulative_sbd([history[j] for j in order])
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


class TemporalEvidenceVoxel(EvidenceVoxel):
    """Standard evidence voxel with per-observation vectors retained for controls."""
    def __init__(self, key, num_classes, voxel_size):
        super().__init__(key, num_classes, voxel_size)
        self.evidence_history = []

    def add_evidence(self, semantic_vector, frame_id):
        normalized = normalize(np.clip(np.asarray(semantic_vector, dtype=np.float64), 0.0, None))
        if normalized is None:
            return False
        accepted = super().add_evidence(normalized, frame_id)
        if accepted:
            self.evidence_history.append(normalized.copy())
        return accepted


def accumulate_temporal_evidence(points_world, labels, frame_id, voxel_size, num_classes, voxels):
    """Produce one normalized class-histogram observation per voxel per frame."""
    indices = np.floor(points_world / voxel_size).astype(np.int32)
    counts = defaultdict(Counter)
    for idx, label in zip(indices, labels):
        label = int(label)
        if 0 <= label < num_classes:
            counts[tuple(int(v) for v in idx)][label] += 1
    accepted = 0
    for key, counter in counts.items():
        vector = np.zeros(num_classes, dtype=np.float64)
        for label, count in counter.items():
            vector[label] = float(count)
        vector = normalize(vector)
        if vector is None:
            continue
        if key not in voxels:
            voxels[key] = TemporalEvidenceVoxel(key, num_classes, voxel_size)
        if voxels[key].add_evidence(vector, frame_id):
            accepted += 1
    return accepted


def op_bin_name(op):
    if op == 1:
        return 'OP=1'
    if op == 2:
        return 'OP=2'
    if 3 <= op <= 5:
        return 'OP=3-5'
    return 'OP>5'


def safe_stat(values, fn):
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(fn(finite)) if finite.size else None


def summarize_subset(records, label):
    if not records:
        return dict(op_group=label, num_voxels=0, mean_op=None, cum_sbd=None,
                    shuffled_cum_sbd=None, direct_sbd=None, cum_minus_shuffled=None,
                    cum_over_shuffled=None)
    cum = [r['cum_sbd'] for r in records]
    shuf = [r['shuffled_cum_sbd'] for r in records]
    direct = [r['direct_sbd'] for r in records]
    cm, sm = safe_stat(cum, np.mean), safe_stat(shuf, np.mean)
    return dict(op_group=label, num_voxels=len(records),
                mean_op=float(np.mean([r['op'] for r in records])),
                cum_sbd=cm, cum_sbd_median=safe_stat(cum, np.median),
                shuffled_cum_sbd=sm, shuffled_cum_sbd_median=safe_stat(shuf, np.median),
                direct_sbd=safe_stat(direct, np.mean),
                direct_sbd_median=safe_stat(direct, np.median),
                cum_minus_shuffled=None if cm is None or sm is None else float(cm - sm),
                cum_over_shuffled=None if cm is None or sm is None or sm == 0 else float(cm / sm))


def evaluate_controls(voxels, num_permutations=20, seed=2026, regression_tolerance=1e-12):
    if num_permutations < 1:
        raise ValueError('num_permutations must be >= 1')
    rng = np.random.default_rng(seed)
    records = []
    max_error = 0.0
    failures = 0
    for key in sorted(voxels):
        voxel = voxels[key]
        op = voxel.observation_count
        online = voxel.sbd()
        cum = cumulative_sbd(voxel.evidence_history)
        if np.isfinite(online) and np.isfinite(cum):
            error = abs(online - cum)
            max_error = max(error, max_error)
            if error > regression_tolerance:
                failures += 1
        if op >= 2:
            shuf, std = shuffled_cumulative_sbd(voxel.evidence_history, num_permutations, rng)
            direct = direct_sbd(voxel.evidence_history)
        else:
            shuf, std, direct = np.nan, np.nan, np.nan
        records.append(dict(key_x=key[0], key_y=key[1], key_z=key[2],
                            op=op, op_bin=op_bin_name(op), cum_sbd=cum,
                            online_week9_sbd=online, shuffled_cum_sbd=shuf,
                            shuffled_cum_sbd_std=std, direct_sbd=direct,
                            cum_minus_shuffled=cum - shuf if np.isfinite(cum) and np.isfinite(shuf) else np.nan))
    if failures:
        raise RuntimeError(f'{failures} cumulative-SBD regression failures; max error {max_error:.3e}')
    groups = [('OP=2', lambda r: r['op'] == 2),
              ('OP=3-5', lambda r: 3 <= r['op'] <= 5),
              ('OP>5', lambda r: r['op'] > 5),
              ('OP>=2', lambda r: r['op'] >= 2),
              ('OP>=3', lambda r: r['op'] >= 3)]
    bins = [summarize_subset([r for r in records if predicate(r)], label) for label, predicate in groups]
    return records, bins, max_error, failures
