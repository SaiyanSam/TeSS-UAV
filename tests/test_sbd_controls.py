"""Tests for temporal order controls and per-frame voxel evidence."""
import numpy as np
from tess_uav.analysis.sbd_controls import (TemporalEvidenceVoxel, cumulative_sbd,
    direct_sbd, shuffled_cumulative_sbd, accumulate_temporal_evidence, evaluate_controls)


def test_identical_observations_have_no_drift():
    h = [np.array([0.,1.,0.]) for _ in range(4)]
    assert cumulative_sbd(h) == 0
    assert direct_sbd(h) == 0
    assert shuffled_cumulative_sbd(h, 20, np.random.default_rng(2026)) == (0., 0.)


def test_two_observation_shuffle_is_order_invariant():
    h = [np.array([1.,0.]), np.array([0.,1.])]
    shuffled, std = shuffled_cumulative_sbd(h, 20, np.random.default_rng(2026))
    assert np.isclose(shuffled, cumulative_sbd(h))
    assert np.isclose(std, 0)
    assert direct_sbd(h) > cumulative_sbd(h)


def test_frame_voxel_histogram_and_metric_regression():
    voxels = {}
    pts = np.array([[0.1,0.1,0.1], [0.12,0.11,0.1], [1.1,0.1,0.1]])
    assert accumulate_temporal_evidence(pts, [1,2,1], 'a', 0.2, 3, voxels) == 2
    assert accumulate_temporal_evidence(pts[:2], [2,2], 'b', 0.2, 3, voxels) == 1
    v = voxels[(0,0,0)]
    assert v.observation_count == 2
    assert np.allclose(v.evidence_history[0], [0,0.5,0.5])
    records, bins, error, failures = evaluate_controls(voxels)
    assert failures == 0 and error < 1e-12
    assert len(records) == 2 and bins[0]['num_voxels'] == 1
    assert bins[3]['num_voxels'] == 1


def test_op_one_is_undefined():
    v = TemporalEvidenceVoxel((0,0,0), 3, 0.2)
    v.add_evidence([0,1,0], 'a')
    assert np.isnan(cumulative_sbd(v.evidence_history))
    assert np.isnan(direct_sbd(v.evidence_history))
