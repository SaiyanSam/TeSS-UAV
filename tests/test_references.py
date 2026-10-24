"""Saved-reference consistency tests: no claim of fresh dataset reproduction."""
from pathlib import Path
import json
import pytest
ROOT=Path(__file__).parent/'fixtures'
SCENES=['scene0','scene1','scene3','scene5','scene8']
KEYS=['num_voxels','mean_op','mean_vss','mean_sbd','mean_entropy_norm','total_sampled_world_points','total_frame_voxel_observations']
@pytest.mark.parametrize('scene',SCENES)
def test_week8_week9_saved_summaries(scene):
    a=json.loads((ROOT/'week8_evidence_voxels'/scene/'stability_summary.json').read_text())
    b=json.loads((ROOT/'week9_repro_segearth'/scene/'stability_summary.json').read_text())
    for key in KEYS:
        if key in a and key in b:
            assert a[key]==pytest.approx(b[key],abs=1e-12)
    assert b['num_voxels']>0
    assert b['used_frames']==20
