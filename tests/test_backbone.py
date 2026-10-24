import numpy as np
from tess_uav.backbones.shared_taxonomy import remap_segearth, remap_cityscapes
from tess_uav.analysis.backbone import compute_coverage, aggregate

def test_segearth_mapping():
    x = np.array([[0,1,2,3],[4,5,6,7]])
    assert remap_segearth(x).tolist() == [[0,1,2,0],[3,3,0,0]]

def test_cityscapes_mapping():
    x = np.array([[1,2,3,4]])
    assert remap_cityscapes(x, {1:'building',2:'road',3:'vegetation',4:'car'}).tolist() == [[1,2,3,0]]

def test_coverage_and_aggregation():
    z = np.array([[10., 10.], [0., 10.]])
    se = np.array([[1, 2], [1, 3]])
    sf = np.array([[1, 0], [2, 3]])
    row = compute_coverage(z, se, sf, 'scene0', '00000', 'd.png', stride=1)
    assert row['valid_depth'] == 3
    assert row['se_retained'] == 3 and row['sf_retained'] == 2
    assert row['both_retained'] == 2 and row['se_only'] == 1
    assert row['se_vegetation'] == 1 and row['sf_vegetation'] == 1
    combined = aggregate([row, row], 'scene0')
    assert combined['frames'] == 2 and combined['valid_depth'] == 6
    assert np.isclose(combined['sf_retention_given_valid_depth'], 2/3)
