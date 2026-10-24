"""Tests of directional metric-depth correspondence and aggregation."""
import numpy as np
import pytest
from tess_uav.analysis.warped_flicker import warped_pair, summarize_pairs

K = np.array([[5,0,2],[0,5,2],[0,0,1.]], dtype=float)
T = np.eye(4)


def test_identity_projection_and_change():
    mask = np.ones((5,5), np.int32)
    later = mask.copy()
    later[2,2]=2
    depth=np.full((5,5), 10., dtype=float)
    s=warped_pair(mask,later,depth,depth,K,T,T,sample_stride=1)
    assert s['matched']==25
    assert s['changed']==1
    assert s['warped_flicker_percent']==4.
    assert s['raw_changed_pixels']==1


def test_depth_consistency_rejects_occlusion():
    mask = np.ones((5,5), np.int32)
    near=np.full((5,5),10.)
    far=np.full((5,5), 20.)
    s=warped_pair(mask,mask,near,far,K,T,T,sample_stride=1)
    assert s['matched']==0
    assert s['warped_flicker_percent'] is None


def test_ignore_label_and_empty():
    zeros=np.zeros((5,5),np.int32)
    depth=np.full((5,5),10.)
    stats=warped_pair(zeros,zeros,depth,depth,K,T,T,sample_stride=1)
    assert stats['source_valid']==0
    assert stats['matched']==0
    assert stats['raw_valid_pixels']==0


def test_frame_direction_geometry():
    mask=np.ones((5,5),np.int32)
    depth=np.full((5,5),10.)
    shifted=T.copy()
    shifted[0,3]=2.0
    stats=warped_pair(mask,mask,depth,depth,K,T,shifted,sample_stride=1)
    assert stats['matched']==20


def test_weighted_pooling():
    rows=[{'matched':3,'changed':1,'source_valid':5,'raw_same_pixel_flicker_percent':10.,'warped_flicker_percent':100/3},
          {'matched':1,'changed':1,'source_valid':2,'raw_same_pixel_flicker_percent':30.,'warped_flicker_percent':100.}]
    out=summarize_pairs(rows,'sceneX')
    assert out['num_pairs']==2
    assert out['total_matched_correspondences']==4
    assert out['pooled_warped_flicker_percent']==50.
    assert out['mean_raw_same_pixel_flicker_percent']==20.
