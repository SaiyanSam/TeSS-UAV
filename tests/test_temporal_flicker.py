import numpy as np
import cv2
import pytest
from tess_uav.analysis.temporal_flicker import pairwise_flicker, evaluate_sequence, summarize


def test_union_validity_and_arithmetic_mean():
    a = np.array([[0,1],[2,0]])
    b = np.array([[1,1],[0,0]])
    result = pairwise_flicker(a,b)
    assert result['valid_pixels'] == 3
    assert result['changed_pixels'] == 2
    assert result['flicker_percent'] == pytest.approx(200/3)
    assert summarize([result, dict(valid_pixels=1, changed_pixels=0, flicker_percent=0)])['mean_flicker_percent'] == pytest.approx(100/3)


def test_empty_pair_is_error():
    with pytest.raises(ValueError, match='No valid'):
        pairwise_flicker(np.zeros((2,2)),np.zeros((2,2)))


def test_generated_sequence(tmp_path):
    for i, content in enumerate((np.array([[1,0],[0,0]],dtype='uint8'),
                                 np.array([[0,1],[0,0]],dtype='uint8'))):
        d=tmp_path/f'{4*i:05d}'
        d.mkdir()
        assert cv2.imwrite(str(d/'segearth_mask.png'), content)
    rows=evaluate_sequence(tmp_path,'segearth_mask.png',num_frames=2)
    assert len(rows)==1 and rows[0]['flicker_percent']==100
    assert rows[0]['frame_pair']=='00000-00004'
