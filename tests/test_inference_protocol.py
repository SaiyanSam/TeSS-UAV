import numpy as np
from tess_uav.inference.protocol import frame_ids,SCENE_TO_SEQUENCE,SEGEARTH_CLASSES
from tess_uav.inference.mappings import sam_area_heuristic,cityscapes_to_three

def test_frame_protocol():
    f=frame_ids()
    assert len(f)==20 and f[0]=='00000' and f[-1]=='00076'
    assert SCENE_TO_SEQUENCE['scene8']=='seq142'
    assert len(SEGEARTH_CLASSES)==8

def test_cityscapes_mapping():
    x=np.array([[7,8,9,10]],dtype=np.uint8)
    assert cityscapes_to_three(x,{7:'road',8:'building',9:'vegetation',10:'car'}).tolist()==[[2,1,3,0]]

def test_sam_area_thresholds_and_overwrite():
    large=np.array([[1,1],[1,1]],bool)
    small=np.array([[0,1],[0,0]],bool)
    result=sam_area_heuristic([{'area':60000,'segmentation':large},{'area':6000,'segmentation':small}],2,2)
    assert result.tolist()==[[2,1],[2,2]]
