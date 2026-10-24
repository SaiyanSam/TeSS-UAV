import numpy as np
from tess_uav.metrics.core import normalize,entropy_bits,normalized_entropy,js_divergence
from tess_uav.fusion.evidence import EvidenceVoxel, accumulate_frame_evidence
from tess_uav.evaluation.summary import summarize

def test_metrics():
    assert np.isclose(js_divergence([1,0],[0,1]),1.0,atol=1e-9)
    assert np.isclose(entropy_bits([.5,.5]),1.0)
    assert np.isclose(normalized_entropy([.5,.5]),1.0)

def test_one_vector_per_frame():
    voxels={}
    pts=np.array([[.1,.1,.1],[.12,.12,.12],[.15,.15,.15]],dtype=np.float32)
    assert accumulate_frame_evidence(pts,np.array([1,1,2]),'00000',.2,3,voxels)==1
    v=next(iter(voxels.values()))
    assert v.observation_count==1
    assert np.allclose(v.evidence,[0,2/3,1/3])
    assert accumulate_frame_evidence(pts[:1],np.array([2]),'00001',.2,3,voxels)==1
    assert v.observation_count==2
    assert np.isclose(v.vss(), (1+1/3)/2)

def test_summary():
    v=EvidenceVoxel((0,0,0),3,.2)
    v.add_evidence([0,1,0],'1')
    v.add_evidence([0,0,1],'2')
    d=summarize([v.to_record()],3)
    assert d['num_voxels']==1
    assert d['mean_op']==2
    assert np.isclose(d['mean_vss'],.5)
