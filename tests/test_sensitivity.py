import csv
import json
from tess_uav.analysis.sensitivity import compare_summaries,summarize_metrics

def test_summary_comparison():
    a={'num_voxels':5,'mean_op':2.,'mean_vss':0.8,'mean_sbd':0.3,'mean_entropy_norm':0.1,'used_frames':20}
    assert all(compare_summaries(a,dict(a)).values())
    b=dict(a,mean_sbd=0.4)
    assert compare_summaries(a,b)['mean_sbd'] is False

def test_undefined_sbd_denominator(tmp_path):
    file=tmp_path/'vox.csv'
    with file.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['op','vss','sbd','entropy_norm']);writer.writeheader()
        writer.writerow(dict(op=1,vss=1,sbd='',entropy_norm=0))
        writer.writerow(dict(op=3,vss=0.8,sbd=0.2,entropy_norm=0.1))
    out=summarize_metrics([file]);assert out['voxels']==2
    assert out['op_ge_3']==1
    assert out['sbd_defined_voxels']==1
    assert out['mean_sbd']==0.2
