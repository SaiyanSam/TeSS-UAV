import csv
import math
from tess_uav.analysis.persistence import aggregate

def write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['op','vss','sbd','entropy_norm']); w.writeheader(); w.writerows(rows)

def test_pooled_weighted_and_undefined_sbd(tmp_path):
    a=tmp_path/'a.csv'; b=tmp_path/'b.csv'
    write(a,[dict(op=1,vss=1,sbd='',entropy_norm=0),dict(op=2,vss=.8,sbd=.2,entropy_norm=.1)])
    write(b,[dict(op=2,vss=.6,sbd=.4,entropy_norm=.2)])
    scenes, pooled=aggregate({'scene0':a,'scene1':b})
    p={r['group']:r for r in pooled}
    assert p['OP=1']['sbd'] is None
    assert p['OP=1']['sbd_defined']==0
    assert p['OP=2']['voxels']==2
    assert math.isclose(p['OP=2']['sbd'],.3)
    assert math.isclose(p['OP=2']['vss'],.7)
    assert math.isclose(p['OP>=2']['fraction'],2/3)
    assert p['OP>=3']['voxels']==0
    assert p['OP>=3']['sbd'] is None
    assert len(scenes)==12

def test_missing_or_nonfinite_excluded(tmp_path):
    a=tmp_path/'a.csv'
    write(a,[dict(op=3,vss=.9,sbd='nan',entropy_norm=.2),dict(op=5,vss=.8,sbd=.1,entropy_norm=.4)])
    _, rows=aggregate({'scene0':a})
    r=next(x for x in rows if x['group']=='3<=OP<=5')
    assert r['sbd_defined']==1 and r['sbd_undefined']==1
    assert math.isclose(r['sbd'],.1)
