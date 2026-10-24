import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]/'scripts'/'verify_manuscript.py'
spec = importlib.util.spec_from_file_location('verify_manuscript', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_precision():
    assert mod.rounded_agreement(25.388341,25.39,2)
    assert mod.rounded_agreement(15.734683,15.73,2)
    assert not mod.rounded_agreement(0.946974,0.908,3)

def test_missing_files_reports_no_false_pass(tmp_path):
    ref={'table1':{'Raw SAM proposals':21.64},'table2':{'scene0':{'voxels':2}},
         'table3':{'OP=2':{'vss':.8}},'table4':{'building':{'voxels':2,'vss':.9,'sbd':.1,'entropy':.1}}}
    found=mod.verify(tmp_path,ref)
    assert found and all(x['status']!='PASS' for x in found)
    assert any(x['status']=='UNSUPPORTED' for x in found)
