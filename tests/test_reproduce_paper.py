"""Focused tests for compact paper orchestrator (no dataset or GPU needed)."""
import csv
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/reproduce_paper.py'
spec = importlib.util.spec_from_file_location('reproduce_paper', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_protocol_parses():
    cfg = mod.load_config(SCRIPT.parents[1] / 'configs/paper.yaml')
    assert cfg['scenes'] == ['scene0','scene1','scene3','scene5','scene8']
    assert cfg['sample_stride'] == 4
    assert cfg['voxel_size'] == 0.2


def test_frame_step_detection(tmp_path):
    root = tmp_path/'undistorted_rgb'/'scene3'
    root.mkdir(parents=True)
    for i in range(0, 200, 10):
        (root/f'seq122_{i:05d}.png').touch()
    assert mod.choose_frames(tmp_path,'scene3',20) == (0,10)
    (root/'seq122_00003.png').touch()
    with pytest.raises(ValueError,match='nonuniform'):
        mod.choose_frames(tmp_path,'scene3',20)


def test_table1_reference_verification(tmp_path):
    path=tmp_path/'table1.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','mean_flicker_percent','target_paper_percent'])
        writer.writeheader()
        writer.writerows([
            dict(model='sam_heuristic',mean_flicker_percent='29.466870',target_paper_percent='29.47'),
            dict(model='segformer',mean_flicker_percent='25.388341',target_paper_percent='25.39'),
            dict(model='segearth',mean_flicker_percent='15.734683',target_paper_percent='15.73'),
        ])
    report,failures=mod.verify_table1(path)
    assert not failures
    assert all(row['match_to_2dp'] for row in report)


def test_dry_run_evidence(tmp_path, monkeypatch, capsys):
    data=tmp_path/'data'
    # Dry-run evidence should not require actual scene artifacts.
    assert mod.main(['--stage','evidence','--scenes','scene0','--dataset-root',str(data),
                     '--output-root',str(tmp_path/'out'),'--dry-run'])==0
    output=capsys.readouterr().out
    assert 'build_evidence.py' in output
    assert '--sample_stride 4' in output
    assert '--voxel_size 0.2' in output
