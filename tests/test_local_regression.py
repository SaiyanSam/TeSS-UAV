"""Optional local regression from the original result/ directory."""
from pathlib import Path
import csv,json,os
import numpy as np
import pytest
from tess_uav.evaluation.summary import summarize
ROOT=Path(os.environ.get('TESS_REFERENCE_ROOT','result'))
@pytest.mark.parametrize('scene',['scene0','scene1','scene3','scene5','scene8'])
def test_historical_metrics_csv(scene):
    csvpath=ROOT/'week9_repro_segearth'/scene/'voxel_metrics.csv'
    summpath=ROOT/'week9_repro_segearth'/scene/'stability_summary.json'
    if not csvpath.exists() or not summpath.exists():
        pytest.skip('Historical CSV/reference not present locally')
    with csvpath.open(newline='') as f:
        records=[{key:(float(row[key]) if row[key] else None) for key in ('vss','sbd','entropy_norm')} | {'op':int(row['op']),'num_observed_frames':int(row['num_observed_frames']),'dominant_class':int(row['dominant_class'])} for row in csv.DictReader(f)]
    expected=json.loads(summpath.read_text())
    actual=summarize(records,int(expected['num_classes']))
    for key,value in actual.items():
        if key=='class_voxel_counts':assert value==expected[key]
        elif value is not None: assert value==pytest.approx(expected[key],abs=1e-10)
