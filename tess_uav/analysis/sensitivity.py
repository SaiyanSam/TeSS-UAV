"""Summaries and reference comparisons for semantic-memory sensitivity sweeps."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

SCENES = ('scene0','scene1','scene3','scene5','scene8')
KEYS = ('num_voxels','mean_op','mean_vss','mean_sbd','mean_entropy_norm','used_frames')


def read_summary(path):
    with Path(path).open() as handle:
        return json.load(handle)


def compare_summaries(candidate, reference, *, atol=1e-10):
    """Compare numeric output fields without comparing machine-specific input paths."""
    checks = {}
    for key in KEYS:
        a, b = candidate.get(key), reference.get(key)
        if a is None or b is None:
            checks[key] = a is b
        elif isinstance(a, (float,int)) and isinstance(b, (float,int)):
            checks[key] = math.isclose(a,b,rel_tol=0,abs_tol=atol)
        else:
            checks[key] = a == b
    return checks


def read_voxel_csv(path):
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            yield {'op':int(row['op']), 'vss':_finite(row.get('vss')),
                   'sbd':_finite(row.get('sbd')), 'entropy_norm':_finite(row.get('entropy_norm'))}


def _finite(value):
    if value is None or value.strip().lower() in ('','none','nan'):
        return None
    x=float(value)
    return x if math.isfinite(x) else None


def summarize_metrics(paths):
    """Pool voxel-level metrics, excluding undefined drift from its denominator."""
    n=op2=op3=0
    sums = {key:0. for key in ('op','vss','sbd','entropy_norm')}
    counts = {key:0 for key in sums}
    for path in paths:
        for row in read_voxel_csv(path):
            n+=1
            op2+=row['op']>=2
            op3+=row['op']>=3
            for key in sums:
                value=row[key]
                if value is not None:
                    sums[key]+=value
                    counts[key]+=1
    return {'voxels':n,'op_ge_2':op2,'op_ge_3':op3,
            'op_ge_2_fraction':op2/n if n else None,
            'op_ge_3_fraction':op3/n if n else None,
            **{f'mean_{key}':sums[key]/counts[key] if counts[key] else None for key in sums},
            'sbd_defined_voxels':counts['sbd']}


def write_csv(path, rows):
    rows=list(rows)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
