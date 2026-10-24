"""Streaming voxel-weighted persistence analysis; no voxel histories required.

SBD undefined entries are *excluded*, never converted to zero. Fractions use
all valid voxel rows. Scene-mean and pooled-voxel means are distinct.
"""
import csv
import math
from pathlib import Path

GROUPS = {
    "OP=1": lambda op: op == 1,
    "OP=2": lambda op: op == 2,
    "3<=OP<=5": lambda op: 3 <= op <= 5,
    "OP>5": lambda op: op > 5,
    "OP>=2": lambda op: op >= 2,
    "OP>=3": lambda op: op >= 3,
}
FIELDNAMES = ["model", "scene", "group", "voxels", "fraction", "mean_op", "vss", "sbd", "entropy", "sbd_defined", "sbd_undefined", "total_voxels"]

class Bucket:
    def __init__(self):
        self.count = 0
        self.totals = {k: 0.0 for k in ('mean_op', 'vss', 'sbd', 'entropy')}
        self.valid = {k: 0 for k in self.totals}

    def add(self, op, vss, sbd, entropy):
        self.count += 1
        for key, value in [('mean_op', op), ('vss', vss), ('sbd', sbd), ('entropy', entropy)]:
            if value is not None and math.isfinite(value):
                self.totals[key] += value
                self.valid[key] += 1

    def export(self, model, scene, name, total):
        def avg(key):
            return self.totals[key] / self.valid[key] if self.valid[key] else None
        return dict(model=model, scene=scene, group=name, voxels=self.count,
                    fraction=(self.count/total if total else None),
                    mean_op=avg('mean_op'), vss=avg('vss'), sbd=avg('sbd'),
                    entropy=avg('entropy'), sbd_defined=self.valid['sbd'],
                    sbd_undefined=self.count-self.valid['sbd'], total_voxels=total)


def parse_optional(value):
    if value is None or str(value).strip().lower() in ('', 'none', 'nan', 'null'):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def aggregate(inputs, model='SegEarth-OV'):
    """inputs: mapping of scene -> voxel_metrics.csv; one streaming pass per file."""
    pooled = {key: Bucket() for key in GROUPS}
    by_scene = []
    total = 0
    for scene, file_path in inputs.items():
        buckets = {key: Bucket() for key in GROUPS}
        scene_total = 0
        with Path(file_path).open(newline='') as handle:
            for raw in csv.DictReader(handle):
                op = int(raw['op'])
                if op < 1:
                    raise ValueError(f'Invalid OP={op} in {file_path}')
                values = (op, parse_optional(raw['vss']), parse_optional(raw['sbd']), parse_optional(raw['entropy_norm']))
                for name, predicate in GROUPS.items():
                    if predicate(op):
                        buckets[name].add(*values)
                        pooled[name].add(*values)
                scene_total += 1
        total += scene_total
        by_scene.extend(b.export(model, scene, name, scene_total) for name,b in buckets.items())
    combined = [b.export(model, 'pooled', name, total) for name,b in pooled.items()]
    return by_scene, combined


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def compare_historical(current, historical, model='SegEarth-OV'):
    """Diagnostic only; never treats historical shared3 as equivalent to 8-class."""
    hist = {}
    with Path(historical).open(newline='') as f:
        for r in csv.DictReader(f):
            if r.get('model') == model:
                hist[r['group']] = r
    comparisons = []
    for row in current:
        other = hist.get(row['group'])
        if other is None:
            continue
        item = {'group': row['group'], 'new_population': row['total_voxels'],
                'historical_population': sum(int(hist[g]['voxels']) for g in ('OP=1','OP=2','3<=OP<=5','OP>5')),
                'new_voxels': row['voxels'], 'historical_voxels': int(other['voxels'])}
        for new_key, old_key in [('vss','vss'), ('sbd','sbd'), ('entropy','entropy')]:
            value = parse_optional(other[old_key]); new = row[new_key]
            item[f'new_{new_key}'] = new
            item[f'historical_{new_key}'] = value
            item[f'delta_{new_key}'] = new-value if new is not None and value is not None else None
        comparisons.append(item)
    return comparisons
