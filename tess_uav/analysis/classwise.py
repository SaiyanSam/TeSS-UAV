"""Voxel-weighted semantic stability statistics grouped by dominant class."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from tess_uav.analysis.persistence import parse_optional

CLASS_NAMES = {0: "background", 1: "building", 2: "road", 3: "static_car",
               4: "tree", 5: "low_vegetation", 6: "human", 7: "moving_car"}
FIELDS = ["scene", "class_id", "class_name", "voxel_count", "mean_op", "median_op", "max_op",
          "op_ge_2_count", "op_ge_2_fraction_percent", "op_ge_3_count", "op_ge_3_fraction_percent",
          "op_gt_5_count", "op_gt_5_fraction_percent", "mean_vss_all", "mean_vss_op_ge_2",
          "mean_sbd_op_ge_2", "mean_entropy_op_ge_2", "mean_vss_op_ge_3",
          "mean_sbd_op_ge_3", "mean_entropy_op_ge_3"]

class ClassStats:
    def __init__(self):
        self.ops = []
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, op, vss, sbd, entropy):
        self.ops.append(op)
        for group, enabled in (("all", True), ("ge2", op >= 2), ("ge3", op >= 3)):
            if not enabled:
                continue
            for label, val in (("vss", vss), ("sbd", sbd), ("entropy", entropy)):
                if val is not None and math.isfinite(val):
                    key = (group, label)
                    self.sums[key] += val
                    self.counts[key] += 1

    def row(self, scene, class_id):
        op = self.ops
        n = len(op)
        def mean(group, key):
            k = (group, key)
            return self.sums[k] / self.counts[k] if self.counts[k] else None
        def count_fraction(pred):
            k = sum(bool(pred(o)) for o in op)
            return k, 100.0 * k / n
        n2, f2 = count_fraction(lambda x: x >= 2)
        n3, f3 = count_fraction(lambda x: x >= 3)
        n5, f5 = count_fraction(lambda x: x > 5)
        return dict(scene=scene, class_id=class_id, class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    voxel_count=n, mean_op=sum(op)/n, median_op=median(op), max_op=max(op),
                    op_ge_2_count=n2, op_ge_2_fraction_percent=f2,
                    op_ge_3_count=n3, op_ge_3_fraction_percent=f3,
                    op_gt_5_count=n5, op_gt_5_fraction_percent=f5,
                    mean_vss_all=mean("all", "vss"), mean_vss_op_ge_2=mean("ge2", "vss"),
                    mean_sbd_op_ge_2=mean("ge2", "sbd"), mean_entropy_op_ge_2=mean("ge2", "entropy"),
                    mean_vss_op_ge_3=mean("ge3", "vss"), mean_sbd_op_ge_3=mean("ge3", "sbd"),
                    mean_entropy_op_ge_3=mean("ge3", "entropy"))

def aggregate_classwise(paths):
    """Return (pooled, per_scene) rows from scene-to-CSV mapping."""
    pooled = defaultdict(ClassStats)
    by_scene = []
    for scene, filename in paths.items():
        groups = defaultdict(ClassStats)
        with Path(filename).open(newline="") as f:
            for r in csv.DictReader(f):
                cls = int(r["dominant_class"])
                op = int(r["op"])
                if op < 1:
                    raise ValueError(f"Invalid OP={op} in {filename}")
                values = (op, parse_optional(r["vss"]), parse_optional(r["sbd"]),
                          parse_optional(r["entropy_norm"]))
                groups[cls].add(*values)
                pooled[cls].add(*values)
        by_scene.extend(groups[c].row(scene, c) for c in sorted(groups))
    combined = [pooled[c].row("all_scenes", c) for c in sorted(pooled)]
    return combined, by_scene

def write_rows(filename, rows):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def compare_rows(current, reference, atol=1e-9):
    """Compare numerically against an independently saved classwise reference."""
    with Path(reference).open(newline="") as f:
        previous = {(r["scene"], int(r["class_id"])): r for r in csv.DictReader(f)}
    report = []
    for row in current:
        key = row["scene"], int(row["class_id"])
        old = previous.get(key)
        if old is None:
            report.append(dict(scene=key[0], class_id=key[1], metric="row", status="MISSING"))
            continue
        for col in FIELDS:
            if col in ("scene", "class_id", "class_name"):
                continue
            expected = parse_optional(old.get(col))
            actual = row[col]
            ok = (expected is None and actual is None) or (
                expected is not None and actual is not None and
                math.isclose(expected, actual, abs_tol=atol, rel_tol=1e-9))
            report.append(dict(scene=key[0], class_id=key[1], metric=col,
                               reference=expected, candidate=actual, status="PASS" if ok else "FAIL"))
    return report
