import csv
from pathlib import Path
import pytest
from tess_uav.analysis.classwise import aggregate_classwise, compare_rows

FIELDS = ["dominant_class", "op", "vss", "sbd", "entropy_norm"]

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def test_classwise_ignores_undefined_sbd(tmp_path):
    path = tmp_path / "voxels.csv"
    write_csv(path, [
        dict(dominant_class=7, op=1, vss=1.0, sbd="", entropy_norm=0.0),
        dict(dominant_class=7, op=2, vss=0.5, sbd=0.2, entropy_norm=0.5),
        dict(dominant_class=7, op=4, vss=0.75, sbd=0.1, entropy_norm=0.2),
        dict(dominant_class=1, op=1, vss=1, sbd="nan", entropy_norm=0),
    ])
    pooled, per_scene = aggregate_classwise({"scene0": path})
    car = next(x for x in pooled if x["class_id"] == 7)
    assert car["voxel_count"] == 3
    assert car["op_ge_3_count"] == 1
    assert car["op_ge_3_fraction_percent"] == pytest.approx(100/3)
    assert car["mean_sbd_op_ge_2"] == pytest.approx(0.15)
    assert car["mean_sbd_op_ge_3"] == pytest.approx(0.1)
    assert car["median_op"] == 2
    building = next(x for x in pooled if x["class_id"] == 1)
    assert building["mean_sbd_op_ge_3"] is None
    assert len(per_scene) == 2

def test_classwise_reference_fixture(tmp_path):
    path = tmp_path / "voxels.csv"
    write_csv(path, [dict(dominant_class=1, op=3, vss=0.8, sbd=0.12, entropy_norm=0.3)])
    rows, _ = aggregate_classwise({"scene0": path})
    ref = tmp_path / "reference.csv"
    with ref.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert all(r["status"] == "PASS" for r in compare_rows(rows, ref))
