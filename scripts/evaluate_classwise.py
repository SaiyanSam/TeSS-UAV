#!/usr/bin/env python3
"""Generate class-conditioned stability and recurrence statistics."""
import argparse
import csv
import json
from pathlib import Path
from tess_uav.analysis.classwise import aggregate_classwise, write_rows, compare_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="results")
    parser.add_argument("--output-dir", default="results/analysis/classwise")
    parser.add_argument("--scenes", nargs="+", default=["scene0", "scene1", "scene3", "scene5", "scene8"])
    parser.add_argument("--reference-root", default=None, help="Optional saved classwise summary directory")
    args = parser.parse_args()
    paths = {s: Path(args.input_root) / s / "voxel_metrics.csv" for s in args.scenes}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        parser.error("Missing inputs: " + ", ".join(missing))
    pooled, scene_rows = aggregate_classwise(paths)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "classwise_recurrence_all.csv", pooled)
    write_rows(output / "classwise_recurrence_by_scene.csv", scene_rows)
    (output / "classwise_summary.json").write_text(json.dumps({
        "scenes": args.scenes, "input_root": args.input_root,
        "grouping": "dominant_class", "aggregation": "voxel-weighted across all scenes",
        "undefined_metric_handling": "exclude non-finite SBD/entropy/VSS; zero is valid",
        "classwise": pooled}, indent=2) + "\n")
    print("Saved classwise results to", output)
    for row in pooled:
        sbd = row["mean_sbd_op_ge_3"]
        print(f"{row['class_name']:16} voxels={row['voxel_count']:7} "
              f"OP>=3={row['op_ge_3_fraction_percent']:7.3f}% "
              f"VSS>=3={row['mean_vss_op_ge_3']} SBD>=3={sbd}")
    if args.reference_root:
        reference = Path(args.reference_root)
        checks = []
        for rows, filename in ((pooled, "classwise_recurrence_all.csv"),
                               (scene_rows, "classwise_recurrence_by_scene.csv")):
            path = reference / filename
            if not path.is_file():
                parser.error(f"Missing reference: {path}")
            checks.extend(compare_rows(rows, path))
        with (output / "comparison_with_reference.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene", "class_id", "metric", "reference", "candidate", "status"])
            writer.writeheader()
            writer.writerows(checks)
        failures = [c for c in checks if c["status"] != "PASS"]
        print(f"Reference comparison: {len(checks)-len(failures)}/{len(checks)} PASS")
        if failures:
            for row in failures[:20]:
                print("FAIL:", row)
            raise SystemExit(1)

if __name__ == "__main__":
    main()
