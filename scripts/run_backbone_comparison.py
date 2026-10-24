#!/usr/bin/env python3
"""Prepare matched three-class predictions and compare their 3D evidence."""
import argparse
import subprocess
import sys
from pathlib import Path
from tess_uav.backbones.shared_taxonomy import SCENES, convert_segearth_scene
from tess_uav.analysis.backbone import audit_scene, aggregate, save_csv, compare_coverage, COUNTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', default='datasets/uavid_3d')
    ap.add_argument('--historical-root', default='result')
    ap.add_argument('--output-root', default='results/backbone_comparison')
    ap.add_argument('--scenes', nargs='+', default=list(SCENES))
    ap.add_argument('--reference-root', default='result')
    ap.add_argument('--skip-fusion', action='store_true')
    args = ap.parse_args()
    root, data, out = Path(args.historical_root), Path(args.dataset_root), Path(args.output_root)
    se_root = out / 'masks' / 'segearth'
    sf_root = root / 'rebuttal/segformer_multiscene'
    audit_rows = []
    summary_rows = []
    for scene in args.scenes:
        n = convert_segearth_scene(root, se_root, scene)
        print(f'{scene}: prepared {n} SegEarth shared-taxonomy masks', flush=True)
        sf_scene = sf_root / scene
        if not sf_scene.exists():
            raise FileNotFoundError(f'Missing SegFormer masks: {sf_scene}')
        rows = audit_scene(scene, se_root / scene, sf_scene, data / 'depth' / scene)
        audit_rows.extend(rows)
        summary_rows.append(aggregate(rows, scene))
        if not args.skip_fusion:
            for name, mask_dir, mask_name in (
                ('segearth', se_root / scene, 'shared3_mask.png'),
                ('segformer', sf_scene, 'baseline_mask.png')):
                command = [sys.executable, 'scripts/build_evidence.py',
                           '--mask_dir', str(mask_dir), '--mask_name', mask_name,
                           '--backbone_name', name + '_shared3',
                           '--depth_dir', str(data / 'depth' / scene),
                           '--intrinsics', str(data / 'intrinsics.json'),
                           '--poses_json', str(data / 'poses.json'),
                           '--scene_id', scene,
                           '--output_dir', str(out / 'voxels' / name / scene),
                           '--num_frames', '20', '--sample_stride', '4',
                           '--voxel_size', '0.2', '--num_classes', '4',
                           '--entropy_num_classes', '3', '--ignore_label', '0']
                subprocess.run(command, check=True)
    save_csv(out / 'coverage_frame_audit.csv', audit_rows)
    save_csv(out / 'coverage_scene_summary.csv', summary_rows)
    overall = aggregate(audit_rows, 'all_scenes')
    save_csv(out / 'coverage_overall.csv', [overall])
    ref_root = Path(args.reference_root) / 'rebuttal/backbone_recurrence_audit'
    if set(args.scenes) == set(SCENES):
        comparisons = []
        for current, reference, keys in (
            (audit_rows, ref_root / 'backbone_coverage_frame_audit.csv', COUNTS),
            (summary_rows, ref_root / 'backbone_coverage_scene_summary.csv', COUNTS)):
            if reference.exists():
                comparisons.extend(compare_coverage(current, reference, keys))
        bad = [item for item in comparisons if not item[2]]
        print(f'Coverage comparison: {len(comparisons)-len(bad)}/{len(comparisons)} PASS')
        for item in bad[:15]:
            print('FAIL:', item)
        if bad:
            raise SystemExit(1)
    print('Coverage:', overall['se_retention_given_valid_depth'], overall['sf_retention_given_valid_depth'])
    print('Saved:', out)

if __name__ == '__main__':
    main()
