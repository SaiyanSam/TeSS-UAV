#!/usr/bin/env python3
"""Run reproducible pose/depth-noise experiments and compare archived summaries."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CONDITIONS = {
    'pose_very_mild': (0.01, 0.05, 0.0),
    'pose_mild_refined': (0.02, 0.1, 0.0),
    'pose_mild': (0.05, 0.5, 0.0),
    'pose_moderate': (0.1, 1.0, 0.0),
    'depth_very_mild': (0.0, 0.0, 0.001),
    'depth_mild_refined': (0.0, 0.0, 0.003),
    'depth_mild': (0.0, 0.0, 0.01),
    'depth_moderate': (0.0, 0.0, 0.03),
    'regression_clean': (0.0, 0.0, 0.0),
}
SCENES = ('scene0', 'scene1', 'scene3', 'scene5', 'scene8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', default='datasets/uavid_3d')
    p.add_argument('--historical-root', default='result')
    p.add_argument('--output-root', default='results/sensitivity/geometry')
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=[k for k in CONDITIONS if k != 'regression_clean'])
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=list(SCENES))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    failures = []
    for condition in args.conditions:
        for scene in args.scenes:
            trans, rot, depth_noise = CONDITIONS[condition]
            reference = Path(args.historical_root) / 'week9_geometry_noise' / condition / scene / 'stability_summary.json'
            if not reference.is_file():
                print(f'SKIP: no archived reference {reference}', flush=True)
                continue
            dest = Path(args.output_root) / condition / scene
            cmd = [sys.executable, 'scripts/evaluate_geometry_robustness.py',
                   '--mask_dir', f'{args.historical_root}/week1_segearth_baseline_{scene}',
                   '--mask_name', 'segearth_mask.png', '--backbone_name', 'segearth_ov',
                   '--depth_dir', f'{args.dataset_root}/depth/{scene}',
                   '--intrinsics', f'{args.dataset_root}/intrinsics.json',
                   '--poses_json', f'{args.dataset_root}/poses.json',
                   '--scene_id', scene, '--output_dir', str(dest),
                   '--num_frames', '20', '--sample_stride', '4', '--voxel_size', '0.2',
                   '--num_classes', '8', '--ignore_label', '0',
                   '--translation_noise', str(trans), '--rotation_noise_deg', str(rot),
                   '--depth_noise', str(depth_noise), '--noise_seed', str(args.seed)]
            print(f'===== {condition} / {scene} =====', flush=True)
            if subprocess.run(cmd).returncode:
                failures.append((condition, scene, 'execution'))
                continue
            candidate = dest / 'stability_summary.json'
            ref_data = json.loads(reference.read_text())
            new_data = json.loads(candidate.read_text())
            keys = ('num_voxels', 'used_frames', 'total_sampled_world_points',
                    'total_frame_voxel_observations', 'mean_op', 'mean_vss',
                    'mean_sbd', 'mean_entropy_norm')
            differences = {k: (ref_data.get(k), new_data.get(k)) for k in keys
                           if ((ref_data.get(k) is None) != (new_data.get(k) is None))
                           or (ref_data.get(k) is not None and new_data.get(k) is not None
                               and abs(ref_data[k] - new_data[k]) > 1e-10)}
            if differences:
                failures.append((condition, scene, differences))
                print(f'DIFFERENCES: {differences}', flush=True)
            else:
                print('PASS: 8/8 metrics', flush=True)
    print(f'Completed with {len(failures)} failed comparisons', flush=True)
    for f in failures:
        print(f)
    return bool(failures)

if __name__ == '__main__':
    sys.exit(main())
