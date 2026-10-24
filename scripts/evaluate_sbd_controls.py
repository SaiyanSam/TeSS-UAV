#!/usr/bin/env python3
"""Evaluate cumulative, shuffled, and direct semantic evidence drift."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from tess_uav.data.uavid3d import (load_intrinsics, load_poses, get_pose_for_frame,
    collect_mask_files, collect_depth_files, infer_frame_id_from_mask,
    match_depth_file, load_mask)
from tess_uav.geometry.depth import load_depth
from tess_uav.geometry.projection import backproject, transform_points
from tess_uav.analysis.sbd_controls import accumulate_temporal_evidence, evaluate_controls


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mask_dir', required=True)
    p.add_argument('--mask_name', default='segearth_mask.png')
    p.add_argument('--backbone_name', default='segearth_ov')
    p.add_argument('--depth_dir', required=True)
    p.add_argument('--intrinsics', required=True)
    p.add_argument('--poses_json', required=True)
    p.add_argument('--scene_id', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--num_frames', type=int, default=20)
    p.add_argument('--sample_stride', type=int, default=4)
    p.add_argument('--voxel_size', type=float, default=0.2)
    p.add_argument('--num_classes', type=int, default=8)
    p.add_argument('--ignore_label', type=int, default=0)
    p.add_argument('--depth_scale', type=float, default=1.0)
    p.add_argument('--max_depth', type=float, default=120.0)
    p.add_argument('--depth_ext', default='.png')
    p.add_argument('--num_permutations', type=int, default=20)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--reference_dir', default=None, help='Optional saved control outputs for strict comparison')
    p.add_argument('--comparison_tolerance', type=float, default=1e-10)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    K = load_intrinsics(args.intrinsics, args.scene_id)
    poses = load_poses(args.poses_json, args.scene_id)
    masks = collect_mask_files(args.mask_dir, args.mask_name)
    depths = collect_depth_files(args.depth_dir, args.depth_ext)
    voxels, used, skipped, audits = {}, [], [], []
    points_count = observations_count = 0
    for idx, mask_path in enumerate(tqdm(masks[:args.num_frames], desc=f'Evidence {args.scene_id}')):
        frame_id = infer_frame_id_from_mask(mask_path, args.mask_dir)
        audit = dict(frame_index=idx, frame_id=frame_id, mask_path=str(mask_path), depth_path='',
                     sampled_world_points=0, frame_voxel_observations=0, status='skipped', error='')
        try:
            T_cam_world, pose_key = get_pose_for_frame(poses, frame_id)
            depth_path = match_depth_file(depths, frame_id, pose_key)
            if depth_path is None:
                raise FileNotFoundError(f'Missing depth for {frame_id}')
            audit['depth_path'] = str(depth_path)
            mask = load_mask(mask_path)
            depth = load_depth(depth_path, depth_scale=args.depth_scale)
            if mask.shape != depth.shape:
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            camera_points, labels = backproject(depth, mask, K, sample_stride=args.sample_stride,
                 max_depth=args.max_depth, ignore_label=args.ignore_label)
            if camera_points.shape[0] == 0:
                raise RuntimeError('No valid semantic points')
            world = transform_points(camera_points, np.linalg.inv(T_cam_world))
            accepted = accumulate_temporal_evidence(world, labels, frame_id,
                                                     args.voxel_size, args.num_classes, voxels)
            audit.update(sampled_world_points=len(world), frame_voxel_observations=accepted, status='used')
            points_count += len(world)
            observations_count += accepted
            used.append(frame_id)
        except Exception as exc:
            audit['error'] = str(exc)
            skipped.append(dict(frame_id=frame_id, error=str(exc)))
            print(f'[WARN] {frame_id}: {exc}')
        audits.append(audit)
    if not voxels:
        raise RuntimeError('No evidence voxels built')
    print(f'Computing controls for {len(voxels):,} voxels with seed={args.seed}')
    records, bins, max_error, failures = evaluate_controls(voxels, args.num_permutations, args.seed)
    write_csv(out / 'sbd_controls_voxel_metrics.csv', records)
    write_csv(out / 'sbd_controls_op_bins.csv', bins)
    write_csv(out / 'frame_audit.csv', audits)
    summary = dict(scene_id=args.scene_id, backbone_name=args.backbone_name,
        mask_dir=args.mask_dir, mask_name=args.mask_name, depth_dir=args.depth_dir,
        voxel_size=args.voxel_size, sample_stride=args.sample_stride,
        num_classes=args.num_classes, ignore_label=args.ignore_label,
        requested_frames=args.num_frames, used_frames=len(used), used_frame_ids=used,
        skipped_frames=skipped, num_voxels=len(records),
        num_recurrent_voxels_op_ge_2=sum(r['op']>=2 for r in records),
        fraction_recurrent_voxels_op_ge_2=sum(r['op']>=2 for r in records)/len(records),
        total_sampled_world_points=points_count,
        total_frame_voxel_observations=observations_count,
        num_permutations=args.num_permutations, seed=args.seed,
        regression_tolerance=1e-12, max_regression_error=max_error,
        regression_failures=failures, op_bin_results=bins)
    (out / 'sbd_controls_summary.json').write_text(json.dumps(summary, indent=2))
    for row in bins:
        print(f"{row['op_group']:8s} n={row['num_voxels']:7d} "
              f"cumulative={row['cum_sbd']} shuffled={row['shuffled_cum_sbd']} direct={row['direct_sbd']}")
    if args.reference_dir:
        ref = json.loads((Path(args.reference_dir)/'sbd_controls_summary.json').read_text())
        check = ['num_voxels','num_recurrent_voxels_op_ge_2',
                 'total_sampled_world_points','total_frame_voxel_observations','used_frames']
        cols = ['num_voxels','mean_op','cum_sbd','cum_sbd_median','shuffled_cum_sbd',
                'shuffled_cum_sbd_median','direct_sbd','direct_sbd_median',
                'cum_minus_shuffled','cum_over_shuffled']
        failures_ref = []
        for name in check:
            if summary[name] != ref[name]:
                failures_ref.append(f'{name}: {summary[name]} != {ref[name]}')
        reference_bins = {r['op_group']:r for r in ref['op_bin_results']}
        for row in bins:
            old = reference_bins[row['op_group']]
            for name in cols:
                a, b = row.get(name), old.get(name)
                if a is None or b is None:
                    same = a is None and b is None
                elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
                    same = bool(np.isclose(a,b,atol=args.comparison_tolerance,rtol=0))
                else:
                    same = a == b
                if not same:
                    failures_ref.append(f"{row['op_group']} {name}: {a} != {b}")
        if failures_ref:
            print('REFERENCE MISMATCH:', *failures_ref[:30], sep='\n  ')
            raise SystemExit(1)
        print(f'REFERENCE PASS: {len(check) + len(cols)*len(bins)} scene-level checks')
    print(f'Saved: {out}')

if __name__ == '__main__':
    main()
