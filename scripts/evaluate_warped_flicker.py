#!/usr/bin/env python3
"""Evaluate geometry-aligned flicker between consecutive UAVid-3D frames."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from tess_uav.data.uavid3d import (collect_mask_files, collect_depth_files,
    get_pose_for_frame, infer_frame_id_from_mask, load_intrinsics, load_mask,
    load_poses, match_depth_file)
from tess_uav.geometry.depth import load_depth
from tess_uav.analysis.warped_flicker import warped_pair, summarize_pairs


def csv_write(path, rows):
    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_scene(args, scene):
    mask_dir = Path(args.mask_root) / scene
    if args.mask_dir_template:
        mask_dir = Path(args.mask_dir_template.format(scene=scene))
    depth_dir = Path(args.dataset_root) / 'depth' / scene
    K = load_intrinsics(Path(args.dataset_root)/'intrinsics.json', scene)
    poses = load_poses(Path(args.dataset_root)/'poses.json', scene)
    masks = collect_mask_files(mask_dir, args.mask_name)[:args.num_frames]
    depths = collect_depth_files(depth_dir, '.png')
    frames = []
    for mask_path in masks:
        fid = infer_frame_id_from_mask(mask_path, mask_dir)
        T, key = get_pose_for_frame(poses, fid)
        dp = match_depth_file(depths, fid, key)
        if dp is None:
            raise FileNotFoundError(f'Missing depth: {scene} {fid}')
        mask = load_mask(mask_path)
        depth = load_depth(dp)
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        frames.append((fid, mask, depth, T))
    if len(frames) < 2:
        raise RuntimeError(f'Not enough frames for {scene}')
    pairs = []
    for a,b in zip(frames[:-1],frames[1:]):
        stats = warped_pair(a[1], b[1], a[2], b[2], K, a[3], b[3],
                            sample_stride=args.sample_stride, ignore_label=args.ignore_label,
                            min_depth=args.min_depth, max_depth=args.max_depth,
                            depth_abs_tol=args.depth_abs_tol, depth_rel_tol=args.depth_rel_tol)
        pairs.append(dict(scene=scene, frame_1=a[0], frame_2=b[0], **stats))
    csv_write(Path(args.output_dir)/f'{scene}_pairwise.csv', pairs)
    return summarize_pairs(pairs, scene), pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_root', default='datasets/uavid_3d')
    p.add_argument('--mask_root', default='')
    p.add_argument('--mask_dir_template', default='result/week1_segearth_baseline_{scene}')
    p.add_argument('--mask_name', default='segearth_mask.png')
    p.add_argument('--scenes', nargs='+', default=['scene0','scene1','scene3','scene5','scene8'])
    p.add_argument('--output_dir', default='results/geometry_warped_flicker')
    p.add_argument('--reference_dir', default='result/rebuttal/geometry_warped_flicker')
    p.add_argument('--num_frames', type=int, default=20)
    p.add_argument('--sample_stride', type=int, default=4)
    p.add_argument('--ignore_label', type=int, default=0)
    p.add_argument('--min_depth', type=float, default=0.5)
    p.add_argument('--max_depth', type=float, default=120.0)
    p.add_argument('--depth_abs_tol', type=float, default=2.0)
    p.add_argument('--depth_rel_tol', type=float, default=0.05)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries=[]
    for scene in args.scenes:
        summ,pairs=run_scene(args,scene)
        summaries.append(summ)
        print(f"{scene}: {summ['total_matched_correspondences']:,} matched, "
              f"{summ['total_changed_correspondences']:,} changed, "
              f"{summ['pooled_warped_flicker_percent']:.6f}%")
    csv_write(out/'scene_summary.csv',summaries)
    matched=sum(s['total_matched_correspondences'] for s in summaries)
    changed=sum(s['total_changed_correspondences'] for s in summaries)
    result=dict(scenes=args.scenes, sample_stride=args.sample_stride,
                min_depth=args.min_depth, max_depth=args.max_depth,
                depth_abs_tol=args.depth_abs_tol,depth_rel_tol=args.depth_rel_tol,
                total_matched_correspondences=matched,
                total_changed_correspondences=changed,
                pooled_warped_flicker_percent=100.*changed/matched if matched else None,
                scene_summaries=summaries)
    (out/'overall_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Pooled: {matched:,} matched, {changed:,} changed, "
          f"{result['pooled_warped_flicker_percent']:.6f}%")
    refpath = Path(args.reference_dir)/'overall_summary.json' if args.reference_dir else None
    if refpath and refpath.is_file():
        ref=json.loads(refpath.read_text())
        print('\nReference comparison (diagnostic; not an automatic pass):')
        for name in ['total_matched_correspondences','total_changed_correspondences',
                     'pooled_warped_flicker_percent']:
            print(f'  {name}: reference={ref[name]}, new={result[name]}')
        for old,new in zip(ref['scene_summaries'],summaries):
            print(f"  {new['scene']}: reference matched={old['total_matched_correspondences']}, "
                  f"new matched={new['total_matched_correspondences']}; "
                  f"reference changed={old['total_changed_correspondences']}, "
                  f"new changed={new['total_changed_correspondences']}")
    print(f'Saved: {out}')

if __name__=='__main__':
    main()
