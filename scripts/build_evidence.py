#!/usr/bin/env python3
"""Construct persistent metric-voxel evidence from semantic masks and depth."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from tess_uav.data.uavid3d import (load_intrinsics,load_poses,get_pose_for_frame,collect_mask_files,collect_depth_files,infer_frame_id_from_mask,match_depth_file,load_mask)
from tess_uav.geometry.depth import load_depth
from tess_uav.geometry.projection import backproject,transform_points
from tess_uav.fusion.evidence import accumulate_frame_evidence
from tess_uav.evaluation.summary import summarize
from tess_uav.io.outputs import save_records_json,save_metrics_csv,save_voxel_ply
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask_dir", required=True)
    parser.add_argument(
        "--mask_name",
        default=None,
        help=(
            "Filename inside each per-frame mask directory. "
            "Examples: segearth_mask.png, baseline_mask.png, shared3_mask.png. "
            "If omitted, the original SegEarth discovery behavior is used."
        ),
    )
    parser.add_argument(
        "--backbone_name",
        default="unknown",
        help="Descriptive name stored in the output summary, e.g. segearth_ov or segformer.",
    )
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--poses_json", required=True)
    parser.add_argument("--scene_id", default="scene0")
    parser.add_argument("--output_dir", default="results/scene0")

    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--sample_stride", "--stride", dest="sample_stride", type=int, default=4)
    parser.add_argument("--voxel_size", type=float, default=0.2)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument(
        "--entropy_num_classes",
        type=int,
        default=None,
        help=(
            "Number of actually observable semantic classes used to normalize entropy. "
            "Default=None preserves the original behavior and uses --num_classes. "
            "For shared taxonomy [0=ignore,1=building,2=road,3=vegetation], use 3."
        ),
    )

    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--max_depth", type=float, default=120.0)
    parser.add_argument("--depth_ext", default=".png")
    parser.add_argument("--ignore_label", type=int, default=0)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    K = load_intrinsics(args.intrinsics, args.scene_id)
    poses = load_poses(args.poses_json, args.scene_id)

    mask_files = collect_mask_files(args.mask_dir, args.mask_name)
    depth_files = collect_depth_files(args.depth_dir, args.depth_ext)

    num_frames = min(args.num_frames, len(mask_files))

    print("=" * 80)
    print("TeSS-UAV - Evidence Voxel Framework")
    print("=" * 80)
    print(f"Scene          : {args.scene_id}")
    print(f"Backbone       : {args.backbone_name}")
    print(f"Mask dir       : {args.mask_dir}")
    print(f"Mask name      : {args.mask_name}")
    print(f"Depth dir      : {args.depth_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Frames         : {num_frames}")
    print(f"Voxel size     : {args.voxel_size}")
    print(f"Sample stride  : {args.sample_stride}")
    print(f"Num classes    : {args.num_classes}")
    print(f"Entropy classes: {args.entropy_num_classes}")
    print(f"Ignore label   : {args.ignore_label}")
    print("=" * 80)

    voxels = {}
    used_frames = []
    skipped_frames = []

    total_points = 0
    total_frame_voxel_observations = 0

    for idx in tqdm(range(num_frames), desc="Building evidence voxels"):
        mask_path = mask_files[idx]
        frame_id = infer_frame_id_from_mask(mask_path, args.mask_dir)

        try:
            T_cam_world, pose_key = get_pose_for_frame(poses, frame_id)
            T_world_cam = np.linalg.inv(T_cam_world)

            depth_path = match_depth_file(depth_files, frame_id, pose_key)

            if depth_path is None:
                raise FileNotFoundError(
                    f"No matching depth for frame={frame_id}, pose_key={pose_key}"
                )

            mask = load_mask(mask_path)
            depth = load_depth(depth_path, depth_scale=args.depth_scale)

            if mask.shape != depth.shape:
                mask = cv2.resize(
                    mask,
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            points_cam, labels = backproject(
                depth=depth,
                mask=mask,
                K=K,
                sample_stride=args.sample_stride,
                max_depth=args.max_depth,
                ignore_label=args.ignore_label,
            )

            if points_cam.shape[0] == 0:
                raise RuntimeError("No valid semantic points after backprojection.")

            points_world = transform_points(points_cam, T_world_cam)

            accepted = accumulate_frame_evidence(
                points_world=points_world,
                labels=labels,
                frame_id=frame_id,
                voxel_size=args.voxel_size,
                num_classes=args.num_classes,
                voxels=voxels,
                entropy_num_classes=args.entropy_num_classes,
            )

            total_points += int(points_world.shape[0])
            total_frame_voxel_observations += int(accepted)
            used_frames.append(frame_id)

        except Exception as e:
            skipped_frames.append(
                {
                    "frame_id": frame_id,
                    "mask_path": str(mask_path),
                    "error": str(e),
                }
            )
            print(f"[WARN] Skipping frame {frame_id}: {e}")

    if len(voxels) == 0:
        raise RuntimeError("No evidence voxels created. Check masks, depth, poses, and labels.")

    records = [v.to_record() for v in voxels.values()]
    records = sorted(records, key=lambda r: (r["key"][0], r["key"][1], r["key"][2]))

    summary = summarize(records, num_classes=args.num_classes)
    summary.update(
        {
            "scene_id": args.scene_id,
            "backbone_name": str(args.backbone_name),
            "mask_dir": str(args.mask_dir),
            "mask_name": args.mask_name,
            "num_classes": int(args.num_classes),
            "entropy_num_classes": (
                None if args.entropy_num_classes is None
                else int(args.entropy_num_classes)
            ),
            "depth_dir": str(args.depth_dir),
            "used_frames": len(used_frames),
            "used_frame_ids": used_frames,
            "skipped_frames": skipped_frames,
            "voxel_size": float(args.voxel_size),
            "sample_stride": int(args.sample_stride),
            "total_sampled_world_points": int(total_points),
            "total_frame_voxel_observations": int(total_frame_voxel_observations),
        }
    )

    records_json = output_dir / "evidence_voxel_records.json"
    metrics_csv = output_dir / "voxel_metrics.csv"
    summary_json = output_dir / "stability_summary.json"
    ply_path = output_dir / "evidence_voxels_dominant.ply"

    save_records_json(records_json, records)
    save_metrics_csv(metrics_csv, records)
    save_records_json(summary_json, summary)
    save_voxel_ply(ply_path, records)

    print("\n" + "=" * 80)
    print("Evidence Voxel Stability Summary")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    print(f"  {records_json}")
    print(f"  {metrics_csv}")
    print(f"  {summary_json}")
    print(f"  {ply_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
