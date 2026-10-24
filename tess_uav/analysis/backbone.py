"""Depth-conditioned semantic coverage for matched segmentation backbones."""
import csv
import math
from pathlib import Path
import cv2
import numpy as np

from tess_uav.data.uavid3d import collect_mask_files, collect_depth_files, infer_frame_id_from_mask, match_depth_file, load_mask
from tess_uav.geometry.depth import load_depth

COLUMNS = ['scene', 'frame_id', 'depth_path', 'sampled_pixels', 'valid_depth',
           'se_retained', 'sf_retained', 'both_retained', 'se_only', 'sf_only',
           'neither_retained', 'se_building', 'sf_building', 'se_road', 'sf_road',
           'se_vegetation', 'sf_vegetation', 'se_retention_given_valid_depth',
           'sf_retention_given_valid_depth']
COUNTS = COLUMNS[3:16]

def compute_coverage(depth, se_mask, sf_mask, scene, frame_id, depth_path, stride=4, min_depth=.5, max_depth=120.):
    h, w = depth.shape
    if se_mask.shape != depth.shape:
        se_mask = cv2.resize(se_mask, (w,h), interpolation=cv2.INTER_NEAREST)
    if sf_mask.shape != depth.shape:
        sf_mask = cv2.resize(sf_mask, (w,h), interpolation=cv2.INTER_NEAREST)
    z = depth[::stride, ::stride]
    se = se_mask[::stride, ::stride]
    sf = sf_mask[::stride, ::stride]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    s = valid & (se != 0)
    f = valid & (sf != 0)
    result = dict(scene=scene, frame_id=frame_id, depth_path=str(depth_path),
                  sampled_pixels=int(z.size), valid_depth=int(valid.sum()),
                  se_retained=int(s.sum()), sf_retained=int(f.sum()),
                  both_retained=int((s & f).sum()), se_only=int((s & ~f).sum()),
                  sf_only=int((f & ~s).sum()),
                  neither_retained=int((valid & ~s & ~f).sum()))
    for label, name in ((1, 'building'), (2, 'road'), (3, 'vegetation')):
        result['se_' + name] = int((s & (se == label)).sum())
        result['sf_' + name] = int((f & (sf == label)).sum())
    n = result['valid_depth']
    result['se_retention_given_valid_depth'] = result['se_retained']/n if n else None
    result['sf_retention_given_valid_depth'] = result['sf_retained']/n if n else None
    return result

def audit_scene(scene, se_dir, sf_dir, depth_dir, stride=4, limit=20):
    se_paths = collect_mask_files(se_dir, 'shared3_mask.png')
    sf_paths = collect_mask_files(sf_dir, 'baseline_mask.png')
    se_ids = {infer_frame_id_from_mask(p, se_dir): p for p in se_paths}
    sf_ids = {infer_frame_id_from_mask(p, sf_dir): p for p in sf_paths}
    if set(se_ids) != set(sf_ids):
        raise ValueError(f'Mismatched frame IDs in {scene}: SegEarth-only={set(se_ids)-set(sf_ids)}, SegFormer-only={set(sf_ids)-set(se_ids)}')
    frames = sorted(se_ids)[:limit]
    if len(frames) != limit:
        raise ValueError(f'{scene}: expected {limit} frames, found {len(frames)}')
    depths = collect_depth_files(depth_dir, '.png')
    rows = []
    for frame in frames:
        path = match_depth_file(depths, frame)
        if path is None:
            raise FileNotFoundError(f'Missing depth for {scene}/{frame}')
        rows.append(compute_coverage(load_depth(path), load_mask(se_ids[frame]),
                                     load_mask(sf_ids[frame]), scene, frame, path, stride))
    return rows

def aggregate(rows, label):
    result = {key: sum(int(row[key]) for row in rows) for key in COUNTS}
    n = result['valid_depth']
    result.update(scene=label, frames=len(rows),
                  se_retention_given_valid_depth=result['se_retained']/n if n else None,
                  sf_retention_given_valid_depth=result['sf_retained']/n if n else None,
                  both_retention_given_valid_depth=result['both_retained']/n if n else None,
                  se_only_given_valid_depth=result['se_only']/n if n else None,
                  sf_only_given_valid_depth=result['sf_only']/n if n else None,
                  sf_over_se_retained=result['sf_retained']/result['se_retained'] if result['se_retained'] else None,
                  retained_difference_se_minus_sf=result['se_retained']-result['sf_retained'],
                  retained_difference_fraction_of_se=(result['se_retained']-result['sf_retained'])/result['se_retained'] if result['se_retained'] else None)
    return result

def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def compare_coverage(candidate, reference, keys, atol=1e-9):
    with open(reference, newline='') as file:
        old = {(r['scene'], r.get('frame_id','')): r for r in csv.DictReader(file)}
    checks = []
    for row in candidate:
        tag = (row['scene'], row.get('frame_id',''))
        prior = old.get(tag)
        if prior is None:
            checks.append((tag, 'row', False, None, None))
            continue
        for key in keys:
            expected = float(prior[key])
            actual = float(row[key])
            ok = math.isclose(actual, expected, abs_tol=atol, rel_tol=1e-9)
            checks.append((tag, key, ok, expected, actual))
    return checks
