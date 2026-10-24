"""Geometry-aligned semantic disagreement between consecutive camera frames."""
import numpy as np


def valid_depth(depth, minimum=0.5, maximum=120.0):
    return np.isfinite(depth) & (depth > minimum) & (depth < maximum)


def warped_pair(mask_source, mask_target, depth_source, depth_target, K,
                pose_source, pose_target, sample_stride=4, ignore_label=0,
                min_depth=0.5, max_depth=120.0, depth_abs_tol=2.0,
                depth_rel_tol=0.05):
    """Forward-project stride-sampled source pixels into the target camera.

    Poses must be world-to-camera extrinsics. Pixels are nearest-neighbor
    sampled after projection; target metric depth gates correspondence validity.
    The count of matches is directional (source to target).
    """
    source = np.asarray(mask_source)
    target = np.asarray(mask_target)
    d1 = np.asarray(depth_source)
    d2 = np.asarray(depth_target)
    if source.shape != d1.shape or target.shape != d2.shape:
        raise ValueError('Mask and depth dimensions must match within each frame')
    if sample_stride <= 0:
        raise ValueError('sample_stride must be positive')
    if source.shape != target.shape:
        raw_valid = raw_changed = 0
    else:
        valid_raw = (valid_depth(d1, min_depth, max_depth) &
                     valid_depth(d2, min_depth, max_depth) &
                     (source != ignore_label) & (target != ignore_label))
        raw_valid = int(valid_raw.sum())
        raw_changed = int(np.count_nonzero(valid_raw & (source != target)))
    ys, xs = np.mgrid[0:d1.shape[0]:sample_stride, 0:d1.shape[1]:sample_stride]
    z = d1[ys, xs]
    keep = valid_depth(z, min_depth, max_depth) & (source[ys, xs] != ignore_label)
    u = xs[keep].astype(np.float64)
    v = ys[keep].astype(np.float64)
    z = z[keep].astype(np.float64)
    labs = source[ys, xs][keep]
    n = int(len(z))
    out = dict(raw_valid_pixels=raw_valid, raw_changed_pixels=raw_changed,
               raw_same_pixel_flicker_percent=100.0 * raw_changed/raw_valid if raw_valid else None,
               source_valid=n, positive_depth=0, inside_image=0,
               target_semantic_valid=0, depth_consistent=0, matched=0,
               changed=0, stable=0, warped_flicker_percent=None,
               matched_fraction_of_source_percent=None)
    if n == 0:
        return out
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p1 = np.stack(((u-cx)*z/fx, (v-cy)*z/fy, z, np.ones_like(z)), axis=0)
    transform = np.asarray(pose_target, dtype=np.float64) @ np.linalg.inv(np.asarray(pose_source, dtype=np.float64))
    p2 = transform @ p1
    z2 = p2[2]
    pos = np.isfinite(z2) & (z2 > 0)
    out['positive_depth'] = int(pos.sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        pu = np.rint(fx*p2[0]/z2+cx)
        pv = np.rint(fy*p2[1]/z2+cy)
    inside = (pos & np.isfinite(pu) & np.isfinite(pv) & (pu >= 0) &
              (pv >= 0) & (pu < target.shape[1]) & (pv < target.shape[0]))
    out['inside_image'] = int(inside.sum())
    idx = np.flatnonzero(inside)
    if idx.size:
        xx = pu[idx].astype(np.intp)
        yy = pv[idx].astype(np.intp)
        semantic = target[yy,xx] != ignore_label
        out['target_semantic_valid'] = int(semantic.sum())
        ii = idx[semantic]
        xx = xx[semantic]
        yy = yy[semantic]
        measured = d2[yy,xx]
        consistent = (valid_depth(measured, min_depth, max_depth) &
                      (np.abs(measured-z2[ii]) <= np.maximum(depth_abs_tol, depth_rel_tol*z2[ii])))
        out['depth_consistent'] = int(consistent.sum())
        out['matched'] = out['depth_consistent']
        out['changed'] = int(np.count_nonzero(labs[ii[consistent]] != target[yy[consistent], xx[consistent]]))
    out['stable'] = out['matched']-out['changed']
    if out['matched']:
        out['warped_flicker_percent'] = 100.0*out['changed']/out['matched']
    out['matched_fraction_of_source_percent'] = 100.0*out['matched']/n
    return out


def summarize_pairs(pairs, scene):
    if not pairs:
        raise ValueError('No valid frame pairs')
    def mean(key):
        values = [float(p[key]) for p in pairs if p[key] is not None]
        return float(np.mean(values)) if values else None
    matched = sum(int(p['matched']) for p in pairs)
    changed = sum(int(p['changed']) for p in pairs)
    percent = 100.0*changed/matched if matched else None
    return dict(scene=scene, num_pairs=len(pairs),
                mean_raw_same_pixel_flicker_percent=mean('raw_same_pixel_flicker_percent'),
                mean_warped_flicker_percent_unweighted=mean('warped_flicker_percent'),
                mean_warped_flicker_percent_weighted=percent,
                pooled_warped_flicker_percent=percent,
                total_source_valid=sum(int(p['source_valid']) for p in pairs),
                total_matched_correspondences=matched,
                total_changed_correspondences=changed)
