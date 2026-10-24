"""Same-pixel temporal flicker used for Table 1 (not geometry-warped)."""
from pathlib import Path
import csv
import numpy as np
from tess_uav.data.uavid3d import collect_mask_files, load_mask


def pairwise_flicker(first, second, ignore_label=0):
    a, b = np.asarray(first), np.asarray(second)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Masks must have identical 2D shapes")
    valid = (a != ignore_label) | (b != ignore_label)
    total = int(np.count_nonzero(valid))
    changed = int(np.count_nonzero(valid & (a != b)))
    if total == 0:
        raise ValueError("No valid pixels for this frame pair")
    return dict(valid_pixels=total, changed_pixels=changed,
                flicker_percent=100.0 * changed / total)


def evaluate_sequence(mask_dir, mask_name, num_frames=20, ignore_label=0):
    paths = collect_mask_files(mask_dir, mask_name)
    if len(paths) < num_frames:
        raise ValueError(f"Expected {num_frames} masks, found {len(paths)} at {mask_dir}")
    paths = paths[:num_frames]
    ids = [p.parent.name if p.parent != Path(mask_dir) else p.stem for p in paths]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate frame IDs")
    rows = []
    previous = load_mask(paths[0])
    for idx in range(1, len(paths)):
        current = load_mask(paths[idx])
        stats = pairwise_flicker(previous, current, ignore_label)
        rows.append(dict(frame_pair=f"{ids[idx-1]}-{ids[idx]}",
                         frame_1=ids[idx-1], frame_2=ids[idx], **stats))
        previous = current
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    if not rows:
        raise ValueError("No frame pairs")
    # Paper Table 1: arithmetic mean of pairwise percentages, NOT count-pooled.
    return dict(num_pairs=len(rows), mean_flicker_percent=float(np.mean(
        [r["flicker_percent"] for r in rows])),
        total_valid_pixels=sum(r["valid_pixels"] for r in rows),
        total_changed_pixels=sum(r["changed_pixels"] for r in rows))
