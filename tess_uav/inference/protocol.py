"""Explicit dataset/frame protocol, independent of historical results."""
from pathlib import Path

SCENE_TO_SEQUENCE = {
    "scene0": "seq029", "scene1": "seq730", "scene3": "seq122",
    "scene5": "seq14", "scene8": "seq142",
}
SEGEARTH_CLASSES = (
    "background clutter", "building", "road", "static car", "tree",
    "low vegetation", "human", "moving car",
)

def frame_ids(num_frames=20, start=0, step=4):
    if num_frames < 1 or step < 1 or start < 0:
        raise ValueError("Require num_frames>=1, step>=1, start>=0")
    return [f"{start + i*step:05d}" for i in range(num_frames)]

def rgb_frames(dataset_root, scene, num_frames=20, start=0, step=4):
    sequence = SCENE_TO_SEQUENCE[scene]
    root = Path(dataset_root) / "undistorted_rgb" / scene
    pairs = [(i, root / f"{sequence}_{i}.png") for i in frame_ids(num_frames, start, step)]
    missing = [str(p) for _, p in pairs if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing source frames; first: {missing[0]}")
    return pairs
