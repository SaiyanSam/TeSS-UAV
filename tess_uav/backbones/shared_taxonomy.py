"""Common building/road/vegetation taxonomy for backbone comparisons."""
from pathlib import Path
import cv2
import numpy as np

SCENES = ('scene0', 'scene1', 'scene3', 'scene5', 'scene8')

def remap_segearth(mask):
    a = np.asarray(mask)
    if a.ndim != 2:
        raise ValueError('Expected single-channel semantic mask')
    out = np.zeros(a.shape, dtype=np.uint8)
    out[a == 1] = 1
    out[a == 2] = 2
    out[(a == 4) | (a == 5)] = 3
    return out

def remap_cityscapes(prediction, id2label):
    a = np.asarray(prediction)
    if a.ndim != 2:
        raise ValueError('Expected single-channel semantic mask')
    out = np.zeros(a.shape, dtype=np.uint8)
    labels = {'building': 1, 'road': 2, 'vegetation': 3}
    for key, name in id2label.items():
        target = labels.get(str(name).lower())
        if target is not None:
            out[a == int(key)] = target
    return out

def convert_segearth_scene(source_root, output_root, scene):
    source = Path(source_root) / f'week1_segearth_baseline_{scene}'
    destination = Path(output_root) / scene
    masks = sorted(source.glob('*/segearth_mask.png'))
    if not masks:
        raise FileNotFoundError(f'No SegEarth masks: {source}')
    for path in masks:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f'Unable to read: {path}')
        mapped = remap_segearth(mask)
        output = destination / path.parent.name / 'shared3_mask.png'
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), mapped):
            raise IOError(f'Unable to write: {output}')
    return len(masks)
