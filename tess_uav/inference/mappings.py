"""Paper-faithful conversion of raw backbone outputs to saved class IDs."""
import numpy as np

def cityscapes_to_three(raw_pred, id2label):
    out = np.zeros(raw_pred.shape, dtype=np.uint8)
    names = {"building":1, "road":2, "vegetation":3}
    for index, name in id2label.items():
        label = names.get(str(name).lower())
        if label is not None:
            out[raw_pred == int(index)] = label
    return out

def sam_area_heuristic(masks, height, width):
    """Original ascending-overwrite behavior after descending area sort."""
    out = np.zeros((height, width), dtype=np.uint8)
    for rec in sorted(masks, key=lambda x: x['area'], reverse=True):
        area = rec['area']
        label = 2 if area > 50000 else 1 if area > 5000 else 3
        out[np.asarray(rec['segmentation'], dtype=bool)] = label
    return out
