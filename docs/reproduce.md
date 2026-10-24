# Reproduction protocol

## 1. Sources and environment

Install with `conda env create -f environment.yml && conda activate tess`. Get UAVid-3D-Scenes from <https://huggingface.co/datasets/hrflr/uavid-3d-scenes>. Respect its license and attribution requirements (original UAVid and TanDepth). The dataset is **not** included in this repository.

Required local geometry layout for the currently implemented CLIs:

```text
datasets/uavid_3d/
  intrinsics.json
  poses.json
  depth/scene0/*.png
  depth/scene1/*.png
  depth/scene3/*.png
  depth/scene5/*.png
  depth/scene8/*.png
```

The reader uses `intrinsics.json` and `poses.json` keyed by scene, takes the supplied extrinsics as world-to-camera, and decodes the supplied 24-bit metric depth maps. The source download may require conversion/organization to this layout; do not assume its on-disk organization is identical.

## 2. Generate semantic predictions (external prerequisite)

**Status:** three model-inference wrappers are now included, and Table 1 was reproduced from fresh RGB images and checkpoints on the authors’ machine. For the recommended concise commands see [compact reproduction](compact-reproduction.md). Use the original model projects with the exact checkpoint, input resolution, preprocessing, label order, image IDs and postprocessing. Avoid treating substitute weights or label mappings as reproductions of the paper.

Paper backbones: SegEarth-OV with an eight-class UAVid task vocabulary; SegFormer (`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`); SAM ViT-H + area-based pseudo-semantic labeling (road when mask area >50,000 pixels; building when >5,000; vegetation otherwise; masks processed largest first). Note that SAM's labels come from the **heuristic**, not SAM itself. All compared Table 1 label images use 0 for ignore/background.

Produce **fresh prediction files** in:

```text
predictions/scene0/
  sam_heuristic/00000/mask.png
  sam_heuristic/00004/mask.png
  ... (20 frames through 00076)
  segformer/00000/baseline_mask.png
  segformer/00004/baseline_mask.png
  ...
  segearth/00000/segearth_mask.png
  segearth/00004/segearth_mask.png
  ...
```

Each PNG must be a **2D integer class-ID image**, not a color visualization. Generate identical frame IDs across models. Do not pull from the archived `result/` folder for a fresh run. Table 1 is a scene0 test (20 frames `00000,00004,...,00076`), distinct from the five-scene 3D analysis.

## 3. Regenerate Table 1 directly from predictions

```bash
python scripts/validate_predictions.py --predictions-root predictions/scene0
python scripts/evaluate_table1.py \
  --predictions-root predictions/scene0 \
  --output-dir outputs/table1
```

Output: `outputs/table1/table1.csv` and `table1.json` plus 19-row per-model pairwise CSVs. The evaluator independently computes, for each pair,

```text
valid = (mask_t != 0) OR (mask_t+1 != 0)
changed = (mask_t != mask_t+1) AND valid
pair_flicker = 100 * changed.sum() / valid.sum()
Table1_flicker = arithmetic_mean(pair_flicker for all 19 pairs)
```

Reference values: SAM + heuristic **29.466870%** (29.47%), SegFormer **25.387967%** (25.39%), SegEarth-OV **15.733625%** (15.73%). Reproducing these requires identical predicted masks/model settings, not only identical evaluator code. Geometry-warped flicker is a **different** experiment.

## 4. Rebuild 3D evidence / persistence (example)

For each of `scene0`, `scene1`, `scene3`, `scene5`, `scene8`, first generate SegEarth predictions (20 corresponding frames per scene) under `predictions/<scene>/segearth/<frame_id>/segearth_mask.png`. Then:

```bash
for scene in scene0 scene1 scene3 scene5 scene8; do
  python scripts/build_evidence.py \
    --mask_dir "predictions/$scene/segearth" \
    --mask_name segearth_mask.png --backbone_name segearth_ov \
    --depth_dir "datasets/uavid_3d/depth/$scene" \
    --intrinsics datasets/uavid_3d/intrinsics.json \
    --poses_json datasets/uavid_3d/poses.json \
    --scene_id "$scene" --output_dir "outputs/voxels/$scene" \
    --num_frames 20 --sample_stride 4 --voxel_size 0.2 \
    --num_classes 8 --ignore_label 0
 done
python scripts/evaluate_persistence.py \
  --input-root outputs/voxels \
  --output-dir outputs/persistence
```

Check `used_frames == 20` and no skipped frames in each generated stability summary. Scripts currently log and skip invalid frames, so an apparently successful exit is **not** proof that the fixed paper protocol was followed.

## 5. Other experimental CLIs

The repository includes `evaluate_classwise.py`, `run_backbone_comparison.py`, `evaluate_sbd_controls.py`, `evaluate_warped_flicker.py`, and geometry/sensitivity controls. Use their `--help` to configure source paths. Some older scripts still have historical directory defaults or diagnostic reference comparisons: override those paths explicitly and never use archived `result/` as a source for a new scientific evaluation. Some sensitivity modes remain archival-only and cannot be called independent data reproduction.

## 6. Validation and claims

`python -m pytest -q` tests algorithmic logic. Historical regression checks may skip when old references are absent; the public workflow does not require them. Table 1 was independently regenerated from fresh model inference, with results SAM 29.466870%, SegFormer 25.388341%, and SegEarth-OV 15.734683% (each matches the paper to two decimal places). The five-scene 3D and sensitivity results have not all been certified from fresh inputs; see [compact reproduction](compact-reproduction.md).
