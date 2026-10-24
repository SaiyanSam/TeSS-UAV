# Compact TeSS-UAV reproduction runner

The repository includes `scripts/reproduce_paper.py`, which orchestrates the **existing** inference, 3D evidence, and analysis scripts. It does not read archived `result/` as a required input. Run from the repository root. Paths below follow the authors' local layout; replace them with your own.

## 1. Complete the one-time setup

- Place UAVid-3D-Scenes RGB, depth, `intrinsics.json`, and `poses.json` under a common dataset root. See [reproduce.md](reproduce.md) for layout.
- Install the lightweight `tess` environment and model dependencies; see [inference.md](inference.md).
- Install SegEarth-OV in its **separate compatible environment**, alongside its upstream checkout and SimFeatUp. The `mmcv._ext` `MagicMock` compatibility shim follows the historical scripts; it is not a substitute for actual compiled operators if those are used.
- Obtain the SAM ViT-H checkpoint. No datasets, weights or model checkouts are distributed with TeSS-UAV.

## 2. Generate predictions

```bash
conda activate tess
python scripts/reproduce_paper.py --stage inference \
  --dataset-root ../PerSem/data/uavid_3d \
  --segearth-root ../PerSem/SegEarth-OV \
  --simfeatup-root ../PerSem/SimFeatUp \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --tess-env tess --segearth-env segearth \
  --skip-existing
```

This runs **SegEarth-OV on five scenes**, and SAM ViT-H + heuristic and SegFormer B0 on `scene0` for Table 1. The runner determines a scene's temporal frame-ID interval from the first 20 image filenames. The voxel sampling stride is a different parameter (`sample_stride: 4` pixels). Review printed commands with `--dry-run` if necessary.

## 3. Generate evidence and tables

```bash
conda activate tess
python scripts/reproduce_paper.py --stage evidence \
  --dataset-root ../PerSem/data/uavid_3d --skip-existing
python scripts/reproduce_paper.py --stage tables
python scripts/reproduce_paper.py --stage verify --strict
```

Or, if you already have predictions:

```bash
python scripts/reproduce_paper.py --stage all --skip-inference \
  --dataset-root ../PerSem/data/uavid_3d --skip-existing --strict
```

Outputs include:

- `outputs/table1/table1.csv`, `table1.json`, pairwise flicker CSVs
- `outputs/voxels/<scene>/stability_summary.json` and `voxel_metrics.csv`
- `outputs/persistence/persistence_pooled.csv` and `persistence_by_scene.csv`
- `outputs/classwise/classwise_recurrence_all.csv` and `classwise_recurrence_by_scene.csv`
- `outputs/reproduction_report.json` with Table 1 rounded-value checks and per-scene evidence summaries

`--skip-existing` skips only completed mask sets by file count and voxel summaries passing basic frame-count checks; it does **not** certify that upstream weights or mask content match the paper. For a true fresh rebuild, omit this option or use a new output directory.

## 4. Verification scope and remaining experiments

Table 1 **fresh RGB inference** has been independently run on the authors' machine with these regenerated means: SAM + heuristic 29.466870% (paper 29.47%), SegFormer 25.388341% (25.39%), SegEarth-OV 15.734683% (15.73%). `--strict` fails if those generated values do not match the published **two-decimal** values. It is not an exact-pixel or exact-six-decimal verification.

One fresh `scene0` 3D rebuild closely matched historical geometry and stability summaries (306959 versus 306957 voxels; VSS 0.950844 versus 0.950894; SBD 0.046703 versus 0.046645). `scene1` displayed larger semantic differences versus archived predictions, so **full five-scene numerical agreement has not yet been established**. The report records fresh summaries and optionally deltas against existing summaries via `--reference-root results`; those historical files are **diagnostics**, not prerequisites or substitutes for fresh results.

The compact default tables stage covers flicker, per-scene voxel summaries, persistence and class-wise analysis. **It does not yet regenerate every manuscript and supplementary table**: backbone comparison, frame-budget, geometry-noise, SBD controls and geometry-warped flicker remain available as separate scripts; some original wrappers currently depend on historical masks or compare only archival results. Do not describe these as end-to-end reproduced until their input paths are modernized and tested. Exact paper table/figure numbers should be mapped to outputs using the accepted manuscript.

## 5. Before public push

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/reproduce_paper.py --stage evidence --dry-run \
  --dataset-root ../PerSem/data/uavid_3d

git status --short
# Make sure checkpoints, generated outputs, RGB/depth data, and archived results are excluded.
git ls-files | grep -E '(^results/|^result/|^predictions/|^outputs/|^checkpoints/|\.pth$|\.ckpt$)' || true
```

Do not publish the author-specific dataset paths as if they were universally available.
