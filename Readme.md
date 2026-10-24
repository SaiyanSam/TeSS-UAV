# TeSS-UAV: Temporal Semantic Stability in UAV Perception

**Reproduction code for:** *Understanding Temporal Semantic Stability in Open-Vocabulary UAV Perception through Metric 3D Fusion* (ICVGIP 2026).

TeSS-UAV evaluates how semantic labels evolve when repeated UAV observations are associated with the same metric 3D voxels. It reports **Voxel Semantic Stability (VSS)**, **Semantic Belief Drift (SBD)**, **Observation Persistence (OP)**, entropy, and raw frame-wise semantic flicker. These quantities diagnose temporal consistency and **do not measure semantic correctness**.

This repository is designed to reconstruct results from **source RGB/depth/poses and externally obtained model weights**. No archived rebuttal outputs, historical segmentation masks, datasets, or model checkpoints are required as *inputs* to the main workflow. Exact numerical agreement with historical inference is not guaranteed; see [Reproduction status](#reproduction-status-and-known-deviations).

## 1. What is reproduced?

| Paper result | Input/protocol | Execution | Status |
|---|---|---|---|
| Table 1: scene0 raw pixel flicker | SAM + heuristic, SegFormer B0, SegEarth-OV | `scripts/reproduce_paper.py --stage all` | Three published entries reproduced to reported precision; raw SAM proposals require a separate validated run |
| Table 2: five-scene voxel stability | Eight-label SegEarth masks, 0.2 m voxels | Main runner; `outputs/voxels/scene*/stability_summary.json` | Evaluation implemented; fresh `scene1` masks differ from historical masks |
| Table 3: persistence-stratified stability | Shared building/road/vegetation taxonomy | Explicit **Table 3 commands below** | Implemented and tested independently; not yet wired into main runner |
| Table 4: class-wise stability | Eight-label evidence | Main runner; `outputs/classwise/` | Evaluation implemented; fresh `scene1` masks differ from historical masks |
| Cross-backbone, warped flicker, geometric/noise/frame-density controls | Separate experimental protocols | Dedicated scripts where available | **Not all fully integrated or validated from scratch**; do not claim the full manuscript is covered by `--stage all` |

## 2. Obtain the dataset and model dependencies

Download the original **UAVid-3D-Scenes** dataset from [the dataset source](https://huggingface.co/datasets/hrflr/uavid-3d-scenes), subject to its license and citation requirements. Arrange the required undistorted RGB images, depth maps, camera intrinsics and poses in a common root:

```text
/path/to/uavid_3d/
├── rgb/                 # Dataset imagery; see scripts' loader requirements
├── depth/
│   ├── scene0/
│   ├── scene1/
│   ├── scene3/
│   ├── scene5/
│   └── scene8/
├── intrinsics.json
└── poses.json
```

**Before inference, confirm the actual RGB directory names** against `scripts/generate_segearth.py --help` and `tess_uav/data/uavid3d.py` for the version checked out. The structure above illustrates required data types rather than promising that every dataset distribution uses the same RGB folder name.

Obtain model weights and inference dependencies from their respective upstream projects:

- **Segment Anything (SAM), ViT-H**, plus its official checkpoint.
- **SegFormer B0**, with the model/checkpoint configuration expected by `scripts/generate_segformer.py`.
- **SegEarth-OV** and its external **SimFeatUp** checkout, in a compatible separate Python environment.

No checkpoint, third-party model source, original dataset, or old prediction mask is redistributed here. See `docs/inference.md` and the checked-in environment requirements for model-specific installation steps. The authors used environments called `tess` and `segearth`; environment names can be changed in the runner flags.

## 3. Install and validate

Run all commands below from the **TeSS-UAV repository root**.

```bash
conda env create -f environment.yml
conda activate tess
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/reproduce_paper.py --help
```

Prepare the separate SegEarth environment according to `docs/inference.md` and upstream installation instructions. Example local path assignments (edit all of them to match your installation):

```bash
export DATA_ROOT=/path/to/uavid_3d
export SEGEARTH_ROOT=/path/to/SegEarth-OV
export SIMFEATUP_ROOT=/path/to/SimFeatUp
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
```

## 4. Generate semantic masks **from scratch**

The compact runner generates SegEarth-OV masks for all five scenes, plus SAM-heuristic and SegFormer masks for scene0. Do **not** pass `--skip-inference` on the initial run.

```bash
python scripts/reproduce_paper.py \
  --stage inference \
  --dataset-root "$DATA_ROOT" \
  --segearth-root "$SEGEARTH_ROOT" \
  --simfeatup-root "$SIMFEATUP_ROOT" \
  --sam-checkpoint "$SAM_CHECKPOINT" \
  --tess-env tess --segearth-env segearth
```

Prediction directories should contain one mask per selected frame, for example:

```text
predictions/
├── scene0/
│   ├── segearth/00000/segearth_mask.png
│   ├── segformer/00000/baseline_mask.png
│   └── sam_heuristic/00000/mask.png
├── scene1/segearth/...
├── scene3/segearth/...
├── scene5/segearth/...
└── scene8/segearth/...
```

All scenes use **20 frames**. The original evaluation selects frame IDs at intervals of **4** for scene0 and scene1 and **20** for scene3, scene5 and scene8. This temporal interval is distinct from the **4-pixel spatial stride** used for voxel fusion. Confirm that the inference runner's automatically selected frame IDs agree with these before continuing. In particular, the expected first and last IDs are `00000`–`00076` (steps of four) and `00000`–`00380` (steps of twenty), respectively.

The model-specific commands and versions can affect the masks. For a truly independent rerun, use a clean `predictions/` directory rather than relying on pre-existing masks or `--skip-existing`.

## 5. Run Tables 1, 2 and 4

With predictions generated, build eight-label semantic evidence and calculate the standard analyses:

```bash
python scripts/reproduce_paper.py \
  --stage all --skip-inference \
  --dataset-root "$DATA_ROOT"
```

Alternatively, execute individual stages:

```bash
python scripts/reproduce_paper.py --stage evidence --dataset-root "$DATA_ROOT"
python scripts/reproduce_paper.py --stage tables --dataset-root "$DATA_ROOT"
python scripts/reproduce_paper.py --stage verify --dataset-root "$DATA_ROOT"
```

The runner uses the main configuration in `configs/paper.yaml`: 8 semantic label slots, 20 frames/scene, 0.2 m voxel size, spatial stride 4, ignored label 0, depth scale 1, and maximum depth 120 m. The following output files are produced:

| Directory/file | Contents |
|---|---|
| `outputs/table1/` | Same-pixel flicker for SAM-heuristic, SegFormer, SegEarth-OV; per-pair details |
| `outputs/voxels/<scene>/voxel_metrics.csv` | Per-voxel records for the main eight-label evaluation |
| `outputs/voxels/<scene>/stability_summary.json` | Scene-level counts, OP, VSS, SBD and entropy for Table 2 |
| `outputs/persistence/` | **Eight-label diagnostic only**; **not** the published Table 3 |
| `outputs/classwise/` | Main eight-label class-wise aggregation for Table 4 |
| `outputs/reproduction_report.json` | Verification report and available paper-number checks |

The current `--strict` option checks rounded **Table 1** entries, not all tables. A passing strict check is **not** proof of exact end-to-end reproduction of every manuscript number.

## 6. Run Table 3: shared-three-class persistence

**Important:** The published Table 3 does **not** use `outputs/persistence/`. It uses a separate shared taxonomy: `0=ignore`, `1=building`, `2=road`, `3=vegetation` (tree and low vegetation merged). The original mapping is already implemented in `tess_uav/backbones/shared_taxonomy.py`.

### 6.1 Convert freshly generated SegEarth masks

```bash
python - <<'PY'
from pathlib import Path
import cv2
from tess_uav.backbones.shared_taxonomy import remap_segearth

scenes = ('scene0', 'scene1', 'scene3', 'scene5', 'scene8')
for scene in scenes:
    src = Path('predictions') / scene / 'segearth'
    masks = sorted(src.glob('*/segearth_mask.png'))
    assert len(masks) == 20, f'{scene}: expected 20 masks, found {len(masks)}'
    for path in masks:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2:
            raise ValueError(f'Invalid semantic mask: {path}')
        dst = Path('outputs/table3/masks') / scene / path.parent.name / 'shared3_mask.png'
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), remap_segearth(image)):
            raise IOError(dst)
    print(f'{scene}: {len(masks)} shared masks')
PY
```

### 6.2 Reconstruct shared-taxonomy voxels

Unlike the main experiment, this uses `--num_classes 4`, `--entropy_num_classes 3`, and `--ignore_label 0`.

```bash
set -e
for scene in scene0 scene1 scene3 scene5 scene8; do
  python scripts/build_evidence.py \
    --mask_dir "outputs/table3/masks/$scene" \
    --mask_name shared3_mask.png \
    --backbone_name segearth_shared3 \
    --depth_dir "$DATA_ROOT/depth/$scene" \
    --intrinsics "$DATA_ROOT/intrinsics.json" \
    --poses_json "$DATA_ROOT/poses.json" \
    --scene_id "$scene" \
    --output_dir "outputs/table3/voxels/$scene" \
    --num_frames 20 \
    --sample_stride 4 \
    --voxel_size 0.2 \
    --num_classes 4 \
    --entropy_num_classes 3 \
    --ignore_label 0 \
    --depth_scale 1 \
    --max_depth 120
done
```

### 6.3 Aggregate published persistence groups

```bash
python scripts/evaluate_persistence.py \
  --input-root outputs/table3/voxels \
  --output-dir outputs/table3/persistence \
  --scenes scene0 scene1 scene3 scene5 scene8
```

Read `outputs/table3/persistence/persistence_pooled.csv`. The published groups are OP=1, OP=2, 3≤OP≤5 and OP>5. **Only use this shared-taxonomy output for Table 3.**

## 7. Published values and scope of comparison

These are **reference numbers printed in the paper**, not bundled result files used by the pipeline.

### Table 1 — raw same-pixel scene0 flicker (%)

| SAM + heuristic | Raw SAM proposals | SegFormer | SegEarth-OV |
|---:|---:|---:|---:|
| 29.47 | 21.64 | 25.39 | 15.73 |

The compact runner presently verifies the first, third and fourth values; the raw SAM proposal result requires a separately verified calculation.

### Table 2 — main eight-label voxel evidence

| Scene | Voxels | Sampled points | Mean OP | VSS | SBD | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| scene0 | 306,957 | 637,068 | 2.08 | 0.951 | 0.0466 | 0.0395 |
| scene1 | 320,725 | 624,234 | 1.95 | 0.908 | 0.0949 | 0.0758 |
| scene3 | 27,904 | 32,999 | 1.18 | 0.981 | 0.0714 | 0.0133 |
| scene5 | 23,862 | 27,433 | 1.15 | 0.978 | 0.1007 | 0.0155 |
| scene8 | 26,454 | 32,169 | 1.22 | 0.966 | 0.1091 | 0.0239 |
| **Total / macro mean** | **705,902** | **1,353,903** | **1.51** | **0.957** | **0.0845** | **0.0336** |

The final row sums counts and takes the **unweighted arithmetic average of scene-level means** for OP, VSS, SBD and entropy; it is not a pooled-voxel mean.

### Table 3 — shared-three-class persistence

| Persistence | Voxels (%) | VSS | SBD | Entropy |
|---|---:|---:|---:|---:|
| OP=1 | 57.89 | 1.000 | — | 0.0000 |
| OP=2 | 20.32 | 0.873 | 0.0790 | 0.1602 |
| 3≤OP≤5 | 18.02 | 0.869 | 0.0454 | 0.2285 |
| OP>5 | 3.77 | 0.869* | 0.0216 | 0.2530 |

*The original archived unrounded VSS in the last row was 0.86948184; conventional three-decimal rounding gives 0.869. Some manuscript versions print 0.870.* The historical three-class voxel population was **696,498**, not 705,902, because taxonomy filtering changes which observations enter fusion.

### Table 4 — main eight-label class-wise statistics

| Class | Voxels | VSS | SBD | Entropy |
|---|---:|---:|---:|---:|
| building | 286,565 | 0.939 | 0.0626 | 0.0464 |
| road | 207,895 | 0.915 | 0.0913 | 0.0685 |
| tree | 97,988 | 0.929 | 0.0851 | 0.0612 |
| low vegetation | 102,703 | 0.961 | 0.0393 | 0.0375 |
| moving car | 10,564 | 0.960 | 0.0498 | 0.0403 |

For Table 4, VSS and entropy are aggregated across all voxels of each class; SBD is averaged over the subset for which SBD is defined (OP≥2). Rare static-car and human classes are excluded from the displayed rows.

## 8. Reproduction status and known deviations

Reconstructing the paper's *method* is distinct from regenerating historically identical *model predictions*. The pipeline has been exercised on fresh predictions, and the following checks have been established:

- **Table 1:** freshly regenerated SAM-heuristic, SegFormer and SegEarth-OV flicker are **29.466870%**, **25.388341%** and **15.734683%**, respectively; these agree with published two-decimal values.
- **Table 2:** the original archived eight-label voxel aggregates reproduce the paper's printed numbers. Fresh evidence reproduces four scenes closely; `scene1` differs in semantic predictions and stability metrics.
- **Table 3:** the shared-taxonomy mapping and aggregation were validated against the original protocol. The historical five-scene evidence contained **696,498 voxels**; the fresh run produced **697,189** (difference 691, ~0.10%). Fresh recurrent-voxel metrics differ because of `scene1`.
- **Table 3, excluding scene1:** across the other four scenes, total voxel counts were **382,104 historical** versus **382,109 fresh**. For each recurrent group, absolute VSS, SBD and entropy differences were at most approximately **0.00022**. This strongly localizes the five-scene discrepancy to scene1, without establishing why its original masks differed.
- **Table 4:** class-wise values reconstruct from the historical eight-label evidence; fresh class-wise values inherit the `scene1` difference.

The source of the `scene1` model-output discrepancy has **not** been conclusively established. Do not silently replace historical paper values with fresh numbers, and do not describe archived-output reconstruction as independent inference reproduction. Equally, do not claim the compact runner reproduces all supplementary analyses when their independent input generation has not been verified.

## 9. Output hygiene and repository cleanup

Only source code, documentation, configuration files, small fixtures, environment specifications and tests should be tracked. Generated results must be regenerable and must not be required at checkout.

Suggested `.gitignore` additions (merge with any existing file):

```gitignore
# Datasets / predictions / generated evidence
/data/
/datasets/
/predictions/
/outputs/
/results/
/result/
/result(ICVGIP Rebuttal)/
/checkpoints/
*.pth
*.pt
*.ckpt
*.safetensors

# Python and local environments
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
.env
```

Before committing, **inspect** tracked large or generated files; ignoring files alone does not stop Git from tracking files already committed.

```bash
python -m pytest -q
python scripts/reproduce_paper.py --stage evidence --dry-run --dataset-root "$DATA_ROOT"
git status --short
git ls-files | grep -E '(^data/|^datasets/|^predictions/|^outputs/|^results/|^result/|^checkpoints/|\.(pth|pt|ckpt|safetensors)$)' || true
```

If those paths are already tracked, review them individually before removing them from Git's index with `git rm --cached`; **do not delete local raw datasets or archived results merely to clean the repository**.

### Release checklist

- [ ] New checkout installs successfully and tests pass.
- [ ] Source dataset and checkpoint download references are documented, including licenses.
- [ ] Running inference from an empty `predictions/` directory generates expected frame IDs and masks.
- [ ] Main runner reconstructs eight-label evidence without accessing any historical folder.
- [ ] Table 3 commands reconstruct shared-taxonomy evidence without accessing any historical folder.
- [ ] Table 1's raw SAM proposals are either independently reproduced or explicitly marked outstanding.
- [ ] Independent commands are added and validated for any additional paper results claimed as reproducible.
- [ ] README limitations accurately distinguish regenerated inference from reference-number agreement.
- [ ] No model weights, source dataset, archival rebuttal results or generated outputs are tracked.

## Citation

If you use the code or metrics, cite the ICVGIP 2026 paper. Add the final published BibTeX entry and DOI once official metadata is available.
