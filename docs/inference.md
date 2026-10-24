# Inference reconstruction (from the original ICVGIP week-wise code)

The original source was audited from `scripts(ICVGIP).zip`; these are clean, optional-dependency entry points. They **do not** read any historical `result/` directory. Input is downloaded UAVid-3D-Scenes RGB stored as `<dataset-root>/undistorted_rgb/<scene>/<sequence>_<frame>.png`.

The five scene/sequence pairs are scene0/seq029, scene1/seq730, scene3/seq122, scene5/seq14, and scene8/seq142. For the paper's Table 1, run only scene0, 20 frames, start 0, step 4. All generators fail on missing RGB instead of silently skipping frames.

## Table 1 input generation

The following commands are intended to be run from the TeSS-UAV root with `pip install -e .` performed in the chosen environment. Model dependencies are **additional** to core `environment.yml`, and can require independent environments for SegEarth and SAM.

```bash
python scripts/generate_segformer.py --dataset-root datasets/uavid_3d --scene scene0 --output-root predictions
python scripts/generate_sam_heuristic.py --dataset-root datasets/uavid_3d --scene scene0 --checkpoint checkpoints/sam_vit_h_4b8939.pth --output-root predictions
python scripts/generate_segearth.py --dataset-root datasets/uavid_3d --scene scene0 --segearth-root ../PerSem/SegEarth-OV --simfeatup-root ../PerSem/SimFeatUp --output-root predictions
python scripts/validate_predictions.py --predictions-root predictions/scene0
python scripts/evaluate_table1.py --predictions-root predictions/scene0 --output-dir outputs/table1
```

SegFormer uses `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, native `SegformerImageProcessor`, bilinear resize of logits, and maps Cityscapes building/road/vegetation to 1/2/3. SAM is the original `SamAutomaticMaskGenerator` with the ViT-H checkpoint, with region area thresholds `>50000` road (2), `>5000` building (1), otherwise vegetation (3), processing masks in descending area order (so later, smaller masks overwrite). SegEarth uses the original `segearth_segmentor.py` class with CLIP ViT-B/16, 448×448 normalized RGB, eight task-specific class names, `prob_thd=0.1`, and nearest-neighbor resizing of predicted class IDs. These are original-code settings, not alternative pretrained backbones.

**Important limitation:** Model checkpoints and the full SegEarth dependency/weight environment are not included in this patch. The original script did not specify immutable commit hashes or package versions. The new wrappers are syntactically verified and unit tested, but **fresh end-to-end inference has not been run or shown to reproduce archived masks**. Compare regenerated masks/19 flicker values to references on the author's machine before claiming exact table reproduction. The original SegEarth script had conflicting hardcoded scene assignments; the CLI makes scene selection explicit.

For the five-scene 3D study, run the SegEarth command for each scene and use `predictions/<scene>/segearth` as `--mask_dir` in `scripts/build_evidence.py`. Follow `docs/reproduce.md` for remaining geometry and evidence commands.
