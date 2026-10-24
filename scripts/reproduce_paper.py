#!/usr/bin/env python3
"""Compact, reproducible TeSS-UAV runner. No archived results required.

Stages: inference (external model environments), evidence, tables, verify.
The optional historical comparison is a diagnostic, not a reproduction input.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCENES = ('scene0', 'scene1', 'scene3', 'scene5', 'scene8')
SEQUENCES = {'scene0':'seq029','scene1':'seq730','scene3':'seq122',
             'scene5':'seq14','scene8':'seq142'}
MODELS = {'sam_heuristic':'generate_sam_heuristic.py',
          'segformer':'generate_segformer.py','segearth':'generate_segearth.py'}
REFERENCE_KEYS = ('num_voxels','used_frames','total_sampled_world_points',
                  'total_frame_voxel_observations','mean_op','mean_vss',
                  'mean_sbd','mean_entropy_norm')


def load_config(path):
    # Accept the small documented YAML subset without requiring an extra parser
    # for the evaluation environment; reject unknown syntax rather than guessing.
    text = path.read_text()
    result = {}
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if ':' not in line:
            raise ValueError(f'Invalid config line: {raw}')
        key, value = line.split(':', 1)
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            parsed = [x.strip() for x in value[1:-1].split(',') if x.strip()]
        else:
            try:
                parsed = float(value) if '.' in value else int(value)
            except ValueError:
                parsed = value.strip('\'"')
        result[key.strip()] = parsed
    return result


def choose_frames(dataset, scene, count):
    root = dataset / 'undistorted_rgb' / scene
    prefix = SEQUENCES[scene] + '_'
    indices = sorted(int(p.stem[len(prefix):]) for p in root.glob(prefix + '*.png')
                     if p.stem[len(prefix):].isdigit())
    if len(indices) < count:
        raise ValueError(f'{scene}: only {len(indices)} RGB frames; need {count}: {root}')
    # Existing scenes use different exported frame ID intervals (e.g. scene3=10).
    # Select the first N stored images, and require equal spacing since the
    # underlying inference scripts use --start and --step.
    chosen = indices[:count]
    if count == 1:
        step = 1
    else:
        step = chosen[1] - chosen[0]
        if step < 1 or any(b-a != step for a,b in zip(chosen,chosen[1:])):
            raise ValueError(f'{scene}: nonuniform frame IDs; set up source dataset explicitly')
    return chosen[0], step


def runner(script, *args, env=None, dry_run=False):
    # Scripts are run from repo root with importable tess_uav/.
    command = [sys.executable, str(PROJECT/'scripts'/script), *map(str,args)]
    if env and env != 'current':
        command = ['conda', 'run', '--no-capture-output', '-n', env, 'python',
                   str(PROJECT/'scripts'/script), *map(str,args)]
    print('+', ' '.join(command), flush=True)
    if not dry_run:
        e = os.environ.copy()
        e['PYTHONNOUSERSITE']='1'
        e['PYTHONPATH']=str(PROJECT) + os.pathsep + e.get('PYTHONPATH','')
        subprocess.run(command, cwd=PROJECT, env=e, check=True)


def ensure_file(path, what):
    if not path.is_file():
        raise FileNotFoundError(f'{what} missing: {path}')


def validate_evidence(path, n):
    summary = json.loads(path.read_text())
    if int(summary.get('used_frames', -1)) != n:
        raise RuntimeError(f'{path}: used_frames={summary.get("used_frames")}; expected {n}')
    if int(summary.get('num_voxels',0)) <= 0:
        raise RuntimeError(f'{path}: no voxels')
    return summary


def verify_table1(file):
    rows = list(csv.DictReader(file.open()))
    if len(rows) != 3:
        raise RuntimeError(f'Table 1 expected 3 models, found {len(rows)}')
    failed = []
    report = []
    for row in rows:
        target = float(row['target_paper_percent'])
        value = float(row['mean_flicker_percent'])
        passed = round(value,2) == target
        report.append(dict(model=row['model'],reference_percent=target,
                           regenerated_percent=value,match_to_2dp=passed))
        if not passed:
            failed.append(row['model'])
    return report, failed


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['inference','evidence','tables','verify','all'], default='all')
    p.add_argument('--config', type=Path, default=PROJECT/'configs/paper.yaml')
    p.add_argument('--dataset-root', type=Path, default=PROJECT/'datasets/uavid_3d')
    p.add_argument('--predictions-root', type=Path, default=PROJECT/'predictions')
    p.add_argument('--output-root', type=Path, default=PROJECT/'outputs')
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=None)
    p.add_argument('--skip-inference', action='store_true')
    p.add_argument('--segearth-root', type=Path, default=None)
    p.add_argument('--simfeatup-root', type=Path, default=None)
    p.add_argument('--sam-checkpoint', type=Path, default=PROJECT/'checkpoints/sam_vit_h_4b8939.pth')
    p.add_argument('--tess-env', default='current', help='Conda env for SAM/SegFormer inference; current uses active Python')
    p.add_argument('--segearth-env', default='segearth', help='Conda env for SegEarth inference')
    p.add_argument('--skip-existing', action='store_true', help='Skip completed predictions/evidence, validating counts')
    p.add_argument('--reference-root', type=Path, default=None, help='Optional historical baseline RESULTS directory, diagnostic only')
    p.add_argument('--dry-run', action='store_true', help='Print commands without running or requiring output files')
    p.add_argument('--strict', action='store_true', help='Return failure if the regenerated Table 1 disagrees with published rounding')
    args=p.parse_args(argv)
    cfg=load_config(args.config)
    scenes=args.scenes if args.scenes else cfg['scenes']
    n=int(cfg['num_frames'])
    data=args.dataset_root.resolve(); preds=args.predictions_root.resolve(); out=args.output_root.resolve()
    stages=['inference','evidence','tables','verify'] if args.stage=='all' else [args.stage]
    if args.skip_inference:
        stages=[stage for stage in stages if stage!='inference']
    if 'inference' in stages:
        for scene in scenes:
            start,step=choose_frames(data,scene,n)
            names=['segearth']+(['segformer','sam_heuristic'] if scene=='scene0' else [])
            for name in names:
                folder=preds/scene/name
                if args.skip_existing and len(list(folder.glob('*/'+ {'segearth':'segearth_mask.png','segformer':'baseline_mask.png','sam_heuristic':'mask.png'}[name])))==n:
                    print(f'SKIP existing {scene}/{name}: {n} masks')
                    continue
                argv=['--dataset-root',data,'--scene',scene,'--output-root',preds,'--num-frames',n,'--start',start,'--step',step]
                if name=='segearth':
                    if args.segearth_root is None or args.simfeatup_root is None:
                        raise ValueError('Inference requires --segearth-root and --simfeatup-root')
                    argv += ['--segearth-root',args.segearth_root.resolve(), '--simfeatup-root',args.simfeatup_root.resolve()]
                    env=args.segearth_env
                elif name=='sam_heuristic':
                    argv += ['--checkpoint',args.sam_checkpoint.resolve()]
                    env=args.tess_env
                else:
                    env=args.tess_env
                runner(MODELS[name], *argv, env=env, dry_run=args.dry_run)
    if 'evidence' in stages:
        for scene in scenes:
            path=out/'voxels'/scene/'stability_summary.json'
            if args.skip_existing and path.exists():
                validate_evidence(path,n)
                print(f'SKIP existing evidence: {scene}')
                continue
            masks=preds/scene/'segearth'
            argv=['--mask_dir',masks,'--mask_name','segearth_mask.png',
                  '--backbone_name','segearth_ov','--depth_dir',data/'depth'/scene,
                  '--intrinsics',data/'intrinsics.json','--poses_json',data/'poses.json',
                  '--scene_id',scene,'--output_dir',out/'voxels'/scene,
                  '--num_frames',n,'--sample_stride',cfg['sample_stride'],
                  '--voxel_size',cfg['voxel_size'],'--num_classes',cfg['num_classes'],
                  '--ignore_label',cfg['ignore_label'], '--depth_scale',cfg['depth_scale'],
                  '--max_depth',cfg['max_depth']]
            runner('build_evidence.py', *argv, dry_run=args.dry_run)
            if not args.dry_run:
                validate_evidence(path,n)
    if 'tables' in stages:
        if 'scene0' in scenes:
            runner('validate_predictions.py','--predictions-root',preds/'scene0',
                   '--models','sam_heuristic','segformer','segearth',dry_run=args.dry_run)
            runner('evaluate_table1.py','--predictions-root',preds/'scene0',
                   '--output-dir',out/'table1', '--models','sam_heuristic','segformer','segearth',
                   '--num-frames',n, dry_run=args.dry_run)
        if not args.dry_run:
            for scene in scenes:
                validate_evidence(out/'voxels'/scene/'stability_summary.json',n)
        for script,sub in [('evaluate_persistence.py','persistence'),('evaluate_classwise.py','classwise')]:
            runner(script,'--input-root',out/'voxels','--output-dir',out/sub,'--scenes',*scenes,
                   dry_run=args.dry_run)
    if 'verify' in stages and not args.dry_run:
        report={'protocol':cfg, 'table1':[], 'scene_evidence':{}, 'limitations':[
            'No paper-number comparison for other tables without a verified manuscript-to-table reference manifest.',
            'Voxel/geometry sensitivity and frame-budget archival-only analyses are not part of the default fresh workflow.']}
        failures=[]
        if 'scene0' in scenes:
            file=out/'table1'/'table1.csv'
            ensure_file(file,'Regenerated Table 1')
            report['table1'],failures=verify_table1(file)
        for scene in scenes:
            current=validate_evidence(out/'voxels'/scene/'stability_summary.json',n)
            row={'regenerated':{k:current.get(k) for k in REFERENCE_KEYS}}
            if args.reference_root is not None:
                original=args.reference_root/scene/'stability_summary.json'
                if original.exists():
                    old=json.loads(original.read_text())
                    row['historical_deltas']={k:current[k]-old[k] for k in REFERENCE_KEYS
                                              if isinstance(current.get(k),(int,float)) and isinstance(old.get(k),(int,float))}
                else:
                    row['historical_reference_missing']=str(original)
            report['scene_evidence'][scene]=row
        out.mkdir(parents=True,exist_ok=True)
        dest=out/'reproduction_report.json'
        dest.write_text(json.dumps(report,indent=2)+'\n')
        print('Verification report:',dest)
        for r in report['table1']:
            print(f"Table 1 {r['model']}: {r['regenerated_percent']:.6f}% vs {r['reference_percent']:.2f}% "
                  f"=> {'PASS' if r['match_to_2dp'] else 'DIFFER'}")
        if failures and args.strict:
            raise RuntimeError(f'Table 1 differs from published values: {failures}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
