#!/usr/bin/env python3
"""Run voxel-size reconstruction and audit archived robustness experiments."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from tess_uav.analysis.sensitivity import SCENES,compare_summaries,read_summary,summarize_metrics,write_csv

RESOLUTIONS={'r01':0.1,'r02':0.2,'r04':0.4}
NOISE_CONDITIONS=('pose_very_mild','pose_mild_refined','depth_very_mild','depth_mild_refined',
                  'pose_mild','pose_moderate','depth_mild','depth_moderate')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('voxel','frame','noise','all'),default='all')
    parser.add_argument('--historical-root',default='result')
    parser.add_argument('--dataset-root',default='datasets/uavid_3d')
    parser.add_argument('--output-root',default='results/sensitivity')
    parser.add_argument('--mask-template',default='result/week1_segearth_baseline_{scene}')
    parser.add_argument('--rebuild-voxel',action='store_true',help='Reconstruct 0.1 and 0.4 m evidence from masks')
    parser.add_argument('--strict',action='store_true',help='Return nonzero on absent reference or numerical mismatch')
    args=parser.parse_args()
    output=Path(args.output_root); output.mkdir(parents=True,exist_ok=True)
    history=Path(args.historical_root)
    report=[]; errors=[]
    if args.mode in ('voxel','all'):
        for name,size in RESOLUTIONS.items():
            for scene in SCENES:
                if name=='r02':
                    candidate=Path('results')/scene/'stability_summary.json'
                    metrics=Path('results')/scene/'voxel_metrics.csv'
                    reference=history/'week9_repro_segearth'/scene/'stability_summary.json'
                else:
                    base=output/'voxel'/name/scene
                    candidate=base/'stability_summary.json'
                    metrics=base/'voxel_metrics.csv'
                    reference=history/'week9_voxel_sensitivity'/name/scene/'stability_summary.json'
                    if args.rebuild_voxel:
                        cmd=[sys.executable,'scripts/build_evidence.py',
                             '--mask_dir',args.mask_template.format(scene=scene),
                             '--mask_name','segearth_mask.png','--backbone_name','segearth_ov',
                             '--depth_dir',str(Path(args.dataset_root)/'depth'/scene),
                             '--intrinsics',str(Path(args.dataset_root)/'intrinsics.json'),
                             '--poses_json',str(Path(args.dataset_root)/'poses.json'),
                             '--scene_id',scene,'--output_dir',str(base),
                             '--num_frames','20','--sample_stride','4','--voxel_size',str(size),
                             '--num_classes','8','--ignore_label','0']
                        subprocess.run(cmd,check=True)
                status='MISSING'; matched=''; n=''
                if candidate.exists() and reference.exists():
                    new=read_summary(candidate); ref=read_summary(reference)
                    checks=compare_summaries(new,ref)
                    matched=f'{sum(checks.values())}/{len(checks)}'
                    status='PASS' if all(checks.values()) else 'DIFFER'
                    n=new.get('num_voxels','')
                    if status!='PASS':errors.append(f'{name}/{scene}: {checks}')
                else:
                    errors.append(f'No candidate/reference: {candidate} / {reference}')
                report.append(dict(experiment='voxel',condition=name,scene=scene,status=status,checks=matched,voxels=n))
        for name in RESOLUTIONS:
            paths=[output/'voxel'/name/scene/'voxel_metrics.csv' for scene in SCENES] if name!='r02' else [Path('results')/scene/'voxel_metrics.csv' for scene in SCENES]
            if all(p.exists() for p in paths):
                pooled=summarize_metrics(paths)
                (output/f'voxel_{name}_pooled.json').write_text(json.dumps(pooled,indent=2))
    if args.mode in ('frame','all'):
        for scene in ('scene0','scene1'):
            for budget in (20,40,80):
                reference=history/'week9_frame_budget_voxels'/scene/f'n{budget}'/'stability_summary.json'
                if reference.exists():
                    ref=read_summary(reference)
                    status='ARCHIVED_ONLY'
                    n=ref.get('num_voxels','')
                else:
                    status='MISSING';n='';errors.append(f'Missing {reference}')
                report.append(dict(experiment='frame',condition=f'n{budget}',scene=scene,status=status,checks='',voxels=n))
    if args.mode in ('noise','all'):
        for cond in NOISE_CONDITIONS:
            for scene in SCENES:
                reference=history/'week9_geometry_noise'/cond/scene/'stability_summary.json'
                if reference.exists():
                    ref=read_summary(reference)
                    status='ARCHIVED_ONLY';n=ref.get('num_voxels','')
                else:
                    status='MISSING';n='';errors.append(f'Missing {reference}')
                report.append(dict(experiment='noise',condition=cond,scene=scene,status=status,checks='',voxels=n))
    write_csv(output/'sensitivity_audit.csv',report)
    counts={}
    for row in report:
        counts[row['status']]=counts.get(row['status'],0)+1
    print('Sensitivity audit:',counts)
    print('Results:',output/'sensitivity_audit.csv')
    print('ARCHIVED_ONLY means historical results were inspected, not regenerated.')
    if errors:
        print('Unresolved checks:')
        for error in errors[:20]:print(' -',error)
    if args.strict and errors:return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
