#!/usr/bin/env python3
"""Fail early when a newly generated prediction set cannot follow the paper protocol."""
import argparse
from pathlib import Path
import numpy as np
from tess_uav.data.uavid3d import load_mask

FILES = {'sam_heuristic':'mask.png','segformer':'baseline_mask.png','segearth':'segearth_mask.png'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions-root',required=True)
    p.add_argument('--models',nargs='+',choices=list(FILES),default=list(FILES))
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--step',type=int,default=4)
    p.add_argument('--num-frames',type=int,default=20)
    p.add_argument('--shape',nargs=2,type=int,default=[512,1024],metavar=('HEIGHT','WIDTH'))
    args=p.parse_args()
    root=Path(args.predictions_root)
    expected=[f'{i:05d}' for i in range(args.start,args.start+args.step*args.num_frames,args.step)]
    for model in args.models:
        model_dir=root/model
        dirs={d.name for d in model_dir.iterdir() if d.is_dir()} if model_dir.is_dir() else set()
        absent=sorted(set(expected)-dirs)
        extra=sorted(dirs-set(expected))
        if absent or extra:
            raise SystemExit(f'{model}: frame IDs mismatch; missing={absent}; extra={extra}')
        for frame in expected:
            f=model_dir/frame/FILES[model]
            if not f.is_file():
                raise SystemExit(f'{model}: missing {f}')
            a=load_mask(f)
            if a.shape != tuple(args.shape):
                raise SystemExit(f'{model}: unexpected shape {a.shape} at {f}')
            if np.any(a<0) or np.any(a>255):
                raise SystemExit(f'{model}: invalid class IDs at {f}')
        print(f'{model}: PASS ({len(expected)} frames, {args.shape[0]}x{args.shape[1]})')

if __name__=='__main__':main()
