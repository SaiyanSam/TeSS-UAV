#!/usr/bin/env python3
"""Original SAM ViT-H automatic masks with area-based pseudo-semantic labels."""
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from tess_uav.inference.protocol import rgb_frames
from tess_uav.inference.mappings import sam_area_heuristic

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root',required=True)
    ap.add_argument('--scene',default='scene0',choices=['scene0'])
    ap.add_argument('--checkpoint',required=True,help='Path to original SAM ViT-H weights (sam_vit_h_4b8939.pth)')
    ap.add_argument('--output-root',default='predictions')
    ap.add_argument('--num-frames',type=int,default=20)
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--step',type=int,default=4)
    ap.add_argument('--device',default='auto')
    args=ap.parse_args()
    frames=rgb_frames(args.dataset_root,args.scene,args.num_frames,args.start,args.step)
    if not Path(args.checkpoint).is_file():raise FileNotFoundError(args.checkpoint)
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    sam=sam_model_registry['vit_h'](checkpoint=args.checkpoint).to(device)
    generator=SamAutomaticMaskGenerator(sam)
    for frame,path in tqdm(frames,desc='SAM heuristic scene0'):
        bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if bgr is None:raise RuntimeError(f'Unable to open RGB: {path}')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        masks=generator.generate(rgb)
        mapped=sam_area_heuristic(masks,*rgb.shape[:2])
        dest=Path(args.output_root)/args.scene/'sam_heuristic'/frame
        dest.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(dest/'mask.png'),mapped):raise RuntimeError(f'Cannot write {dest}')
    print(f'Generated {len(frames)} SAM heuristic masks')
if __name__=='__main__':main()
