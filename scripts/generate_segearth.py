#!/usr/bin/env python3
"""Paper-configured SegEarth-OV inference using the ORIGINAL upstream implementation.

Requires the upstream SegEarth-OV and SimFeatUp sources and their weights.
Run in the environment required by those sources; no substitute backbone is used.
"""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from tess_uav.inference.protocol import SEGEARTH_CLASSES,SCENE_TO_SEQUENCE,rgb_frames

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root',required=True)
    ap.add_argument('--scene',default='scene0',choices=SCENE_TO_SEQUENCE)
    ap.add_argument('--segearth-root',required=True,help='Original SegEarth-OV checkout including model dependencies and weights')
    ap.add_argument('--simfeatup-root',required=True,help='Original SimFeatUp checkout')
    ap.add_argument('--output-root',default='predictions')
    ap.add_argument('--num-frames',type=int,default=20)
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--step',type=int,default=4)
    args=ap.parse_args()
    frames=rgb_frames(args.dataset_root,args.scene,args.num_frames,args.start,args.step)
    root=Path(args.segearth_root).resolve()
    if not (root/'segearth_segmentor.py').is_file():raise FileNotFoundError(root/'segearth_segmentor.py')
    import torch
    from torchvision import transforms
    # Compatibility workaround copied from original Week5 implementation.
    from unittest.mock import MagicMock
    sys.modules['mmcv._ext']=MagicMock()
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(Path(args.simfeatup_root).resolve()))
    
    # Compatibility workaround from the original ICVGIP SegEarth pipeline
    # Bypasses imports of unused MMCV compiled operators
    from unittest.mock import MagicMock
    sys.modules["mmcv._ext"] = MagicMock()

    from segearth_segmentor import SegEarthSegmentation
    names=root/'configs'/'uavid_names.txt'
    names.parent.mkdir(parents=True,exist_ok=True)
    names.write_text('\n'.join(SEGEARTH_CLASSES))
    model=SegEarthSegmentation(clip_type='CLIP',vit_type='ViT-B/16',model_type='SegEarth',ignore_residual=True,feature_up=False,name_path=str(names),prob_thd=0.1)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711]),transforms.Resize((448,448))])
    if not torch.cuda.is_available():raise RuntimeError('Original SegEarth Week5 code expects CUDA; CUDA not detected')
    for frame,path in tqdm(frames,desc=f'SegEarth {args.scene}'):
        bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if bgr is None:raise RuntimeError(f'Unable to open RGB: {path}')
        h,w=bgr.shape[:2]
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        tensor=transform(Image.fromarray(rgb)).unsqueeze(0).to('cuda')
        with torch.no_grad():result=model.predict(tensor,data_samples=None)
        pred=result.data.cpu().numpy().squeeze(0).astype(np.uint8)
        pred=cv2.resize(pred,(w,h),interpolation=cv2.INTER_NEAREST)
        dest=Path(args.output_root)/args.scene/'segearth'/frame
        dest.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(dest/'segearth_mask.png'),pred):raise RuntimeError(f'Cannot write {dest}')
    print(f'Generated {len(frames)} SegEarth masks for {args.scene}')
if __name__=='__main__':main()
