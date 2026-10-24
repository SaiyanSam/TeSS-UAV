#!/usr/bin/env python3
"""Generate paper-style SegFormer label masks from fresh UAVid-3D-Scenes RGB."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from tess_uav.inference.protocol import SCENE_TO_SEQUENCE, rgb_frames
from tess_uav.inference.mappings import cityscapes_to_three

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root',required=True)
    ap.add_argument('--scene', choices=SCENE_TO_SEQUENCE, default='scene0')
    ap.add_argument('--output-root',default='predictions')
    ap.add_argument('--model',default='nvidia/segformer-b0-finetuned-cityscapes-1024-1024')
    ap.add_argument('--num-frames',type=int,default=20)
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--step',type=int,default=4)
    ap.add_argument('--device',default='auto')
    args=ap.parse_args()
    frames=rgb_frames(args.dataset_root,args.scene,args.num_frames,args.start,args.step)
    import torch
    import torch.nn.functional as F
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    processor=SegformerImageProcessor.from_pretrained(args.model)
    model=SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    for frame,path in tqdm(frames,desc=f'SegFormer {args.scene}'):
        bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if bgr is None:raise RuntimeError(f'Unable to open RGB: {path}')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        inputs=processor(images=Image.fromarray(rgb),return_tensors='pt').to(device)
        with torch.no_grad(): logits=model(**inputs).logits
        logits=F.interpolate(logits,size=rgb.shape[:2],mode='bilinear',align_corners=False)
        raw=logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        mapped=cityscapes_to_three(raw,model.config.id2label)
        dest=Path(args.output_root)/args.scene/'segformer'/frame
        dest.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(dest/'baseline_mask.png'),mapped):raise RuntimeError(f'Cannot write {dest}')
    print(f'Generated {len(frames)} SegFormer masks for {args.scene}')
if __name__=='__main__':main()
