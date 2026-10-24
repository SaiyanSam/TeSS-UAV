#!/usr/bin/env python3
"""Compare scene-level evaluation outputs against a reference summary."""
import argparse,json,math
from pathlib import Path
KEYS=['num_voxels','mean_op','mean_vss','mean_sbd','mean_entropy_norm','total_sampled_world_points','total_frame_voxel_observations','used_frames']
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',default='result/week9_repro_segearth/scene0/stability_summary.json');p.add_argument('--candidate',default='results/scene0/stability_summary.json');p.add_argument('--atol',type=float,default=1e-9);a=p.parse_args()
 old=json.loads(Path(a.reference).read_text());new=json.loads(Path(a.candidate).read_text());bad=[]
 for k in KEYS:
  x,y=old.get(k),new.get(k)
  match=(math.isclose(x,y,abs_tol=a.atol,rel_tol=0) if isinstance(x,(int,float)) and isinstance(y,(int,float)) else x==y)
  print(f'{k:40} reference={str(x):20} new={str(y):20} {"PASS" if match else "FAIL"}')
  if not match:bad.append(k)
 raise SystemExit(1 if bad else 0)
if __name__=='__main__':main()
