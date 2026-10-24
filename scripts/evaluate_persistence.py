#!/usr/bin/env python3
"""Generate pooled and per-scene OP-stratified results from TeSS voxel CSVs."""
import argparse
import csv
import json
from pathlib import Path
from tess_uav.analysis.persistence import aggregate, write_table, compare_historical

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', default='results', help='Directory containing scene*/voxel_metrics.csv')
    parser.add_argument('--output-dir', default='results/analysis/persistence')
    parser.add_argument('--scenes', nargs='+', default=['scene0','scene1','scene3','scene5','scene8'])
    parser.add_argument('--model', default='SegEarth-OV')
    parser.add_argument('--historical', default=None, help='Optional historical CSV; comparison is diagnostic only')
    args = parser.parse_args()
    paths = {s: Path(args.input_root)/s/'voxel_metrics.csv' for s in args.scenes}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing: parser.error('Missing input files: '+', '.join(missing))
    by_scene, pooled = aggregate(paths, args.model)
    out=Path(args.output_dir)
    write_table(out/'persistence_by_scene.csv', by_scene)
    write_table(out/'persistence_pooled.csv', pooled)
    (out/'persistence_summary.json').write_text(json.dumps({
        'model':args.model, 'input_root':str(args.input_root), 'scenes':args.scenes,
        'protocol':'Pooled voxel-weighted; SBD excludes undefined entries; no OP=1 SBD zero-imputation',
        'pooled':pooled}, indent=2)+'\n')
    if args.historical:
        rows=compare_historical(pooled,args.historical,args.model)
        if rows:
            with (out/'comparison_with_historical.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            print('Historical comparison is diagnostic: may use a different semantic taxonomy/population.')
    print('Saved to',out)
    for row in pooled:
        print(f"{row['group']:9} voxels={row['voxels']:7} fraction={row['fraction']:.6f} "+
              f"VSS={row['vss']:.6f} SBD={row['sbd'] if row['sbd'] is not None else 'N/A'} "+
              f"SBD_defined={row['sbd_defined']}")
if __name__=='__main__': main()
