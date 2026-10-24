#!/usr/bin/env python3
"""Compare regenerated TeSS-UAV CSV/JSON against an explicit manuscript reference.

Reference defaults to the PUBLIC arXiv v1, NOT the final ICVGIP camera-ready.
Use --reference <verified-camera-ready.json> to check the accepted version.
Never edits generated predictions, source metrics or paper references.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rounded_agreement(actual, expected, decimals):
    # Approximate comparison to displayed precision: half a last decimal unit.
    return abs(float(actual) - float(expected)) < 0.5 * 10 ** (-decimals) + 1e-12

def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def verify(output, reference):
    checks = []
    def test(table, key, metric, actual, expected, decimals=3, note=None):
        if expected is None:
            return
        try:
            a = None if actual in (None, '', 'nan', 'None') else float(actual)
        except (TypeError, ValueError):
            a = None
        good = a is not None and rounded_agreement(a, expected, decimals)
        checks.append({'table':table,'row':key,'metric':metric,'expected':expected,
                       'actual':a,'decimals':decimals,'status':'PASS' if good else ('MISSING' if a is None else 'DIFFER'),
                       **({'note':note} if note else {})})
    t1 = output/'table1'/'table1.csv'
    model_alias = {'sam_heuristic':'SAM + heuristic','segformer':'SegFormer','segearth':'SegEarth-OV'}
    seen = {}
    if t1.exists():
        for r in rows(t1):
            seen[model_alias.get(r['model'],r['model'])] = r.get('mean_flicker_percent')
    for model, expected in reference['table1'].items():
        test('Table 1',model,'flicker_percent',seen.get(model),expected,2)

    for scene, ref in reference['table2'].items():
        path=output/'voxels'/scene/'stability_summary.json'
        d=json.loads(path.read_text()) if path.exists() else {}
        for field, pair in {'voxels':('num_voxels',0),'sampled_points':('total_sampled_world_points',0),
                            'op':('mean_op',2),'vss':('mean_vss',3),'sbd':('mean_sbd',4),
                            'entropy':('mean_entropy_norm',4)}.items():
            if field in ref:
                metric, decimal=pair
                test('Table 2',scene,field,d.get(metric),ref[field],decimal)

    p3=output/'persistence'/'persistence_pooled.csv'
    p3rows={r['group']:r for r in rows(p3)} if p3.exists() else {}
    for group, ref in reference['table3'].items():
        r=p3rows.get(group,{})
        for field, (col, dec, factor) in {'fraction_percent':('fraction',2,100),
                                          'vss':('vss',3,1), 'sbd':('sbd',4,1),
                                          'entropy':('entropy',4,1)}.items():
            if field in ref and ref[field] is not None:
                val=r.get(col)
                val=float(val)*factor if val not in (None,'','None','nan') else None
                test('Table 3',group,field,val,ref[field],dec)

    p4=output/'classwise'/'classwise_recurrence_all.csv'
    # The default analyzer emits multiple conditional versions of class metrics;
    # Table 4 is NOT assumed to be an OP>=3-conditioned table.
    classrows={r['class_name']:r for r in rows(p4)} if p4.exists() else {}
    for name,ref in reference['table4'].items():
        row=classrows.get(name,{})
        test('Table 4',name,'voxels',row.get('voxel_count'),ref['voxels'],0)
        # Only compare columns when manuscript measure/conditioning is established.
        # mean_vss_all is unconditioned and matches Table 4's caption.
        test('Table 4',name,'vss',row.get('mean_vss_all'),ref['vss'],3)
        # SBD is naturally defined only for recurrent voxels; the current
        # analyzer stores mean_sbd_op_ge_2. Mark as conditional convention.
        test('Table 4',name,'sbd',row.get('mean_sbd_op_ge_2'),ref['sbd'],4,
             'Uses analyzer OP>=2 SBD; confirm camera-ready aggregation convention')
        # Entropy in the paper is unconditioned; current classwise CSV does not
        # store unconditioned entropy. Explicitly report not-verifiable.
        checks.append({'table':'Table 4','row':name,'metric':'entropy',
                       'expected':ref['entropy'],'actual':None,'status':'UNSUPPORTED',
                       'note':'Current classwise CSV lacks unconditioned mean entropy; do not substitute OP>=2 entropy'})
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    p.add_argument('--reference',type=Path,default=ROOT/'configs'/'arxiv_v1_tables.json')
    p.add_argument('--report',type=Path,default=None)
    p.add_argument('--strict',action='store_true',help='Fail unless every expected check passes')
    a=p.parse_args()
    ref=json.loads(a.reference.read_text())
    checks=verify(a.output_root,ref)
    counts={s:sum(c['status']==s for c in checks) for s in ('PASS','DIFFER','MISSING','UNSUPPORTED')}
    report={'reference_id':ref.get('reference_id'),'reference_warning':ref.get('warning'),
            'counts':counts,'checks':checks}
    dest=a.report or (a.output_root/'manuscript_verification.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print('Manuscript reference:',ref.get('reference_id'))
    for table in ('Table 1','Table 2','Table 3','Table 4'):
        subset=[x for x in checks if x['table']==table]
        summary={s:sum(x['status']==s for x in subset) for s in counts}
        print(table,summary)
    print('Report:',dest)
    return 1 if a.strict and any(c['status']!='PASS' for c in checks) else 0
if __name__=='__main__':
    raise SystemExit(main())
