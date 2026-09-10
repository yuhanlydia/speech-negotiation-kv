#!/usr/bin/env python
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def read_score(path):
    p=Path(path)
    if p.suffix=='.json':
        data=json.loads(p.read_text());
        for key in ('score','overall','final_score','normalized_score'):
            if key in data: return float(data[key])
        raise ValueError(f'no recognized score key in {p}')
    rows=list(csv.DictReader(p.open()))
    if not rows: raise ValueError(f'empty score file {p}')
    for key in ('score','overall','final_score','normalized_score'):
        if key in rows[0]: return sum(float(r[key]) for r in rows)/len(rows)
    raise ValueError(f'no recognized score column in {p}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline-static',required=True); ap.add_argument('--ours-static',required=True); ap.add_argument('--baseline-dynamic',required=True); ap.add_argument('--ours-dynamic',required=True); ap.add_argument('--static-min-gain',type=float,default=5.0); ap.add_argument('--dynamic-min-gain',type=float,default=8.0); ap.add_argument('--output',default='results/parageo_pilot_decision.json'); args=ap.parse_args()
    bs,os=read_score(args.baseline_static),read_score(args.ours_static); bd,od=read_score(args.baseline_dynamic),read_score(args.ours_dynamic); gs,gd=os-bs,od-bd; passed=bool(gs>=args.static_min_gain or gd>=args.dynamic_min_gain)
    result={'baseline_static':bs,'ours_static':os,'static_gain':gs,'baseline_dynamic':bd,'ours_dynamic':od,'dynamic_gain':gd,'static_min_gain':args.static_min_gain,'dynamic_min_gain':args.dynamic_min_gain,'parageo_pilot_passes':passed,'decision':'continue_to_full_benchmark' if passed else 'stop_parageo'}; Path(args.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
