#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from speech_negotiation_kv.parageo_eval import select_scale


def read_score(path: str) -> float:
    p=Path(path)
    if p.suffix=='.json':
        data=json.loads(p.read_text())
        for key in ('score','overall','final_score','normalized_score'):
            if key in data: return float(data[key])
    else:
        rows=list(csv.DictReader(p.open()))
        for key in ('score','overall','final_score','normalized_score'):
            if rows and key in rows[0]: return sum(float(row[key]) for row in rows)/len(rows)
    raise ValueError(f'no recognized score in {p}')


def main():
    ap=argparse.ArgumentParser(description='Freeze ParaGeo steering scale from dev-only judge scores')
    ap.add_argument('--score',action='append',required=True,help='ALPHA=PATH')
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    scores={}
    for item in args.score:
        alpha,path=item.split('=',1); scores[float(alpha)]=read_score(path)
    chosen=select_scale(scores)
    result={'dev_scores':{str(k):v for k,v in sorted(scores.items())},'selected_scale':chosen,'selection_rule':'highest dev score; smallest alpha breaks exact ties'}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
