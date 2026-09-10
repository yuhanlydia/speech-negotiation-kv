#!/usr/bin/env python
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from speech_negotiation_kv.parageo_eval import pilot_decision


def read_score(path):
    p=Path(path)
    if p.suffix=='.json':
        data=json.loads(p.read_text())
        for key in ('score','overall','final_score','normalized_score'):
            if key in data: return float(data[key])
    rows=list(csv.DictReader(p.open())) if p.suffix!='.json' else []
    for key in ('score','overall','final_score','normalized_score'):
        if rows and key in rows[0]: return sum(float(r[key]) for r in rows)/len(rows)
    raise ValueError(f'no recognized score in {p}')


def read_wer(path):
    data=json.loads(Path(path).read_text())
    return float(data['mean_text_channel_wer'])


def main():
    ap=argparse.ArgumentParser(description='Apply the frozen ParaGeo pilot stop gate')
    for task in ('static','composed','dynamic'):
        ap.add_argument(f'--baseline-{task}')
        ap.add_argument(f'--ours-{task}')
        ap.add_argument(f'--random-{task}')
        ap.add_argument(f'--baseline-{task}-fidelity')
        ap.add_argument(f'--ours-{task}-fidelity')
    ap.add_argument('--static-min-gain',type=float,default=5.0)
    ap.add_argument('--dynamic-min-gain',type=float,default=8.0)
    ap.add_argument('--max-wer-degradation',type=float,default=0.02)
    ap.add_argument('--output',default='results/parageo_pilot_decision.json')
    args=ap.parse_args()
    baseline={}; ours={}; random={}; degradation={}
    for task in ('static','composed','dynamic'):
        b=getattr(args,f'baseline_{task}'); o=getattr(args,f'ours_{task}'); r=getattr(args,f'random_{task}')
        bf=getattr(args,f'baseline_{task}_fidelity'); of=getattr(args,f'ours_{task}_fidelity')
        supplied=[b,o,r,bf,of]
        if not any(supplied): continue
        if not all(supplied): raise ValueError(f'{task}: baseline/ours/random scores and both fidelity files are required together')
        baseline[task]=read_score(b); ours[task]=read_score(o); random[task]=read_score(r)
        degradation[task]=read_wer(of)-read_wer(bf)
    result=pilot_decision(baseline=baseline,ours=ours,random=random,wer_degradation=degradation,static_min_gain=args.static_min_gain,dynamic_min_gain=args.dynamic_min_gain,max_wer_degradation=args.max_wer_degradation)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
