#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
from speech_negotiation_kv.speechparaling import load_prompt_jsonl, parse_static_control, attribute_key

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark-root', required=True)
    ap.add_argument('--language', default='en', choices=['en','ch'])
    ap.add_argument('--dimensions', nargs='+', default=['Pitch','Timbre','Pace','Volume','Rhythm','Age'])
    ap.add_argument('--output', default='results/parageo_attribute_catalog.json')
    args=ap.parse_args()
    root=Path(args.benchmark_root)
    prompt=root/f'jsonl_prompt_{args.language}'/'para_con'/'short_sin.jsonl'
    items=load_prompt_jsonl(prompt, task='para_con_short_sin')
    allowed=set(args.dimensions); catalog={}
    for item in items:
        if len(item.dimensions)!=1 or item.dimensions[0] not in allowed:
            continue
        try: control=parse_static_control(item.prompt)
        except ValueError: continue
        key=attribute_key(item.dimensions[0], control)
        catalog.setdefault(key, {'dimension':item.dimensions[0], 'control':control, 'example_prompt':item.prompt})
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(out),'attributes':len(catalog),'dimensions':sorted({v['dimension'] for v in catalog.values()})},indent=2))
if __name__=='__main__': main()
