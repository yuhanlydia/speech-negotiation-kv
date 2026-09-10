#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import yaml
from speech_negotiation_kv.glm_voice import GLMVoiceBackend, normalize_transcript

def seed_for(base:int, content_idx:int, attr_idx:int)->int:
    return int((base + 1009*content_idx + 9176*attr_idx) % (2**31-1))

def main():
    ap=argparse.ArgumentParser(description='Collect matched-content ParaGeo calibration speech')
    ap.add_argument('--config',default='configs/parageo_speechparaling_pilot.yaml')
    ap.add_argument('--catalog',default='results/parageo_attribute_catalog.json')
    ap.add_argument('--output',default='results/parageo_calibration.jsonl')
    ap.add_argument('--fresh',action='store_true')
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); m=cfg['model']; exp=cfg['parageo']
    catalog=json.loads(Path(args.catalog).read_text()); attrs=sorted(catalog)
    backend=GLMVoiceBackend(model_name=m['name'],quantization=m.get('quantization','int4'),device=m.get('device','cuda:0'),max_new_tokens=m.get('max_new_tokens',256),temperature=m.get('temperature',0.8),top_p=m.get('top_p',0.8),audio_vocab_size=m.get('audio_vocab_size',16384))
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    if args.fresh and out.exists(): raise FileExistsError(out)
    existing=set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                row=json.loads(line); existing.add((row['content_id'],row['attribute']))
    with out.open('a',encoding='utf-8') as handle:
        for ci,text in enumerate(exp['calibration_sentences']):
            content_id=f'cal:{ci:02d}'
            for ai,key in enumerate(attrs):
                if (content_id,key) in existing: continue
                gen=backend.render_exact(text,catalog[key]['control'],seed=seed_for(int(exp['calibration_seed']),ci,ai))
                matched=normalize_transcript(gen.transcript)==normalize_transcript(text)
                branch_id=hashlib.sha1(f'{content_id}|{key}'.encode()).hexdigest()[:16]
                row={'branch_id':branch_id,'content_id':content_id,'attribute':key,'dimension':catalog[key]['dimension'],'control':catalog[key]['control'],'target_text':text,'transcript':gen.transcript,'audio_token_ids':gen.audio_token_ids,'matched_semantics':matched}
                handle.write(json.dumps(row,ensure_ascii=False)+'\n'); handle.flush(); os.fsync(handle.fileno())
    print(str(out))
if __name__=='__main__': main()
