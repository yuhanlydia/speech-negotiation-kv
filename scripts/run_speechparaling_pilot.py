#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, yaml
from speech_negotiation_kv.glm_official_waveform import GLMOfficialWaveformBackend, OfficialVoiceAssets
from speech_negotiation_kv.parageo import compose_coordinates,direction_from_coordinate,dynamic_coordinate_schedule
from speech_negotiation_kv.parageo_steering import ScheduledFusedQKVSteerer
from speech_negotiation_kv.speechparaling import load_prompt_jsonl,pair_with_audio,select_catalog_covered_items,split_pilot_items,match_catalog_controls,parse_dynamic_control,attribute_key,extract_target_text

def load_geometry(path):
    data=np.load(path); names=[str(x) for x in data['attribute_names']]; coords={n:data['coordinates'][i].astype(np.float64) for i,n in enumerate(names)}
    return data['basis'].astype(np.float64),coords,[int(x) for x in data['layers']]

def steerer_for(backend,basis,layers,coordinate_schedule,scale):
    backend._ensure_loaded(); mc=backend.model.config
    schedule=np.asarray(coordinate_schedule,dtype=np.float64)
    if schedule.ndim==1: schedule=schedule[None,:]
    directions=np.stack([direction_from_coordinate(basis,row) for row in schedule])
    return ScheduledFusedQKVSteerer(backend.model,layer_indices=layers,directions=directions,num_attention_heads=int(mc.num_attention_heads),kv_channels=int(mc.kv_channels),multi_query_group_num=int(mc.multi_query_group_num),scale=scale)

def main():
    ap=argparse.ArgumentParser(description='Generate SpeechParaling-Bench pilot outputs with ParaGeo')
    ap.add_argument('--config',default='configs/parageo_speechparaling_pilot.yaml'); ap.add_argument('--split',choices=['all','dev','heldout'],default='all'); ap.add_argument('--prompt-jsonl',required=True); ap.add_argument('--audio-dir',required=True); ap.add_argument('--task',required=True,choices=['static','composed','dynamic']); ap.add_argument('--method',required=True,choices=['prompt_only','parageo_static','parageo_composed','parageo_dynamic','random']); ap.add_argument('--basis',default='results/parageo_basis.npz'); ap.add_argument('--catalog',default='results/parageo_attribute_catalog.json'); ap.add_argument('--output-dir',required=True); ap.add_argument('--limit',type=int,default=None); ap.add_argument('--scale',type=float,default=None); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); m=cfg['model']; official=cfg['official_glm']; pilot=cfg['pilot']; scale=float(args.scale if args.scale is not None else pilot['steering_scale'])
    assets=OfficialVoiceAssets(repo_root=str(Path(__import__('os').path.expandvars(official['repo_root'])).expanduser()),speech_tokenizer_path=__import__('os').path.expandvars(official['speech_tokenizer_path']),decoder_path=str(Path(__import__('os').path.expandvars(official['decoder_path'])).expanduser()))
    backend=GLMOfficialWaveformBackend(assets=assets,model_name=m['name'],quantization=m.get('quantization','int4'),device=m.get('device','cuda:0'),max_new_tokens=m.get('max_new_tokens',768),temperature=m.get('temperature',0.8),top_p=m.get('top_p',0.8),audio_vocab_size=m.get('audio_vocab_size',16384))
    basis,coords,layers=load_geometry(args.basis); catalog=json.loads(Path(args.catalog).read_text())
    all_items=pair_with_audio(load_prompt_jsonl(args.prompt_jsonl,task=args.task),args.audio_dir)
    items=select_catalog_covered_items(all_items,catalog,task=args.task,limit=args.limit)
    if args.limit is not None and len(items) < args.limit:
        raise ValueError(f'only {len(items)} catalog-covered items available, requested {args.limit}')
    items=split_pilot_items(items,split=args.split,modulus=int(pilot['dev_modulus']),remainder=int(pilot['dev_remainder']))
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True); records=[]; rng=np.random.default_rng(int(pilot['random_seed']))
    for index,item in enumerate(items):
        steering=None; selected=[]; schedule_meta=None
        if args.method!='prompt_only':
            if args.task in {'static','composed'}:
                selected=match_catalog_controls(item.prompt,catalog)
                if args.task=='static' and selected: selected=selected[:1]
                if not selected: raise ValueError(f'no catalog control matched {item.item_id}: {item.prompt}')
                coordinate=compose_coordinates(coords,selected,normalize=False)
                if args.method=='random':
                    random=rng.normal(size=basis.shape[0]); random/=max(np.linalg.norm(random),1e-12); target=direction_from_coordinate(basis,coordinate); random*=np.linalg.norm(target); coordinate=basis.T@random
                steering=steerer_for(backend,basis,layers,coordinate,scale)
            elif args.task=='dynamic':
                start,end,mode=parse_dynamic_control(item.prompt); dimension=item.dimensions[0]; start_key=attribute_key(dimension,start); end_key=attribute_key(dimension,end)
                if start_key not in coords or end_key not in coords: raise KeyError(f'missing dynamic endpoint(s): {start_key}, {end_key}')
                schedule=dynamic_coordinate_schedule(coords[start_key],coords[end_key],steps=int(pilot['steering_steps']),mode=mode,transition_at=float(pilot['transition_at']))
                selected=[start_key,end_key]; schedule_meta={'mode':mode,'start':start_key,'end':end_key}
                if args.method=='random':
                    random=rng.normal(size=basis.shape[0]); random/=max(np.linalg.norm(random),1e-12); reference=direction_from_coordinate(basis,schedule.mean(0)); random*=max(np.linalg.norm(reference),1e-12); schedule=np.repeat((basis.T@random)[None,:],len(schedule),axis=0)
                steering=steerer_for(backend,basis,layers,schedule,scale)
        output=outdir/Path(item.audio_path).name
        generation=backend.generate_from_audio_instruction(item.audio_path,output,seed=int(pilot['generation_seed'])+index,steering=steering)
        records.append({'item_id':item.item_id,'task':args.task,'method':args.method,'dimensions':list(item.dimensions),'prompt':item.prompt,'target_text':extract_target_text(item.prompt),'input_audio':item.audio_path,'output_audio':generation.wav_path,'selected_attributes':selected,'dynamic_schedule':schedule_meta,'scale':scale,'transcript':generation.generation.transcript,'audio_token_count':len(generation.generation.audio_token_ids)})
        if (index+1)%10==0: print(f'{index+1}/{len(items)}')
    (outdir/'parageo_manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n')
    print(json.dumps({'method':args.method,'task':args.task,'split':args.split,'samples':len(records),'output_dir':str(outdir)},indent=2))
if __name__=='__main__': main()
