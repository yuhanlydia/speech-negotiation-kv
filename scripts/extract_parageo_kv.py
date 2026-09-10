#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, yaml
from speech_negotiation_kv.glm_voice import GLMVoiceBackend, audio_ids_to_prompt
from speech_negotiation_kv.kv_hooks import FusedQKVRecorder
from speech_negotiation_kv.records import read_jsonl

def representation_prompt(audio_ids):
    return '<|system|>\nListen to the speech and represent how it is spoken. Focus on paralinguistic delivery and ignore lexical content.<|user|>\n'+audio_ids_to_prompt(audio_ids)+'<|assistant|>streaming_transcription\n'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/parageo_speechparaling_pilot.yaml'); ap.add_argument('--records',default='results/parageo_calibration.jsonl'); ap.add_argument('--output',default='results/parageo_calibration_kv.npz'); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); m=cfg['model']; layers=list(cfg['parageo']['kv_layers'])
    backend=GLMVoiceBackend(model_name=m['name'],quantization=m.get('quantization','int4'),device=m.get('device','cuda:0'),max_new_tokens=m.get('max_new_tokens',256),temperature=m.get('temperature',0.8),top_p=m.get('top_p',0.8),audio_vocab_size=m.get('audio_vocab_size',16384)); backend._ensure_loaded()
    model,tokenizer=backend.model,backend.tokenizer; mc=model.config
    recorder=FusedQKVRecorder(model,layer_indices=layers,num_attention_heads=int(mc.num_attention_heads),kv_channels=int(mc.kv_channels),multi_query_group_num=int(mc.multi_query_group_num),pooling='audio_only')
    import torch
    offset=int(tokenizer.convert_tokens_to_ids('<|audio_0|>')); vocab=int(m.get('audio_vocab_size',16384)); bids=[]; memories=[]
    rows=read_jsonl(args.records)
    for i,row in enumerate(rows):
        if not row.get('matched_semantics',False) or not row.get('audio_token_ids'): continue
        prompt=representation_prompt(row['audio_token_ids']); inputs=tokenizer([prompt],return_tensors='pt'); device=next(model.parameters()).device; inputs={k:v.to(device) for k,v in inputs.items()}; mask=(inputs['input_ids']>=offset)&(inputs['input_ids']<offset+vocab); recorder.set_token_mask(mask); recorder.clear()
        with torch.inference_mode(), recorder: model(**inputs,use_cache=False)
        memories.append(torch.stack([torch.stack([recorder.layer_values[layer][0],recorder.layer_values[layer][1]]) for layer in layers]).numpy()); bids.append(row['branch_id'])
        if (i+1)%50==0: print(f'extracted {i+1}/{len(rows)}')
    if not memories: raise RuntimeError('no matched calibration K/V')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,branch_ids=np.asarray(bids),memory=np.stack(memories).astype(np.float16),layers=np.asarray(layers,dtype=np.int16),pooling=np.asarray('audio_only'))
    print(out)
if __name__=='__main__': main()
