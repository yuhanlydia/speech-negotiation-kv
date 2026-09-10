#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,yaml
from speech_negotiation_kv.parageo import fit_content_invariant_basis,attribute_prototypes,coordinates_from_prototypes,semantic_subspace,orthogonalize_against_semantics,cross_content_centroid_accuracy
from speech_negotiation_kv.records import read_jsonl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/parageo_speechparaling_pilot.yaml'); ap.add_argument('--records',default='results/parageo_calibration.jsonl'); ap.add_argument('--kv',default='results/parageo_calibration_kv.npz'); ap.add_argument('--output',default='results/parageo_basis.npz'); ap.add_argument('--summary',default='results/parageo_basis_summary.json'); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); pcfg=cfg['parageo']; rows={r['branch_id']:r for r in read_jsonl(args.records) if r.get('matched_semantics',False)}; kv=np.load(args.kv); memory=kv['memory'].astype(np.float64); features=memory.reshape(len(memory),-1); aligned=[rows[str(b)] for b in kv['branch_ids'] if str(b) in rows]
    idx=[i for i,b in enumerate(kv['branch_ids']) if str(b) in rows]; features=features[idx]
    content=[r['content_id'] for r in aligned]; attrs=[r['attribute'] for r in aligned]
    fit=fit_content_invariant_basis(features,content,rank=pcfg.get('basis_rank'),max_rank=int(pcfg.get('max_basis_rank',32)),energy=float(pcfg.get('basis_energy',0.95)))
    means=np.stack([features[np.asarray(content)==cid].mean(0) for cid in sorted(set(content))]); S=semantic_subspace(means,rank=int(pcfg.get('semantic_rank',8))); B=orthogonalize_against_semantics(fit.basis,S)
    protos=attribute_prototypes(fit.centered_features,attrs); coords=coordinates_from_prototypes(protos,B); names=sorted(coords); coord_matrix=np.stack([coords[n] for n in names])
    unique=sorted(set(content)); split=set(unique[:max(1,len(unique)//2)]); acc=cross_content_centroid_accuracy(features,content,attrs,split)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,basis=B.astype(np.float32),raw_basis=fit.basis.astype(np.float32),attribute_names=np.asarray(names),coordinates=coord_matrix.astype(np.float32),layers=kv['layers'],singular_values=fit.singular_values.astype(np.float32),semantic_basis=S.astype(np.float32))
    summary={'rows':len(features),'contents':len(unique),'attributes':len(names),'raw_rank':fit.rank,'orthogonal_rank':int(B.shape[1]),'cross_content_centroid_accuracy':acc,'chance':1.0/len(names),'top_explained_variance':fit.explained_variance_ratio[:fit.rank].tolist(),'output':str(out)}
    Path(args.summary).write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
