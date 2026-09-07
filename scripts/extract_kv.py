#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.crad import load_crad
from speech_negotiation_kv.glm_voice import GLMVoiceBackend, audio_ids_to_prompt
from speech_negotiation_kv.kv_hooks import FusedQKVRecorder
from speech_negotiation_kv.records import read_jsonl


def creditor_observation_prompt(scenario: dict, opponent_audio_ids: list[int]) -> str:
    private = (
        f"You are {scenario.get('Creditor Name', 'the creditor')} collecting a debt. "
        f"Your target is repayment within {int(scenario['Creditor Target Days'])} days. "
        f"The debtor prefers {int(scenario['Debtor Target Days'])} days. "
        "Listen to the debtor's speech and prepare your next negotiation action."
    )
    return (
        f"<|system|>\n{private}\n"
        f"<|user|>\n{audio_ids_to_prompt(opponent_audio_ids)}"
        "<|assistant|>streaming_transcription\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract pooled fused K/V features for sweep branches")
    ap.add_argument("--config", default="configs/crad_16gb.yaml")
    ap.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    ap.add_argument("--records", default="results/pilot_sweep.jsonl")
    ap.add_argument("--output", default="results/pilot_kv.npz")
    ap.add_argument("--include-unmatched", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = cfg["model"]
    layers = list(cfg["experiment"]["kv_layers"])
    backend = GLMVoiceBackend(
        model_name=mcfg["name"], quantization=mcfg.get("quantization", "int4"),
        device=mcfg.get("device", "cuda:0"), max_new_tokens=mcfg.get("max_new_tokens", 192),
        temperature=mcfg.get("temperature", 0.2), top_p=mcfg.get("top_p", 0.8),
        audio_vocab_size=mcfg.get("audio_vocab_size", 16384),
    )
    backend._ensure_loaded()
    model, tokenizer = backend.model, backend.tokenizer
    model_cfg = model.config
    df = load_crad(args.data)
    records = read_jsonl(args.records)
    branch_ids, memories = [], []

    recorder = FusedQKVRecorder(
        model,
        layer_indices=layers,
        num_attention_heads=int(model_cfg.num_attention_heads),
        kv_channels=int(model_cfg.kv_channels),
        multi_query_group_num=int(model_cfg.multi_query_group_num),
    )
    import torch
    for i, rec in enumerate(records):
        if not args.include_unmatched and not rec.get("matched_semantics", True):
            continue
        scenario = df.iloc[int(rec["scenario_id"])].to_dict()
        prompt = creditor_observation_prompt(scenario, rec["opponent_audio_token_ids"])
        inputs = tokenizer([prompt], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        recorder.clear()
        with torch.inference_mode(), recorder:
            model(**inputs, use_cache=False)
        branch_ids.append(rec["branch_id"])
        memories.append(recorder.vector().numpy())
        if (i + 1) % 25 == 0:
            print(f"extracted {i + 1}/{len(records)}")

    if not memories:
        raise RuntimeError("no K/V features extracted")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        branch_ids=np.asarray(branch_ids),
        memory=np.stack(memories).astype(np.float16),
        layers=np.asarray(layers, dtype=np.int16),
    )
    print(f"saved {len(branch_ids)} K/V vectors of dim {memories[0].shape[0]} -> {out}")


if __name__ == "__main__":
    main()
