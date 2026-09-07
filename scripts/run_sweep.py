#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import yaml

from speech_negotiation_kv.crad import load_crad, split_crad
from speech_negotiation_kv.glm_voice import GLMVoiceBackend
from speech_negotiation_kv.records import write_jsonl
from speech_negotiation_kv.sweep import MockSpeechBackend, run_one_turn_matched_sweep


def main() -> None:
    ap = argparse.ArgumentParser(description="Matched same-semantics vocal sweep on CRAD")
    ap.add_argument("--config", default="configs/crad_16gb.yaml")
    ap.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--limit-scenarios", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--output", default="results/pilot_sweep.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = load_crad(args.data)
    train, test = split_crad(df)
    split_name = args.split or cfg["experiment"].get("split", "train")
    part = train if split_name == "train" else test
    limit = args.limit_scenarios or cfg["experiment"].get("pilot_scenarios")
    if limit:
        part = part.iloc[: int(limit)]
    styles = list(cfg["experiment"]["styles"])
    seeds = args.seeds if args.seeds else list(cfg["experiment"]["seeds"])

    if args.dry_run:
        shifts = {style: -2 * i for i, style in enumerate(styles)}
        backend = MockSpeechBackend(style_offer_shift=shifts)
    else:
        m = cfg["model"]
        backend = GLMVoiceBackend(
            model_name=m["name"],
            quantization=m.get("quantization", "int4"),
            device=m.get("device", "cuda:0"),
            max_new_tokens=m.get("max_new_tokens", 192),
            temperature=m.get("temperature", 0.2),
            top_p=m.get("top_p", 0.8),
            audio_vocab_size=m.get("audio_vocab_size", 16384),
        )

    rows = run_one_turn_matched_sweep(part, backend=backend, styles=styles, seeds=seeds)
    write_jsonl(args.output, rows)
    matched = sum(r.matched_semantics for r in rows) / max(1, len(rows))
    utilities = [r.utility for r in rows if r.matched_semantics]
    utility_std = statistics.pstdev(utilities) if len(utilities) > 1 else 0.0
    print(f"wrote {len(rows)} branches -> {args.output}")
    print(f"matched_semantics_rate={matched:.3f}; utility_std={utility_std:.4f}")
    min_rate = float(cfg["experiment"].get("matched_transcript_min_rate", 0.8))
    if matched < min_rate:
        print(f"GATE-A WARNING: matched transcript rate {matched:.3f} < required {min_rate:.3f}")


if __name__ == "__main__":
    main()
