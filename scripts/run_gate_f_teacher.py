#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import yaml

from speech_negotiation_kv.crad import load_crad
from speech_negotiation_kv.glm_voice import GLMVoiceBackend
from speech_negotiation_kv.sweep import MockSpeechBackend, run_one_turn_matched_sweep


def existing_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[int, int, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                keys.add((int(row["scenario_id"]), int(row["seed"]), str(row["style"])))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate F one-step matched vocal-strategy teacher sweep")
    parser.add_argument("--config", default="configs/crad_gate_f_16gb.yaml")
    parser.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    parser.add_argument("--scenario-start", type=int, default=None)
    parser.add_argument("--scenario-end", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--output", default="results/gate_f_train_teacher.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["experiment"]
    start = int(args.scenario_start if args.scenario_start is not None else exp["train_start"])
    end = int(args.scenario_end if args.scenario_end is not None else exp["train_end"])
    if not (0 <= start < end <= 80):
        parser.error("Gate-F one-step teacher ranges must satisfy 0 <= start < end <= 80")
    frame = load_crad(args.data)
    part = frame.iloc[start:end]
    styles = [str(style) for style in exp["styles"]]
    seeds = [int(seed) for seed in (args.seeds if args.seeds else exp["teacher_seeds"])]

    if args.dry_run:
        shifts = {style: -2 * index for index, style in enumerate(styles)}
        backend = MockSpeechBackend(style_offer_shift=shifts)
    else:
        model = cfg["model"]
        backend = GLMVoiceBackend(
            model_name=model["name"],
            quantization=model.get("quantization", "int4"),
            device=model.get("device", "cuda:0"),
            max_new_tokens=model.get("max_new_tokens", 96),
            temperature=model.get("temperature", 0.2),
            top_p=model.get("top_p", 0.8),
            audio_vocab_size=model.get("audio_vocab_size", 16384),
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    completed = existing_keys(output)
    expected = len(part) * len(seeds) * len(styles)
    written = 0
    with output.open("a", encoding="utf-8") as handle:
        for scenario_id, _series in part.iterrows():
            one = frame.loc[[scenario_id]]
            for seed in seeds:
                rows = run_one_turn_matched_sweep(one, backend=backend, styles=styles, seeds=[seed])
                for row in rows:
                    key = (int(row.scenario_id), int(row.seed), str(row.style))
                    if key in completed:
                        continue
                    handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed.add(key)
                    written += 1
                    if written % 20 == 0:
                        print(f"wrote {written} new branches; {len(completed)}/{expected} requested keys available")

    requested = {
        (int(scenario_id), int(seed), style)
        for scenario_id in part.index for seed in seeds for style in styles
    }
    available = len(completed & requested)
    print(json.dumps({
        "output": str(output),
        "scenario_range": [start, end],
        "new_branches": written,
        "available_requested_branches": available,
        "expected_branches": expected,
        "complete": bool(available == expected),
        "dry_run": bool(args.dry_run),
    }, indent=2))


if __name__ == "__main__":
    main()
