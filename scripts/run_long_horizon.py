#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from speech_negotiation_kv.crad import load_crad, split_crad
from speech_negotiation_kv.glm_voice import GLMVoiceBackend
from speech_negotiation_kv.long_horizon import MockLongHorizonBackend, run_long_horizon_branch


def existing_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                keys.add((int(row["scenario_id"]), int(row["seed"]), str(row["style"])))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate E paired long-horizon CRAD speech sweep")
    parser.add_argument("--config", default="configs/crad_gate_e_16gb.yaml")
    parser.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    parser.add_argument("--output", default="results/gate_e_long_horizon.jsonl")
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--scenario-start", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="refuse to overwrite an existing output")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    experiment = cfg["experiment"]
    frame = load_crad(args.data)
    train, test = split_crad(frame)
    part = train if experiment.get("split", "train") == "train" else test
    scenario_count = int(args.limit_scenarios or experiment["scenarios"])
    scenario_start = int(
        args.scenario_start if args.scenario_start is not None
        else experiment.get("scenario_start", 0)
    )
    part = part.iloc[scenario_start:scenario_start + scenario_count]
    styles = [str(style) for style in experiment["styles"]]
    seeds = list(args.seeds) if args.seeds else [int(seed) for seed in experiment["seeds"]]
    horizon = int(experiment["horizon"])
    terminal_max_horizon = int(experiment["terminal_max_horizon"])
    terminal_scenarios = {int(value) for value in experiment["terminal_scenarios"]}
    base_seed = int(experiment["base_seed"])

    if args.dry_run:
        backend = MockLongHorizonBackend(styles)
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
    expected = len(part) * len(styles) * len(seeds)
    written = 0
    with output.open("a", encoding="utf-8") as handle:
        for scenario_id, series in part.iterrows():
            scenario = series.to_dict()
            for seed in seeds:
                for style in styles:
                    key = (int(scenario_id), int(seed), style)
                    if key in completed:
                        continue
                    result = run_long_horizon_branch(
                        scenario,
                        scenario_id=int(scenario_id),
                        style=style,
                        branch_seed=int(seed),
                        backend=backend,
                        horizon=horizon,
                        max_horizon=(terminal_max_horizon if int(scenario_id) in terminal_scenarios else horizon),
                        base_seed=base_seed,
                    )
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed.add(key)
                    written += 1
                    if written % 10 == 0:
                        print(f"wrote {written} new branches; {len(completed)}/{expected} requested keys available")
    print(json.dumps({
        "output": str(output),
        "new_branches": written,
        "available_requested_branches": sum(
            (int(scenario_id), int(seed), style) in completed
            for scenario_id in part.index for seed in seeds for style in styles
        ),
        "expected_branches": expected,
        "dry_run": bool(args.dry_run),
    }, indent=2))


if __name__ == "__main__":
    main()
