#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import yaml

from speech_negotiation_kv.icassp_eval import paired_wer_degradation, sample_preference_scores
from speech_negotiation_kv.static_power_eval import hierarchical_macro_preference, pool_seed_records


_AUDIO_STEM = re.compile(r"^(static_power_\d{3})_")


def load_dataset(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_stem = {Path(row["audio_path"]).stem: row for row in rows}
    if len(rows) != 180 or len(by_stem) != 180:
        raise ValueError(f"dataset must contain 180 uniquely named rows, got {len(rows)}/{len(by_stem)}")
    return rows, by_stem


def load_judge_records(metadata_dir: Path, by_stem: dict[str, dict]) -> tuple[list[dict], list[str]]:
    records, failures = [], []
    for path in sorted(metadata_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        match = _AUDIO_STEM.match(path.stem)
        if not match or match.group(1) not in by_stem:
            failures.append(f"unmapped_metadata:{path.name}")
            continue
        scores = sample_preference_scores(payload)
        if len(scores) != 1:
            failures.append(f"missing_decision:{path.name}")
            continue
        item = by_stem[match.group(1)]
        records.append({
            "attribute_key": item["attribute_key"],
            "item_id": item["item_id"],
            "score": scores[0],
        })
    observed = {row["item_id"] for row in records}
    failures.extend(f"missing_metadata:{row['item_id']}" for row in by_stem.values() if row["item_id"] not in observed)
    return records, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize balanced static ParaGeo-vs-random power experiment")
    parser.add_argument("--config", default="configs/icassp2027_static_power.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp, root = cfg["experiment"], Path(cfg["outputs"]["root"])
    output = Path(args.output) if args.output else root / "summary.json"
    dataset, by_stem = load_dataset(Path(cfg["dataset"]["manifest"]))
    records_by_seed, per_seed, failures = {}, {}, {}
    for index, seed_value in enumerate(exp["random_seeds"]):
        seed = int(seed_value)
        records, seed_failures = load_judge_records(
            root / "judge" / f"main_vs_random_seed_{seed}" / "metadata", by_stem
        )
        records_by_seed[seed] = records
        result = hierarchical_macro_preference(
            records, repeats=int(exp["bootstrap_repeats"]), seed=int(exp["bootstrap_seed"]) + index
        )
        scores = [row["score"] for row in records]
        result.update({
            "wins": sum(score == 1.0 for score in scores),
            "ties": sum(score == 0.5 for score in scores),
            "losses": sum(score == 0.0 for score in scores),
            "complete": len(records) == len(dataset) and not seed_failures,
        })
        per_seed[str(seed)] = result
        failures[str(seed)] = seed_failures
    pooled_records = pool_seed_records(records_by_seed)
    pooled = hierarchical_macro_preference(
        pooled_records, repeats=int(exp["bootstrap_repeats"]), seed=int(exp["bootstrap_seed"]) + 100
    )
    complete_seed_counts = {row["seed_count"] for row in pooled_records}
    positive_seeds = sum(
        result["macro_preference"] is not None and result["macro_preference"] > 50.0
        for result in per_seed.values()
    )
    all_complete = all(result["complete"] for result in per_seed.values())
    claim_passes = bool(
        all_complete and complete_seed_counts == {5} and pooled["lower_95"] > 50.0 and positive_seeds >= 4
    )
    fidelity = {}
    main_manifest = root / "generation" / "main" / "manifest.jsonl"
    for seed_value in exp["random_seeds"]:
        seed = int(seed_value)
        fidelity[str(seed)] = paired_wer_degradation(
            main_manifest, root / "generation" / f"random_seed_{seed}" / "manifest.jsonl"
        )
    payload = {
        "protocol": "balanced 18-attribute static direct ParaGeo-vs-five-random-seeds evaluation",
        "experiment_version": exp["version"],
        "alpha": float(exp["alpha"]),
        "items": len(dataset),
        "attributes": len({row["attribute_key"] for row in dataset}),
        "items_per_attribute": int(exp["items_per_attribute"]),
        "random_seeds": [int(value) for value in exp["random_seeds"]],
        "per_seed": per_seed,
        "pooled": pooled,
        "positive_seeds": positive_seeds,
        "fidelity_main_minus_random": fidelity,
        "failures": failures,
        "all_complete": all_complete,
        "claim_rule": "pooled hierarchical 95% CI above 50% and >50% macro preference for at least 4/5 seeds",
        "claim_passes": claim_passes,
        "decision": "parageo_beats_random" if claim_passes else "no_reliable_parageo_advantage",
        "limitations": ["All expanded input instructions use one synthetic neutral TTS voice."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    paper_dir = Path("paper/generated")
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.joinpath("static_power_results.tex").write_text(
        "\\newcommand{\\StaticPowerMacro}{" + f"{pooled['macro_preference']:.1f}" + "}\n"
        "\\newcommand{\\StaticPowerCI}{[" + f"{pooled['lower_95']:.1f}, {pooled['upper_95']:.1f}" + "]}\n"
        "\\newcommand{\\StaticPowerPositiveSeeds}{" + str(positive_seeds) + "/5}\n"
        "\\newcommand{\\StaticPowerDecision}{" + payload["decision"].replace("_", "\\_") + "}\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "decision": payload["decision"], "claim_passes": claim_passes}, indent=2))


if __name__ == "__main__":
    main()
