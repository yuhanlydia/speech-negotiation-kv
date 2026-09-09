#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.gate_e2 import terminal_state_metrics
from speech_negotiation_kv.records import read_jsonl


def bootstrap_mean_interval(
    values: list[float], *, seed: int = 4242424242, repeats: int = 10000
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": None, "lower_95": None, "upper_95": None, "repeats": repeats}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(repeats, len(array)), replace=True)
    means = samples.mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "repeats": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze confirmatory Gate E2")
    parser.add_argument("--config", default="configs/crad_gate_e2_16gb.yaml")
    parser.add_argument("--records", default="results/gate_e2_formal_v1.jsonl")
    parser.add_argument("--output", default="results/gate_e2_formal_v1_summary.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    experiment = config["experiment"]
    rows = read_jsonl(args.records)
    styles = {str(style) for style in experiment["styles"]}
    seeds = {int(seed) for seed in experiment["seeds"]}
    scenario_start = int(experiment.get("scenario_start", 0))
    scenario_ids = set(range(scenario_start, scenario_start + int(experiment["scenarios"])))
    expected_keys = {
        (scenario_id, seed, style)
        for scenario_id in scenario_ids
        for seed in seeds
        for style in styles
    }
    actual_keys = [
        (int(row["scenario_id"]), int(row["seed"]), str(row["style"]))
        for row in rows
    ]
    expected_branches = len(expected_keys)
    key_set_matches = len(actual_keys) == len(set(actual_keys)) == expected_branches and set(actual_keys) == expected_keys

    generated = sum(int(row.get("generated_move_count", 0)) for row in rows)
    parseable = sum(int(row.get("valid_move_count", 0)) for row in rows)
    strategic = sum(int(row.get("strategically_valid_move_count", 0)) for row in rows)
    matched_rate = float(np.mean([bool(row.get("matched_semantics")) for row in rows])) if rows else 0.0
    parseable_rate = float(parseable / generated) if generated else 0.0
    strategic_rate = float(strategic / generated) if generated else 0.0

    terminal_scenarios = {int(value) for value in experiment["terminal_scenarios"]}
    expected_terminal_keys = {
        key for key in expected_keys if key[0] in terminal_scenarios
    }
    terminal_rows = [
        row for row in rows
        if (int(row["scenario_id"]), int(row["seed"]), str(row["style"])) in expected_terminal_keys
        and bool(row.get("terminal_subset"))
    ]
    terminal_complete_rows = [
        row for row in terminal_rows if row.get("outcome") in {"agreement", "no_deal"}
    ]
    terminal_completion_rate = (
        float(len(terminal_complete_rows) / len(expected_terminal_keys))
        if expected_terminal_keys else 0.0
    )
    metrics = terminal_state_metrics(terminal_rows, expected_styles=styles)
    policy_valid_metrics = terminal_state_metrics(
        terminal_rows, expected_styles=styles, require_policy_valid=True
    )
    expected_states = len(terminal_scenarios & scenario_ids) * len(seeds)
    complete_state_rate = (
        float(metrics["n_complete_states"] / expected_states) if expected_states else 0.0
    )

    data_gate = bool(
        key_set_matches
        and matched_rate >= float(experiment["matched_transcript_min_rate"])
        and parseable_rate >= float(experiment["parseable_move_min_rate"])
        and strategic_rate >= float(experiment["strategically_valid_move_min_rate"])
        and terminal_completion_rate >= float(experiment["terminal_subset_min_completion_rate"])
        and metrics["n_complete_states"] >= int(np.ceil(
            float(experiment["terminal_state_min_completion_rate"]) * expected_states
        ))
    )
    correlations = [
        float(item["spearman"])
        for item in metrics["per_state"]
        if item["spearman"] is not None
    ]
    bootstrap = bootstrap_mean_interval(correlations, seed=int(experiment["base_seed"]))
    positive_alignment = bool(
        data_gate and bootstrap["lower_95"] is not None and bootstrap["lower_95"] > 0.0
    )
    median_regret = metrics["regret"]["median"]
    strong_alignment = bool(
        positive_alignment
        and bootstrap["lower_95"] is not None
        and bootstrap["lower_95"] > float(experiment["strong_spearman_lower_bound"])
        and metrics["top1_agreement_rate"] is not None
        and metrics["top1_agreement_rate"] > float(experiment["top1_min_rate"])
        and median_regret is not None
        and median_regret < float(experiment["median_regret_max"])
    )
    if not data_gate:
        interpretation = "inconclusive_data_gate_failed"
    elif strong_alignment:
        interpretation = "supports_strong_immediate_predicts_terminal"
    elif positive_alignment:
        interpretation = "supports_immediate_predicts_terminal"
    else:
        interpretation = "mixed_or_inconclusive_alignment"

    summary = {
        "formal_gate_e2": True,
        "confirmatory": True,
        "exploratory": False,
        "records": args.records,
        "protocol": {
            "scenario_ids": sorted(scenario_ids),
            "seeds": sorted(seeds),
            "styles": sorted(styles),
            "horizon": int(experiment["horizon"]),
            "terminal_max_horizon": int(experiment["terminal_max_horizon"]),
        },
        "expected_branches": expected_branches,
        "n_branches": len(rows),
        "key_set_matches": key_set_matches,
        "matched_semantics_rate": matched_rate,
        "parseable_move_rate": parseable_rate,
        "strategically_valid_move_rate": strategic_rate,
        "terminal_expected_branches": len(expected_terminal_keys),
        "terminal_branches": len(terminal_rows),
        "terminal_completion_rate": terminal_completion_rate,
        "terminal_outcomes": dict(Counter(str(row.get("outcome")) for row in terminal_rows)),
        "expected_terminal_states": expected_states,
        "complete_terminal_states": metrics["n_complete_states"],
        "complete_terminal_state_rate": complete_state_rate,
        "terminal_metrics": metrics,
        "policy_valid_terminal_metrics": policy_valid_metrics,
        "spearman_mean_bootstrap_95": bootstrap,
        "top1_agreement_rate": metrics["top1_agreement_rate"],
        "median_regret": median_regret,
        "data_gate_passes": data_gate,
        "positive_alignment_passes": positive_alignment,
        "strong_alignment_passes": strong_alignment,
        "interpretation": interpretation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "n_branches": len(rows),
        "terminal_completion_rate": terminal_completion_rate,
        "complete_terminal_states": metrics["n_complete_states"],
        "spearman_mean": bootstrap["mean"],
        "spearman_lower_95": bootstrap["lower_95"],
        "top1_agreement_rate": metrics["top1_agreement_rate"],
        "median_regret": median_regret,
        "data_gate_passes": data_gate,
        "strong_alignment_passes": strong_alignment,
        "interpretation": interpretation,
    }, indent=2))


if __name__ == "__main__":
    main()
