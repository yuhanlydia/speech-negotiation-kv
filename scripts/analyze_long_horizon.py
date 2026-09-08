#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.long_horizon_analysis import style_ranking_correlations
from speech_negotiation_kv.records import read_jsonl


def bootstrap_mean_interval(values: list[float], *, seed: int = 4242424242,
                            repeats: int = 10000) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": None, "lower_95": None, "upper_95": None, "repeats": repeats}
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(array, size=len(array), replace=True).mean()
        for _ in range(repeats)
    ])
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "repeats": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Gate E long-horizon trajectories")
    parser.add_argument("--config", default="configs/crad_gate_e_16gb.yaml")
    parser.add_argument("--records", default="results/gate_e_long_horizon.jsonl")
    parser.add_argument("--output", default="results/gate_e_summary.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    experiment = config["experiment"]
    rows = read_jsonl(args.records)
    styles = {str(style) for style in experiment["styles"]}
    terminal_rows = [row for row in rows if row.get("terminal_subset")]
    valid_terminal_rows = [row for row in terminal_rows if row.get("policy_valid")]
    matched_rate = float(np.mean([row.get("matched_semantics", False) for row in rows])) if rows else 0.0
    generated = sum(int(row.get("generated_move_count", 0)) for row in rows)
    valid = sum(int(row.get("valid_move_count", 0)) for row in rows)
    parseable_rate = float(valid / generated) if generated else 0.0
    strategic_valid = sum(int(row.get("strategically_valid_move_count", 0)) for row in rows)
    strategically_valid_rate = float(strategic_valid / generated) if generated else 0.0
    horizon_complete = sum(row.get("horizon_outcome") in {"agreement", "no_deal"} for row in rows)
    terminal_complete = sum(row.get("outcome") in {"agreement", "no_deal"} for row in terminal_rows)
    horizon_ranking = style_ranking_correlations(rows, expected_styles=styles, outcome_key="horizon_utility")
    terminal_ranking = style_ranking_correlations(
        valid_terminal_rows, expected_styles=styles, outcome_key="utility"
    )
    correlations = [
        float(item["spearman"])
        for item in terminal_ranking["per_state"]
        if item["spearman"] is not None
    ]
    expected_branches = int(experiment["scenarios"]) * len(styles) * len(experiment["seeds"])
    expected_terminal = len(experiment["terminal_scenarios"]) * len(styles) * len(experiment["seeds"])
    terminal_completion_rate = float(terminal_complete / len(terminal_rows)) if terminal_rows else 0.0
    expected_terminal_states = len(experiment["terminal_scenarios"]) * len(experiment["seeds"])
    complete_state_rate = float(
        terminal_ranking["n_complete_states"] / expected_terminal_states
    ) if expected_terminal_states else 0.0
    data_gate = bool(
        len(rows) == expected_branches
        and matched_rate >= float(experiment["matched_transcript_min_rate"])
        and parseable_rate >= float(experiment["parseable_move_min_rate"])
        and strategically_valid_rate >= float(experiment["strategically_valid_move_min_rate"])
        and terminal_completion_rate >= float(experiment["terminal_subset_min_completion_rate"])
        and complete_state_rate >= float(experiment["terminal_subset_min_completion_rate"])
    )
    bootstrap = bootstrap_mean_interval(correlations)
    if not data_gate:
        interpretation = "inconclusive_data_gate_failed"
    elif bootstrap["upper_95"] is not None and bootstrap["upper_95"] < 0.5:
        interpretation = "supports_immediate_influence_long_horizon_value_gap"
    elif bootstrap["lower_95"] is not None and bootstrap["lower_95"] > 0.0:
        interpretation = "supports_positive_immediate_to_long_horizon_alignment"
    else:
        interpretation = "mixed_or_inconclusive_alignment"

    summary = {
        "records": args.records,
        "expected_branches": expected_branches,
        "n_branches": len(rows),
        "matched_semantics_rate": matched_rate,
        "parseable_move_rate": parseable_rate,
        "strategically_valid_move_rate": strategically_valid_rate,
        "policy_valid_branch_rate": float(np.mean([row.get("policy_valid", False) for row in rows])) if rows else 0.0,
        "horizon_completion_rate": float(horizon_complete / len(rows)) if rows else 0.0,
        "horizon_outcomes": dict(Counter(str(row.get("horizon_outcome")) for row in rows)),
        "terminal_subset_expected_branches": expected_terminal,
        "terminal_subset_branches": len(terminal_rows),
        "terminal_subset_completion_rate": terminal_completion_rate,
        "terminal_subset_outcomes": dict(Counter(str(row.get("outcome")) for row in terminal_rows)),
        "terminal_subset_complete_state_rate": complete_state_rate,
        "horizon_style_ranking": horizon_ranking,
        "terminal_style_ranking": terminal_ranking,
        "terminal_spearman_mean_bootstrap_95": bootstrap,
        "data_gate_passes": data_gate,
        "interpretation": interpretation,
        "gate_f_authorized": bool(data_gate),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "n_branches": len(rows),
        "terminal_completion_rate": terminal_completion_rate,
        "complete_terminal_states": terminal_ranking["n_complete_states"],
        "terminal_spearman_median": terminal_ranking["spearman"]["median"],
        "data_gate_passes": data_gate,
        "interpretation": interpretation,
    }, indent=2))


if __name__ == "__main__":
    main()
