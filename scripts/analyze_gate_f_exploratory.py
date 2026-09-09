#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.crad import load_crad, split_crad
from speech_negotiation_kv.gate_f import evaluate_state_predictions, interaction_features, ridge_fit_predict
from speech_negotiation_kv.records import read_jsonl
from speech_negotiation_kv.subspace import spearman_rank_correlation
from run_strategy_geometry import aligned_arrays


def complete_terminal_rows(records: list[dict], styles: set[str]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        if row.get("terminal_subset") and row.get("policy_valid") and row.get("outcome") in {"agreement", "no_deal"}:
            groups[str(row["state_id"])].append(row)
    selected = []
    for rows in groups.values():
        if {str(row["style"]) for row in rows} == styles and len(rows) == len(styles):
            selected.extend(rows)
    return selected


def style_coordinates(records_path: str, kv_path: str, styles: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    raw, arrays, metadata = aligned_arrays(records_path, kv_path, component="kv", layer=None)
    prototypes = []
    for style in styles:
        mask = arrays["styles"] == style
        if not mask.any():
            raise ValueError(f"Gate D data has no style {style!r}")
        prototypes.append(raw[mask].mean(axis=0))
    matrix = np.asarray(prototypes, dtype=np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(len(styles) - 1, vt.shape[0])
    coords = matrix @ vt[:rank].T
    return {style: coords[index] for index, style in enumerate(styles)}, {
        "source": "Gate-D action-audio K/V style prototypes",
        "rank": int(rank),
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory Gate F contextual-value analysis")
    parser.add_argument("--config", default="configs/crad_gate_e_16gb.yaml")
    parser.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    parser.add_argument("--records", default="results/gate_e_formal_v7.jsonl")
    parser.add_argument("--gate-d-records", default="results/gateA_glm_sweep.jsonl")
    parser.add_argument("--gate-d-kv", default="results/gateA_action_audio_kv.npz")
    parser.add_argument("--output", default="results/gate_f_exploratory_summary.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    styles = [str(style) for style in config["experiment"]["styles"]]
    style_set = set(styles)
    rows = complete_terminal_rows(read_jsonl(args.records), style_set)
    if not rows:
        raise ValueError("no complete terminal states available")
    state_ids = sorted({str(row["state_id"]) for row in rows})
    state_scenarios = {str(row["state_id"]): int(row["scenario_id"]) for row in rows}
    scenarios = sorted({state_scenarios[state] for state in state_ids})
    train_frame, _ = split_crad(load_crad(args.data))
    coordinates, coordinate_metadata = style_coordinates(args.gate_d_records, args.gate_d_kv, styles)
    coordinate_matrix = np.stack([coordinates[style] for style in styles])

    feature_by_state = {}
    for state_id in state_ids:
        scenario = train_frame.iloc[state_scenarios[state_id]]
        creditor = float(scenario["Creditor Target Days"])
        debtor = float(scenario["Debtor Target Days"])
        feature_by_state[state_id] = np.asarray([creditor, debtor, debtor - creditor, creditor, 0.0, 0.0])

    model_rows: dict[str, list[dict]] = defaultdict(list)
    for held_out in scenarios:
        test_states = [state for state in state_ids if state_scenarios[state] == held_out]
        train_states = [state for state in state_ids if state_scenarios[state] != held_out]
        train_rows = [row for row in rows if str(row["state_id"]) in train_states]
        test_rows = [row for row in rows if str(row["state_id"]) in test_states]
        train_phi = np.stack([feature_by_state[str(row["state_id"])] for row in train_rows])
        test_phi = np.stack([feature_by_state[str(row["state_id"])] for row in test_rows])
        train_c = np.stack([coordinates[str(row["style"])] for row in train_rows])
        test_c = np.stack([coordinates[str(row["style"])] for row in test_rows])
        y_train = np.asarray([float(row["utility"]) for row in train_rows])

        style_means = {
            style: float(np.mean([float(row["utility"]) for row in train_rows if row["style"] == style]))
            for style in styles
        }
        predictions = {
            "style_lookup": np.asarray([style_means[str(row["style"])] for row in test_rows]),
            "global_strategy": ridge_fit_predict(train_c, y_train, test_c, ridge=1.0),
            "state_only": ridge_fit_predict(train_phi, y_train, test_phi, ridge=10.0),
            "contextual_bilinear": ridge_fit_predict(
                interaction_features(train_phi, train_c), y_train,
                interaction_features(test_phi, test_c), ridge=10.0,
            ),
        }
        for model, values in predictions.items():
            for row, prediction in zip(test_rows, values):
                model_rows[model].append({
                    "state_id": str(row["state_id"]),
                    "scenario_id": int(row["scenario_id"]),
                    "style": str(row["style"]),
                    "utility": float(row["utility"]),
                    "prediction": float(prediction),
                })

    metrics = {
        model: evaluate_state_predictions(model_rows[model], expected_styles=style_set)
        for model in model_rows
    }
    output = {
        "exploratory": True,
        "formal_gate_e_passed": False,
        "paper_pass_authorized": False,
        "records": args.records,
        "gate_d_records": args.gate_d_records,
        "gate_d_kv": args.gate_d_kv,
        "n_complete_terminal_states": len(state_ids),
        "n_terminal_scenarios": len(scenarios),
        "held_out_split": "leave_one_terminal_scenario_out",
        "state_features": ["creditor_target", "debtor_target", "bargaining_gap", "opening_offer", "previous_concession", "round_fraction"],
        "style_coordinate_metadata": coordinate_metadata,
        "models": {
            "style_lookup": "global per-style mean",
            "global_strategy": "ridge utility from Gate-D style coordinate c_z",
            "state_only": "ridge utility from phi(s)",
            "contextual_bilinear": "ridge utility from phi(s) tensor c_z",
            "full_kv": "unavailable: formal Gate-E records contain no K/V captures",
        },
        "metrics": metrics,
        "interpretation": "exploratory_only_due_to_gate_e_data_gate_failure",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(output_path),
        "complete_terminal_states": len(state_ids),
        "models": {model: {
            "spearman_median": values["spearman"]["median"],
            "top1_accuracy": values["top1_accuracy"],
        } for model, values in metrics.items()},
        "exploratory": True,
        "paper_pass_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
