#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.crad import load_crad
from speech_negotiation_kv.records import read_jsonl
from speech_negotiation_kv.short_horizon_selector import (
    evaluate_selector_states,
    fit_geometry_selector,
    fit_onehot_selector,
    opening_state_features,
    pairwise_unseen_style_accuracy,
    score_geometry_selector,
    score_onehot_selector,
    teacher_targets_and_weights,
)
from speech_negotiation_kv.subspace import spearman_rank_correlation


def complete_rows(rows: list[dict], styles: list[str]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("matched_semantics", True) and row.get("utility") is not None:
            groups[str(row["state_id"])].append(row)
    expected = set(styles)
    selected = []
    for state in sorted(groups):
        by_style = {str(row["style"]): row for row in groups[state]}
        if set(by_style) == expected:
            selected.extend(by_style[style] for style in styles)
    return selected


def load_artifact(path: str) -> dict:
    data = np.load(path)
    styles = [str(value) for value in data["styles"].tolist()]
    return {
        "styles": styles,
        "coordinates": {style: np.asarray(data["style_coordinates"][i], dtype=np.float64) for i, style in enumerate(styles)},
        "geometry": {
            "weights": data["geometry_weights"], "state_mean": data["geometry_state_mean"],
            "state_scale": data["geometry_state_scale"], "ridge": float(data["geometry_ridge"][0]),
        },
        "onehot": {
            "weights": data["onehot_weights"], "state_mean": data["onehot_state_mean"],
            "state_scale": data["onehot_state_scale"], "styles": styles,
            "ridge": float(data["onehot_ridge"][0]),
        },
    }


def arrays(rows: list[dict], frame, coordinates: dict[str, np.ndarray]):
    phi = np.stack([opening_state_features(
        float(frame.loc[int(row["scenario_id"]), "Creditor Target Days"]),
        float(frame.loc[int(row["scenario_id"]), "Debtor Target Days"]),
    ) for row in rows])
    labels = np.asarray([str(row["style"]) for row in rows])
    coords = np.stack([coordinates[str(row["style"])] for row in rows])
    y = np.asarray([float(row["utility"]) for row in rows])
    states = np.asarray([str(row["state_id"]) for row in rows])
    return phi, labels, coords, y, states


def rows_with_predictions(rows: list[dict], predictions: np.ndarray) -> list[dict]:
    return [{
        "state_id": str(row["state_id"]), "scenario_id": int(row["scenario_id"]),
        "style": str(row["style"]), "utility": float(row["utility"]),
        "prediction": float(pred),
    } for row, pred in zip(rows, predictions)]


def safe_spearman(truth: list[float], prediction: list[float]) -> float | None:
    if len(truth) < 2:
        return None
    try:
        value = float(spearman_rank_correlation(np.asarray(prediction), np.asarray(truth)))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate G1/G2 geometry controls")
    parser.add_argument("--config", default="configs/crad_gate_f_16gb.yaml")
    parser.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    parser.add_argument("--train-records", default="results/gate_f_train_teacher.jsonl")
    parser.add_argument("--robustness-records", default="results/gate_g_robustness_teacher.jsonl")
    parser.add_argument("--selector", default="results/gate_f_selector.npz")
    parser.add_argument("--output", default="results/gate_g_controls_summary.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    styles = [str(style) for style in cfg["experiment"]["styles"]]
    teacher_temperature = float(cfg["experiment"].get("teacher_temperature", 0.1))
    frame = load_crad(args.data)
    artifact = load_artifact(args.selector)
    if artifact["styles"] != styles:
        raise ValueError("selector styles do not match config styles")
    coordinates = artifact["coordinates"]
    train_rows = complete_rows(read_jsonl(args.train_records), styles)
    test_rows = complete_rows(read_jsonl(args.robustness_records), styles)
    train_phi, train_labels, train_c, train_y, train_states = arrays(train_rows, frame, coordinates)
    test_phi, test_labels, test_c, _, _ = arrays(test_rows, frame, coordinates)

    geometry_prediction = score_geometry_selector(artifact["geometry"], test_phi, test_c)
    onehot_prediction = score_onehot_selector(artifact["onehot"], test_phi, test_labels)
    g1 = {
        "geometry": evaluate_selector_states(rows_with_predictions(test_rows, geometry_prediction), expected_styles=set(styles)),
        "onehot": evaluate_selector_states(rows_with_predictions(test_rows, onehot_prediction), expected_styles=set(styles)),
        "geometry_parameter_count": int(len(artifact["geometry"]["weights"])),
        "onehot_parameter_count": int(len(artifact["onehot"]["weights"])),
    }

    g2 = []
    for held_out in styles:
        seen = [style for style in styles if style != held_out]
        mask = train_labels != held_out
        subset_teacher_targets, subset_row_weights = teacher_targets_and_weights(
            train_y[mask], train_states[mask], temperature=teacher_temperature
        )
        geometry = fit_geometry_selector(
            train_phi[mask], train_c[mask], subset_teacher_targets, train_states[mask],
            ridge=artifact["geometry"]["ridge"], row_weights=subset_row_weights,
        )
        onehot = fit_onehot_selector(
            train_phi[mask], train_labels[mask], subset_teacher_targets, train_states[mask],
            styles=seen, ridge=artifact["onehot"]["ridge"], row_weights=subset_row_weights,
        )
        geometry_pred = score_geometry_selector(geometry, test_phi, test_c)

        onehot_pred = np.empty(len(test_rows), dtype=np.float64)
        by_state_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(test_rows):
            by_state_indices[str(row["state_id"])].append(index)
        for indices in by_state_indices.values():
            seen_indices = [i for i in indices if str(test_rows[i]["style"]) != held_out]
            seen_phi = test_phi[seen_indices]
            seen_labels = test_labels[seen_indices]
            seen_scores = score_onehot_selector(onehot, seen_phi, seen_labels)
            score_by_index = dict(zip(seen_indices, seen_scores))
            fallback = float(np.mean(seen_scores))
            for i in indices:
                onehot_pred[i] = fallback if str(test_rows[i]["style"]) == held_out else float(score_by_index[i])

        geometry_rows = rows_with_predictions(test_rows, geometry_pred)
        onehot_rows = rows_with_predictions(test_rows, onehot_pred)
        held_truth = [float(row["utility"]) for row in geometry_rows if row["style"] == held_out]
        held_geom = [float(row["prediction"]) for row in geometry_rows if row["style"] == held_out]
        held_onehot = [float(row["prediction"]) for row in onehot_rows if row["style"] == held_out]
        g2.append({
            "held_out_style": held_out,
            "geometry": {
                "all_six": evaluate_selector_states(geometry_rows, expected_styles=set(styles)),
                "held_out_style_spearman": safe_spearman(held_truth, held_geom),
                "pairwise_sign_accuracy": pairwise_unseen_style_accuracy(
                    geometry_rows, held_out_style=held_out, seen_styles=set(seen)
                ),
                "parameter_count": int(geometry["parameter_count"]),
            },
            "onehot_fallback": {
                "all_six": evaluate_selector_states(onehot_rows, expected_styles=set(styles)),
                "held_out_style_spearman": safe_spearman(held_truth, held_onehot),
                "pairwise_sign_accuracy": pairwise_unseen_style_accuracy(
                    onehot_rows, held_out_style=held_out, seen_styles=set(seen)
                ),
                "parameter_count": int(onehot["parameter_count"]),
                "fallback": "state mean over seen-style scores",
            },
        })

    result = {
        "gate": "G1_G2",
        "train_records": args.train_records,
        "robustness_records": args.robustness_records,
        "n_train_states": len(set(train_states.tolist())),
        "n_robustness_states": len({str(row["state_id"]) for row in test_rows}),
        "g1_geometry_necessity": g1,
        "g2_leave_one_style_out": g2,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "g1_geometry_top1": g1["geometry"]["top1_agreement_rate"],
        "g1_onehot_top1": g1["onehot"]["top1_agreement_rate"],
        "g2_styles": len(g2),
    }, indent=2))


if __name__ == "__main__":
    main()
