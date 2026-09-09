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
    prototype_style_coordinates,
    score_geometry_selector,
    score_onehot_selector,
    teacher_targets_and_weights,
)
from speech_negotiation_kv.strategy_geometry import center_within_states


def complete_rows(rows: list[dict], styles: list[str]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    expected = set(styles)
    for row in rows:
        if row.get("matched_semantics", True) and row.get("utility") is not None:
            groups[str(row["state_id"])].append(row)
    out = []
    for state_id in sorted(groups):
        by_style = {str(row["style"]): row for row in groups[state_id]}
        if set(by_style) == expected:
            out.extend(by_style[style] for style in styles)
    return out


def load_style_coordinates(records_path: str, kv_path: str, styles: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    records = read_jsonl(records_path)
    kv = np.load(kv_path)
    memory = np.asarray(kv["memory"], dtype=np.float64)
    if memory.ndim != 4 or memory.shape[2] != 2:
        raise ValueError("Gate-D action K/V must have shape [rows, layers, K/V, width]")
    flat = memory.reshape(memory.shape[0], -1)
    feature_by_branch = {str(branch): flat[index] for index, branch in enumerate(kv["branch_ids"])}
    matched = [row for row in records if row.get("matched_semantics", True) and str(row["branch_id"]) in feature_by_branch]
    X = np.stack([feature_by_branch[str(row["branch_id"])] for row in matched])
    state_ids = np.asarray([str(row["state_id"]) for row in matched])
    labels = np.asarray([str(row["style"]) for row in matched])
    centered = center_within_states(X, state_ids)
    coordinates, metadata = prototype_style_coordinates(centered, labels, styles, rank=len(styles) - 1)
    metadata.update({
        "source": "Gate-D action-audio K/V centered style prototypes",
        "observation": str(kv["observation"].item()) if "observation" in kv else "unknown",
        "pooling": str(kv["pooling"].item()) if "pooling" in kv else "unknown",
        "layers": [int(x) for x in kv["layers"].tolist()] if "layers" in kv else [],
    })
    return coordinates, metadata


def row_arrays(rows: list[dict], frame, coordinates: dict[str, np.ndarray]):
    phi = np.stack([
        opening_state_features(
            float(frame.loc[int(row["scenario_id"]), "Creditor Target Days"]),
            float(frame.loc[int(row["scenario_id"]), "Debtor Target Days"]),
        ) for row in rows
    ])
    labels = np.asarray([str(row["style"]) for row in rows])
    coords = np.stack([coordinates[label] for label in labels])
    y = np.asarray([float(row["utility"]) for row in rows], dtype=np.float64)
    state_ids = np.asarray([str(row["state_id"]) for row in rows])
    return phi, labels, coords, y, state_ids


def metric_key(metric: dict) -> tuple[float, float, float]:
    rho = metric["spearman"]["mean"]
    top1 = metric["top1_agreement_rate"]
    regret = metric["regret"]["mean"]
    return (
        float(rho) if rho is not None else -1e9,
        float(top1) if top1 is not None else -1e9,
        -float(regret) if regret is not None else -1e9,
    )


def prediction_rows(rows: list[dict], predictions: np.ndarray) -> list[dict]:
    return [
        {
            "state_id": str(row["state_id"]),
            "scenario_id": int(row["scenario_id"]),
            "style": str(row["style"]),
            "utility": float(row["utility"]),
            "prediction": float(prediction),
        }
        for row, prediction in zip(rows, predictions)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit formal Gate-F one-step strategy selectors")
    parser.add_argument("--config", default="configs/crad_gate_f_16gb.yaml")
    parser.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    parser.add_argument("--train-records", default="results/gate_f_train_teacher.jsonl")
    parser.add_argument("--validation-records", default="results/gate_f_validation_teacher.jsonl")
    parser.add_argument("--gate-d-records", default="results/gateA_glm_sweep.jsonl")
    parser.add_argument("--gate-d-kv", default="results/gateA_action_audio_kv.npz")
    parser.add_argument("--model-output", default="results/gate_f_selector.npz")
    parser.add_argument("--summary-output", default="results/gate_f_selector_summary.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["experiment"]
    styles = [str(style) for style in exp["styles"]]
    ridge_grid = [float(value) for value in exp.get("ridge_grid", [0.01, 0.1, 1.0, 10.0, 100.0])]
    teacher_temperature = float(exp.get("teacher_temperature", 0.1))
    if teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")
    frame = load_crad(args.data)
    train_rows = complete_rows(read_jsonl(args.train_records), styles)
    val_rows = complete_rows(read_jsonl(args.validation_records), styles)
    if not train_rows or not val_rows:
        raise ValueError("train and validation teacher files must contain complete six-style states")

    coordinates, coordinate_metadata = load_style_coordinates(args.gate_d_records, args.gate_d_kv, styles)
    train_phi, train_labels, train_c, train_y, train_states = row_arrays(train_rows, frame, coordinates)
    train_teacher_targets, train_row_weights = teacher_targets_and_weights(
        train_y, train_states, temperature=teacher_temperature
    )
    val_phi, val_labels, val_c, _, _ = row_arrays(val_rows, frame, coordinates)

    best_geometry = None
    best_geometry_metric = None
    best_geometry_ridge = None
    geometry_grid = []
    for ridge in ridge_grid:
        model = fit_geometry_selector(
            train_phi, train_c, train_teacher_targets, train_states,
            ridge=ridge, row_weights=train_row_weights,
        )
        predictions = score_geometry_selector(model, val_phi, val_c)
        metric = evaluate_selector_states(prediction_rows(val_rows, predictions), expected_styles=set(styles))
        geometry_grid.append({"ridge": ridge, "metrics": metric})
        if best_geometry_metric is None or metric_key(metric) > metric_key(best_geometry_metric):
            best_geometry, best_geometry_metric, best_geometry_ridge = model, metric, ridge

    best_onehot = None
    best_onehot_metric = None
    best_onehot_ridge = None
    onehot_grid = []
    for ridge in ridge_grid:
        model = fit_onehot_selector(
            train_phi, train_labels, train_teacher_targets, train_states,
            styles=styles, ridge=ridge, row_weights=train_row_weights,
        )
        predictions = score_onehot_selector(model, val_phi, val_labels)
        metric = evaluate_selector_states(prediction_rows(val_rows, predictions), expected_styles=set(styles))
        onehot_grid.append({"ridge": ridge, "metrics": metric})
        if best_onehot_metric is None or metric_key(metric) > metric_key(best_onehot_metric):
            best_onehot, best_onehot_metric, best_onehot_ridge = model, metric, ridge

    style_means = {
        style: float(np.mean([float(row["utility"]) for row in train_rows if str(row["style"]) == style]))
        for style in styles
    }
    lookup_predictions = np.asarray([style_means[str(row["style"])] for row in val_rows])
    lookup_metric = evaluate_selector_states(prediction_rows(val_rows, lookup_predictions), expected_styles=set(styles))
    best_fixed_style = max(styles, key=lambda style: (style_means[style], -styles.index(style)))

    style_matrix = np.stack([coordinates[style] for style in styles])
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        styles=np.asarray(styles),
        style_coordinates=style_matrix.astype(np.float32),
        geometry_weights=np.asarray(best_geometry["weights"], dtype=np.float64),
        geometry_state_mean=np.asarray(best_geometry["state_mean"], dtype=np.float64),
        geometry_state_scale=np.asarray(best_geometry["state_scale"], dtype=np.float64),
        geometry_ridge=np.asarray([best_geometry_ridge], dtype=np.float64),
        onehot_weights=np.asarray(best_onehot["weights"], dtype=np.float64),
        onehot_state_mean=np.asarray(best_onehot["state_mean"], dtype=np.float64),
        onehot_state_scale=np.asarray(best_onehot["state_scale"], dtype=np.float64),
        onehot_ridge=np.asarray([best_onehot_ridge], dtype=np.float64),
        style_lookup=np.asarray([style_means[style] for style in styles], dtype=np.float64),
        best_fixed_style=np.asarray(best_fixed_style),
        teacher_temperature=np.asarray([teacher_temperature], dtype=np.float64),
    )

    summary = {
        "method": "short_horizon_strategy_distillation",
        "train_records": args.train_records,
        "validation_records": args.validation_records,
        "n_train_states": len({str(row["state_id"]) for row in train_rows}),
        "n_validation_states": len({str(row["state_id"]) for row in val_rows}),
        "styles": styles,
        "teacher_temperature": teacher_temperature,
        "teacher_target": "softmax(centered immediate utility / temperature)",
        "teacher_weight": "within-state immediate utility spread",
        "coordinate_metadata": coordinate_metadata,
        "geometry": {
            "selected_ridge": best_geometry_ridge,
            "parameter_count": int(best_geometry["parameter_count"]),
            "validation": best_geometry_metric,
            "grid": geometry_grid,
        },
        "onehot": {
            "selected_ridge": best_onehot_ridge,
            "parameter_count": int(best_onehot["parameter_count"]),
            "validation": best_onehot_metric,
            "grid": onehot_grid,
        },
        "style_lookup": {
            "values": style_means,
            "best_fixed_style": best_fixed_style,
            "validation": lookup_metric,
        },
        "final_test_untouched": True,
        "final_test_scenarios": [80, 99],
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "model_output": str(model_path),
        "summary_output": str(summary_path),
        "geometry_ridge": best_geometry_ridge,
        "geometry_top1": best_geometry_metric["top1_agreement_rate"],
        "onehot_top1": best_onehot_metric["top1_agreement_rate"],
        "best_fixed_style": best_fixed_style,
    }, indent=2))


if __name__ == "__main__":
    main()
