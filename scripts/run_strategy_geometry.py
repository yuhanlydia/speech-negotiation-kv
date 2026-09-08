#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from speech_negotiation_kv.records import read_jsonl
from speech_negotiation_kv.strategy_geometry import (
    benjamini_hochberg,
    center_within_states,
    cosine_centroid_accuracy,
    pairwise_style_directions,
    shuffle_labels_within_states,
)
from speech_negotiation_kv.subspace import balanced_scenario_splits


def select_feature(memory: np.ndarray, layers: list[int], *, component: str,
                   layer: int | None) -> np.ndarray:
    if memory.ndim != 4 or memory.shape[2] != 2:
        raise ValueError("Gate D requires memory shaped [rows, layers, K/V, width]")
    positions = list(range(len(layers))) if layer is None else [layers.index(int(layer))]
    selected = memory[:, positions, :, :]
    if component == "k":
        selected = selected[:, :, 0, :]
    elif component == "v":
        selected = selected[:, :, 1, :]
    elif component != "kv":
        raise ValueError(f"unknown component: {component}")
    return selected.reshape(selected.shape[0], -1).astype(np.float64)


def aligned_arrays(records_path: str, kv_path: str, *, component: str,
                   layer: int | None) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    records = read_jsonl(records_path)
    kv = np.load(kv_path)
    layers = [int(layer_id) for layer_id in kv["layers"].tolist()]
    features = select_feature(kv["memory"], layers, component=component, layer=layer)
    feature_by_branch = {str(branch): features[index] for index, branch in enumerate(kv["branch_ids"])}
    matched = [record for record in records
               if record.get("matched_semantics", True) and str(record["branch_id"]) in feature_by_branch]
    arrays = {
        "state_ids": np.asarray([str(record["state_id"]) for record in matched]),
        "scenario_ids": np.asarray([int(record["scenario_id"]) for record in matched]),
        "styles": np.asarray([str(record["style"]) for record in matched]),
        "seeds": np.asarray([int(record["seed"]) for record in matched]),
    }
    X = np.stack([feature_by_branch[str(record["branch_id"])] for record in matched])
    metadata = {
        "observation": str(kv["observation"].item()) if "observation" in kv else "unknown",
        "pooling": str(kv["pooling"].item()) if "pooling" in kv else "unknown",
        "available_layers": layers,
    }
    return X, arrays, metadata


def distribution(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def decode_all_splits(features: np.ndarray, styles: np.ndarray, scenario_ids: np.ndarray,
                      splits: list[tuple[set[int], set[int]]]) -> tuple[list[float], np.ndarray]:
    classes = sorted(str(style) for style in np.unique(styles))
    class_index = {style: index for index, style in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    accuracies = []
    for first, second in splits:
        directional = []
        for train_scenarios in (first, second):
            accuracy, predictions = cosine_centroid_accuracy(features, styles, scenario_ids, train_scenarios)
            test_mask = ~np.isin(scenario_ids, list(train_scenarios))
            for truth, prediction in zip(styles[test_mask], predictions):
                confusion[class_index[str(truth)], class_index[str(prediction)]] += 1
            directional.append(accuracy)
        accuracies.append(float(np.mean(directional)))
    return accuracies, confusion


def pairwise_null(features: np.ndarray, styles: np.ndarray, scenario_ids: np.ndarray,
                  state_ids: np.ndarray, *, repeats: int, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    observed = pairwise_style_directions(features, styles, scenario_ids, state_ids)
    null = {pair: [] for pair in observed}
    for _ in range(repeats):
        shuffled = shuffle_labels_within_states(styles, state_ids, rng)
        shuffled_result = pairwise_style_directions(features, shuffled, scenario_ids, state_ids)
        for pair in observed:
            null[pair].append(float(shuffled_result[pair]["cross_scenario_mean_cosine"]))
    pairs = sorted(observed)
    raw_p = []
    for pair in pairs:
        observed_mean = float(observed[pair]["cross_scenario_mean_cosine"])
        null_values = np.asarray(null[pair], dtype=np.float64)
        raw_p.append(float((1 + np.sum(null_values >= observed_mean)) / (repeats + 1)))
    adjusted = benjamini_hochberg(raw_p)
    rows = []
    for pair, p_value, q_value in zip(pairs, raw_p, adjusted):
        item = observed[pair]
        rows.append({
            "first_style": pair[0],
            "second_style": pair[1],
            "n_directions": int(item["n_directions"]),
            "cross_scenario_mean_cosine": float(item["cross_scenario_mean_cosine"]),
            "cross_scenario_median_cosine": float(item["cross_scenario_median_cosine"]),
            "same_scenario_seed_mean_cosine": float(item["same_scenario_mean_cosine"]),
            "same_scenario_seed_median_cosine": float(item["same_scenario_median_cosine"]),
            "shuffled_mean_cosine": distribution(null[pair]),
            "permutation_p_value": p_value,
            "bh_fdr_q_value": float(q_value),
            "passes_fdr_0_05": bool(q_value <= 0.05 and item["cross_scenario_mean_cosine"] > 0),
        })
    return observed, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate D: cross-scenario vocal-strategy geometry")
    parser.add_argument("--records", default="results/gateA_glm_sweep.jsonl")
    parser.add_argument("--kv", default="results/gateA_action_audio_kv.npz")
    parser.add_argument("--component", choices=["k", "v", "kv"], default="kv")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--shuffle-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4242424242)
    parser.add_argument("--output", default="results/gate_d_strategy_geometry_summary.json")
    args = parser.parse_args()
    if args.shuffle_repeats < 1:
        parser.error("--shuffle-repeats must be positive")

    raw_features, arrays, metadata = aligned_arrays(
        args.records, args.kv, component=args.component, layer=args.layer
    )
    features = center_within_states(raw_features, arrays["state_ids"])
    scenario_ids = sorted(int(value) for value in np.unique(arrays["scenario_ids"]))
    splits = balanced_scenario_splits(scenario_ids)
    observed_accuracies, confusion = decode_all_splits(
        features, arrays["styles"], arrays["scenario_ids"], splits
    )

    rng = np.random.default_rng(args.seed)
    null_medians = []
    null_by_split = [[] for _ in splits]
    for _ in range(args.shuffle_repeats):
        shuffled = shuffle_labels_within_states(arrays["styles"], arrays["state_ids"], rng)
        null_accuracies, _ = decode_all_splits(features, shuffled, arrays["scenario_ids"], splits)
        null_medians.append(float(np.median(null_accuracies)))
        for index, accuracy in enumerate(null_accuracies):
            null_by_split[index].append(accuracy)
    observed_median = float(np.median(observed_accuracies))
    decode_p = float((1 + np.sum(np.asarray(null_medians) >= observed_median)) / (args.shuffle_repeats + 1))
    split_null_p95 = [float(np.quantile(values, 0.95)) for values in null_by_split]
    split_passes = int(sum(value > threshold for value, threshold in zip(observed_accuracies, split_null_p95)))

    _, pairwise = pairwise_null(
        features, arrays["styles"], arrays["scenario_ids"], arrays["state_ids"],
        repeats=args.shuffle_repeats, rng=rng,
    )
    n_styles = len(np.unique(arrays["styles"]))
    d1_pass = bool(observed_median > 1.0 / n_styles and decode_p <= 0.05)
    stable_pairs = [item for item in pairwise if item["passes_fdr_0_05"]]
    d2_pass = bool(stable_pairs)
    gate_pass = bool(d1_pass and d2_pass)

    output = {
        "hypothesis": "shared_strategy_space_context_dependent_value",
        "records": args.records,
        "kv": args.kv,
        **metadata,
        "component": args.component,
        "layer": args.layer,
        "n_rows": int(len(features)),
        "n_states": int(len(np.unique(arrays["state_ids"]))),
        "n_scenarios": len(scenario_ids),
        "n_styles": n_styles,
        "styles": sorted(str(style) for style in np.unique(arrays["styles"])),
        "n_unique_balanced_splits": len(splits),
        "split_evaluation": "bidirectional_mean_A_to_B_and_B_to_A",
        "shuffle_scheme": "independent_style_permutation_within_state",
        "shuffle_repeats": args.shuffle_repeats,
        "d1_style_decoding": {
            "chance_accuracy": 1.0 / n_styles,
            "held_out_accuracy": distribution(observed_accuracies),
            "null_median_accuracy": distribution(null_medians),
            "permutation_p_value": decode_p,
            "split_null_p95": distribution(split_null_p95),
            "splits_above_own_null_p95": split_passes,
            "fraction_splits_above_own_null_p95": float(split_passes / len(splits)),
            "classes": sorted(str(style) for style in np.unique(arrays["styles"])),
            "aggregate_bidirectional_confusion": confusion.tolist(),
            "passes": d1_pass,
        },
        "d2_pairwise_style_directions": {
            "multiple_testing": "Benjamini-Hochberg FDR 0.05 over 15 pairs",
            "n_pairs": len(pairwise),
            "n_pairs_passing_fdr": len(stable_pairs),
            "pairs": pairwise,
            "passes": d2_pass,
        },
        "gate_d_passes": gate_pass,
        "gate_d_rule": "D1 p<=0.05 and median>chance, plus at least one positive D2 pair at BH-FDR q<=0.05",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(path),
        "held_out_accuracy_median": observed_median,
        "decode_p_value": decode_p,
        "stable_pairs_fdr": len(stable_pairs),
        "gate_d_passes": gate_pass,
    }, indent=2))


if __name__ == "__main__":
    main()
