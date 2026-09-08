#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from speech_negotiation_kv.records import read_jsonl
from speech_negotiation_kv.subspace import (
    advantage_memory_directions,
    balanced_scenario_splits,
    fit_low_rank_subspace,
    spearman_rank_correlation,
    subspace_overlap,
)


def shuffled_copy(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    by_state: dict[str, list[dict]] = {}
    for row in rows:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    out = []
    for group in by_state.values():
        utilities = np.asarray([r["utility"] for r in group], dtype=float)
        rng.shuffle(utilities)
        out.extend({**row, "utility": float(utility)} for row, utility in zip(group, utilities))
    return out


def _scalar(npz, key: str, default: str) -> str:
    if key not in npz:
        return default
    value = npz[key]
    return str(value.item() if value.ndim == 0 else value.tolist())


def _feature(memory: np.ndarray, layers: list[int], *, component: str, layer: int | None) -> np.ndarray:
    if memory.ndim != 4 or memory.shape[2] != 2:
        raise ValueError("Debug Gate B' requires memory shaped [rows, layers, K/V, width]")
    layer_positions = list(range(len(layers))) if layer is None else [layers.index(int(layer))]
    selected = memory[:, layer_positions, :, :]
    if component == "k":
        selected = selected[:, :, 0, :]
    elif component == "v":
        selected = selected[:, :, 1, :]
    elif component != "kv":
        raise ValueError(f"unknown component: {component}")
    return selected.reshape(selected.shape[0], -1).astype(np.float64)


def _rows_with_feature(records: list[dict], branch_ids: np.ndarray, feature: np.ndarray) -> list[dict]:
    lookup = {str(branch): feature[i] for i, branch in enumerate(branch_ids)}
    rows = []
    for record in records:
        if not record.get("matched_semantics", True):
            continue
        memory = lookup.get(str(record["branch_id"]))
        if memory is not None:
            rows.append({**record, "memory": memory})
    return rows


def _same_seed_diagnostics(rows: list[dict]) -> dict:
    utility_pairs = []
    for scenario_id in sorted({int(r["scenario_id"]) for r in rows}):
        by_seed = {}
        for row in rows:
            if int(row["scenario_id"]) == scenario_id:
                by_seed.setdefault(int(row["seed"]), {})[str(row["style"])] = float(row["utility"])
        seeds = sorted(by_seed)
        if len(seeds) < 2:
            continue
        common = sorted(set(by_seed[seeds[0]]) & set(by_seed[seeds[1]]))
        if len(common) >= 2:
            rho = spearman_rank_correlation(
                [by_seed[seeds[0]][style] for style in common],
                [by_seed[seeds[1]][style] for style in common],
            )
            if np.isfinite(rho):
                utility_pairs.append(rho)

    _, directions = advantage_memory_directions(rows)
    direction_lookup = {}
    for state_id, direction in zip(
        advantage_memory_directions(rows)[0], directions
    ):
        parts = str(state_id).split(":")
        direction_lookup[(int(parts[1]), int(parts[-1].removeprefix("seed")))] = direction
    cosines = []
    scenarios = sorted({key[0] for key in direction_lookup})
    for scenario_id in scenarios:
        seeds = sorted(seed for sid, seed in direction_lookup if sid == scenario_id)
        if len(seeds) >= 2:
            a, b = direction_lookup[(scenario_id, seeds[0])], direction_lookup[(scenario_id, seeds[1])]
            cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    def summary(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
        arr = np.asarray(values, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
        }
    return {
        "utility_spearman_seed0_vs_seed1": summary(utility_pairs),
        "advantage_direction_cosine_seed0_vs_seed1": summary(cosines),
    }


def _fit_split(rows: list[dict], split: tuple[set[int], set[int]], rank: int):
    results = []
    for scenario_ids in split:
        subset = [row for row in rows if int(row["scenario_id"]) in scenario_ids]
        _, directions = advantage_memory_directions(subset)
        if len(directions) < 2:
            return None
        results.append(fit_low_rank_subspace(directions, rank))
    return results[0], results[1]


def _summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def analyze_config(rows: list[dict], scenario_ids: list[int], ranks: list[int], *, rng: np.random.Generator,
                   shuffle_repeats: int, run_null: bool) -> dict:
    splits = balanced_scenario_splits(scenario_ids)
    states_all, directions_all = advantage_memory_directions(rows)
    all_results = {}
    for requested_rank in ranks:
        result_all = fit_low_rank_subspace(directions_all, requested_rank)
        observed = []
        effective_a, effective_b = [], []
        split_pairs = []
        for split in splits:
            fitted = _fit_split(rows, split, requested_rank)
            if fitted is None:
                continue
            a, b = fitted
            observed.append(subspace_overlap(a.basis, b.basis))
            effective_a.append(int(a.basis.shape[1]))
            effective_b.append(int(b.basis.shape[1]))
            split_pairs.append(split)

        null_values = []
        null_p95_by_split = []
        if run_null:
            per_split = [[] for _ in split_pairs]
            for _ in range(int(shuffle_repeats)):
                shuffled = shuffled_copy(rows, rng)
                for index, split in enumerate(split_pairs):
                    fitted = _fit_split(shuffled, split, requested_rank)
                    if fitted is not None:
                        value = subspace_overlap(fitted[0].basis, fitted[1].basis)
                        null_values.append(value)
                        per_split[index].append(value)
            null_p95_by_split = [float(np.quantile(v, 0.95)) for v in per_split if v]

        observed_arr = np.asarray(observed, dtype=float)
        passed_split_count = None
        if null_p95_by_split:
            passed_split_count = int(sum(value > threshold for value, threshold in zip(observed_arr, null_p95_by_split)))
        item = {
            "requested_rank": int(requested_rank),
            "effective_rank_all": int(result_all.basis.shape[1]),
            "effective_rank_half_a": sorted(set(effective_a)),
            "effective_rank_half_b": sorted(set(effective_b)),
            "n_states_all": int(len(states_all)),
            "n_splits": len(splits),
            "n_valid_splits": len(observed),
            "explained_variance_all": float(result_all.explained_variance),
            "observed_overlap": _summary(observed),
            "shuffle_repeats": int(shuffle_repeats) if run_null else 0,
            "shuffled_overlap": _summary(null_values) if null_values else None,
            "shuffled_overlap_p95_by_split": _summary(null_p95_by_split) if null_p95_by_split else None,
            "splits_observed_above_shuffled_p95": passed_split_count,
            "fraction_splits_observed_above_shuffled_p95": (
                float(passed_split_count / len(observed)) if passed_split_count is not None and observed else None
            ),
        }
        all_results[str(requested_rank)] = item
    return all_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug Gate B' with action-side K/V and exhaustive split diagnostics")
    ap.add_argument("--records", default="results/gateA_glm_sweep.jsonl")
    ap.add_argument("--kv", default="results/gateA_action_audio_kv.npz")
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--shuffle-repeats", type=int, default=200)
    ap.add_argument("--output", default="results/debug_gate_b_prime_summary.json")
    ap.add_argument("--layerwise-shuffle-repeats", type=int, default=0,
                    help="set >0 to run shuffled nulls for every layer/component config")
    args = ap.parse_args()

    records = read_jsonl(args.records)
    kv = np.load(args.kv)
    memory = kv["memory"]
    layers = [int(x) for x in kv["layers"].tolist()]
    branch_ids = kv["branch_ids"]
    scenario_ids = sorted({int(row["scenario_id"]) for row in records if row.get("matched_semantics", True)})
    rng = np.random.default_rng(4242424242)
    configs = [("all_layers_kv", "kv", None, args.shuffle_repeats > 0)]
    configs.extend((f"layer{layer}_{component}", component, layer, args.layerwise_shuffle_repeats > 0)
                   for layer, component in itertools.product(layers, ("k", "v", "kv")))

    summary = {
        "records": str(args.records),
        "kv": str(args.kv),
        "observation": _scalar(kv, "observation", "unknown"),
        "pooling": _scalar(kv, "pooling", "unknown"),
        "layers": layers,
        "n_records": len(records),
        "n_scenarios": len(scenario_ids),
        "n_unique_balanced_splits": len(balanced_scenario_splits(scenario_ids)),
        "same_seed_diagnostics": {},
        "configs": {},
    }
    for name, component, layer, run_null in configs:
        features = _feature(memory, layers, component=component, layer=layer)
        rows = _rows_with_feature(records, branch_ids, features)
        if not summary["same_seed_diagnostics"]:
            summary["same_seed_diagnostics"] = _same_seed_diagnostics(rows)
        summary["configs"][name] = {
            "component": component,
            "layer": layer,
            "analysis": analyze_config(
                rows,
                scenario_ids,
                args.ranks,
                rng=rng,
                shuffle_repeats=args.shuffle_repeats if run_null and layer is None else args.layerwise_shuffle_repeats,
                run_null=run_null,
            ),
        }
        print(f"analyzed {name}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(out), "configs": len(configs), "splits": summary["n_unique_balanced_splits"]}, indent=2))


if __name__ == "__main__":
    main()
