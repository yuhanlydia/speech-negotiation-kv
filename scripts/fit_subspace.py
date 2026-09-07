#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from speech_negotiation_kv.records import read_jsonl
from speech_negotiation_kv.subspace import advantage_memory_directions, fit_low_rank_subspace, subspace_overlap


def joined_rows(records_path: str, kv_path: str) -> list[dict]:
    records = read_jsonl(records_path)
    kv = np.load(kv_path)
    lookup = {str(b): m.astype(np.float64) for b, m in zip(kv["branch_ids"], kv["memory"])}
    rows = []
    for rec in records:
        if not rec.get("matched_semantics", True):
            continue
        mem = lookup.get(str(rec["branch_id"]))
        if mem is None:
            continue
        rows.append({**rec, "memory": mem})
    return rows


def shuffled_copy(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    by_state: dict[str, list[dict]] = {}
    for row in rows:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    out = []
    for group in by_state.values():
        utilities = np.asarray([r["utility"] for r in group], dtype=float)
        rng.shuffle(utilities)
        for r, u in zip(group, utilities):
            out.append({**r, "utility": float(u)})
    return out


def fit_for_scenarios(rows: list[dict], scenario_ids: set[int], rank: int):
    subset = [r for r in rows if int(r["scenario_id"]) in scenario_ids]
    states, G = advantage_memory_directions(subset)
    if len(states) < 2:
        raise RuntimeError("not enough informative matched states for subspace fitting")
    return states, G, fit_low_rank_subspace(G, rank)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit and audit advantage-KV subspace")
    ap.add_argument("--records", default="results/pilot_sweep.jsonl")
    ap.add_argument("--kv", default="results/pilot_kv.npz")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--shuffle-repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/subspace_r8")
    args = ap.parse_args()

    rows = joined_rows(args.records, args.kv)
    scenario_ids = sorted({int(r["scenario_id"]) for r in rows})
    if len(scenario_ids) < 4:
        raise RuntimeError("need at least four scenarios for half-split stability")
    half_a = set(scenario_ids[::2])
    half_b = set(scenario_ids[1::2])
    _, G_all = advantage_memory_directions(rows)
    result = fit_low_rank_subspace(G_all, args.rank)
    _, _, result_a = fit_for_scenarios(rows, half_a, args.rank)
    _, _, result_b = fit_for_scenarios(rows, half_b, args.rank)
    observed_overlap = subspace_overlap(result_a.basis, result_b.basis)

    rng = np.random.default_rng(args.seed)
    null_overlaps = []
    for _ in range(args.shuffle_repeats):
        shuffled = shuffled_copy(rows, rng)
        try:
            _, _, sa = fit_for_scenarios(shuffled, half_a, args.rank)
            _, _, sb = fit_for_scenarios(shuffled, half_b, args.rank)
            null_overlaps.append(subspace_overlap(sa.basis, sb.basis))
        except RuntimeError:
            continue
    null95 = float(np.quantile(null_overlaps, 0.95)) if null_overlaps else float("nan")

    mean_direction = G_all.mean(axis=0)
    if np.linalg.norm(mean_direction) > 0:
        mean_direction = mean_direction / np.linalg.norm(mean_direction)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "basis.npz",
        basis=result.basis.astype(np.float32),
        singular_values=result.singular_values.astype(np.float32),
        mean_direction=mean_direction.astype(np.float32),
    )
    summary = {
        "rank": args.rank,
        "n_rows": len(rows),
        "n_states": int(G_all.shape[0]),
        "n_scenarios": len(scenario_ids),
        "explained_variance": result.explained_variance,
        "scenario_half_overlap": observed_overlap,
        "shuffled_overlap_p95": null95,
        "passes_overlap_null": bool(observed_overlap > null95) if np.isfinite(null95) else False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
