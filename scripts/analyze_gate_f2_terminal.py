#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.records import read_jsonl
from speech_negotiation_kv.short_horizon_selector import paired_bootstrap_mean_delta


def parse_record_args(items: list[str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--records entries must use METHOD=PATH")
        method, path = item.split("=", 1)
        paths[method] = path
    return paths


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Gate-F2 immediate-search replication")
    parser.add_argument("--config", default="configs/crad_gate_f2_16gb.yaml")
    parser.add_argument("--records", action="append", required=True, help="METHOD=PATH")
    parser.add_argument("--output", default="results/gate_f2_terminal_summary.json")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["experiment"]
    expected_keys = {
        (scenario_id, int(seed))
        for scenario_id in range(int(exp["final_start"]), int(exp["final_end"]))
        for seed in exp["final_seeds"]
    }
    paths = parse_record_args(args.records)
    required = {"immediate_search", "best_fixed"}
    if required - set(paths):
        parser.error(f"missing required methods: {sorted(required - set(paths))}")

    raw_rows: dict[str, list[dict]] = {method: read_jsonl(path) for method, path in paths.items()}
    maps: dict[str, dict[tuple[int, int], dict]] = {}
    methods: dict[str, dict] = {}
    exact_coverage: dict[str, bool] = {}
    for method, rows in raw_rows.items():
        keys = [(int(row["scenario_id"]), int(row["seed"])) for row in rows]
        exact_coverage[method] = bool(len(keys) == len(expected_keys) and set(keys) == expected_keys)
        terminal = [
            row for row in rows
            if row.get("outcome") in {"agreement", "no_deal"} and row.get("utility") is not None
        ]
        mapping = {(int(row["scenario_id"]), int(row["seed"])): row for row in terminal}
        maps[method] = mapping
        methods[method] = {
            "path": paths[method],
            "rows": len(rows),
            "terminal_states": len(mapping),
            "utility": distribution([float(row["utility"]) for row in mapping.values()]),
            "rounds": distribution([float(row["rounds"]) for row in mapping.values()]),
            "outcomes": dict(Counter(str(row["outcome"]) for row in mapping.values())),
            "forced_no_deal": int(sum(bool(row.get("terminal_forced_no_deal")) for row in mapping.values())),
            "selected_styles": dict(Counter(str(row.get("selected_style")) for row in mapping.values())),
        }

    search_rows = raw_rows["immediate_search"]
    all_probes = [probe for row in search_rows for probe in row.get("opening_probes", [])]
    selected_eligible = []
    for row in search_rows:
        selected = [
            probe for probe in row.get("opening_probes", [])
            if str(probe.get("style")) == str(row.get("selected_style"))
        ]
        selected_eligible.append(bool(len(selected) == 1 and selected[0].get("eligible")))
    probe_quality = {
        "states": len(search_rows),
        "total_probes": len(all_probes),
        "six_probes_per_state_rate": float(np.mean([
            len(row.get("opening_probes", [])) == 6 for row in search_rows
        ])) if search_rows else 0.0,
        "eligible_probe_rate": float(np.mean([bool(probe.get("eligible")) for probe in all_probes])) if all_probes else 0.0,
        "selected_probe_eligible_rate": float(np.mean(selected_eligible)) if selected_eligible else 0.0,
        "verified_replay_rate": float(np.mean([
            row.get("opening_probe_replay_verified") is True for row in search_rows
        ])) if search_rows else 0.0,
        "fallback_states": int(sum(bool(row.get("opening_probe_fallback_used")) for row in search_rows)),
    }

    common = expected_keys & set(maps["immediate_search"]) & set(maps["best_fixed"])
    primary = paired_bootstrap_mean_delta(
        [float(maps["immediate_search"][key]["utility"]) for key in sorted(common)],
        [float(maps["best_fixed"][key]["utility"]) for key in sorted(common)],
        repeats=args.bootstrap_repeats,
    ) if common else {"n": 0, "mean_delta": None, "lower_95": None, "upper_95": None,
                       "repeats": int(args.bootstrap_repeats)}
    data_gate = bool(
        all(exact_coverage.get(method, False) for method in required)
        and all(set(maps[method]) == expected_keys for method in required)
        and probe_quality["six_probes_per_state_rate"] == 1.0
        and probe_quality["selected_probe_eligible_rate"] == 1.0
        and probe_quality["verified_replay_rate"] == 1.0
    )
    gate_pass = bool(data_gate and primary["lower_95"] is not None and primary["lower_95"] > 0.0)
    summary = {
        "gate": "F2_immediate_search_replication",
        "expected_states_per_method": len(expected_keys),
        "exact_coverage": exact_coverage,
        "common_paired_states": len(common),
        "methods": methods,
        "probe_quality": probe_quality,
        "data_gate_passes": data_gate,
        "primary_comparison": "immediate_search - best_fixed",
        "primary_delta": primary,
        "pass_rule": "data gate passes and paired bootstrap 95% CI lower bound > 0",
        "gate_f2_passes": gate_pass,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "data_gate_passes": data_gate,
        "primary_delta": primary,
        "gate_f2_passes": gate_pass,
    }, indent=2))


if __name__ == "__main__":
    main()
