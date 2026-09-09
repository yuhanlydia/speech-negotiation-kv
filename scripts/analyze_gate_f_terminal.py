#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.records import read_jsonl
from speech_negotiation_kv.short_horizon_selector import paired_bootstrap_mean_delta


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "n": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)), "p95": float(np.quantile(array, 0.95)),
    }


def parse_record_args(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--records entries must use METHOD=PATH")
        method, path = item.split("=", 1)
        out[method] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze held-out terminal Gate-F method comparison")
    parser.add_argument("--config", default="configs/crad_gate_f_16gb.yaml")
    parser.add_argument("--records", action="append", required=True, help="METHOD=PATH; repeat for each method")
    parser.add_argument("--output", default="results/gate_f_terminal_summary.json")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["experiment"]
    expected = (int(exp["final_end"]) - int(exp["final_start"])) * len(exp["final_seeds"])
    paths = parse_record_args(args.records)
    required = {"geometry", "best_fixed", "neutral", "random"}
    missing = required - set(paths)
    if missing:
        parser.error(f"missing required methods: {sorted(missing)}")

    method_maps = {}
    method_summary = {}
    for method, path in paths.items():
        rows = read_jsonl(path)
        valid = [row for row in rows if row.get("outcome") in {"agreement", "no_deal"} and row.get("utility") is not None]
        mapping = {(int(row["scenario_id"]), int(row["seed"])): row for row in valid}
        method_maps[method] = mapping
        method_summary[method] = {
            "path": path,
            "n_terminal": len(mapping),
            "utility": distribution([float(row["utility"]) for row in mapping.values()]),
            "rounds": distribution([float(row["rounds"]) for row in mapping.values()]),
            "outcomes": dict(Counter(str(row["outcome"]) for row in mapping.values())),
            "forced_no_deal": int(sum(bool(row.get("terminal_forced_no_deal")) for row in mapping.values())),
            "selected_styles": dict(Counter(str(row.get("selected_style", row.get("style"))) for row in mapping.values())),
        }

    common = set.intersection(*(set(mapping) for mapping in method_maps.values()))
    paired = {}
    geometry = method_maps["geometry"]
    for baseline in sorted(set(method_maps) - {"geometry"}):
        keys = sorted(common)
        paired[baseline] = paired_bootstrap_mean_delta(
            [float(geometry[key]["utility"]) for key in keys],
            [float(method_maps[baseline][key]["utility"]) for key in keys],
            repeats=args.bootstrap_repeats,
        )
    complete = bool(len(common) == expected and all(len(mapping) == expected for mapping in method_maps.values()))
    primary = paired["best_fixed"]
    gate_pass = bool(complete and primary["lower_95"] > 0.0)
    summary = {
        "gate": "F_terminal",
        "expected_states_per_method": expected,
        "common_paired_states": len(common),
        "complete": complete,
        "methods": method_summary,
        "geometry_paired_deltas": paired,
        "pass_rule": "geometry vs best_fixed paired bootstrap 95% CI lower bound > 0",
        "gate_f_passes": gate_pass,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str(output), "common_states": len(common),
        "geometry_minus_best_fixed": primary, "gate_f_passes": gate_pass,
    }, indent=2))


if __name__ == "__main__":
    main()
