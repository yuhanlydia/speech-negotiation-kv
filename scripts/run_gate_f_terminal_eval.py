#!/usr/bin/env python
from __future__ import annotations

import argparse
from hashlib import sha1
import json
import os
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.crad import load_crad
from speech_negotiation_kv.glm_voice import GLMVoiceBackend
from speech_negotiation_kv.long_horizon import MockLongHorizonBackend, run_long_horizon_branch
from speech_negotiation_kv.long_horizon import probe_opening_styles
from speech_negotiation_kv.short_horizon_selector import (
    opening_state_features, score_geometry_selector, score_onehot_selector,
)


def load_artifact(path: str) -> dict:
    data = np.load(path)
    styles = [str(value) for value in data["styles"].tolist()]
    return {
        "styles": styles,
        "coordinates": np.asarray(data["style_coordinates"], dtype=np.float64),
        "geometry": {
            "weights": data["geometry_weights"], "state_mean": data["geometry_state_mean"],
            "state_scale": data["geometry_state_scale"],
        },
        "onehot": {
            "weights": data["onehot_weights"], "state_mean": data["onehot_state_mean"],
            "state_scale": data["onehot_state_scale"], "styles": styles,
        },
        "best_fixed_style": str(data["best_fixed_style"].item()),
    }


def existing_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out.add((int(row["scenario_id"]), int(row["seed"])))
    return out


def deterministic_random_style(styles: list[str], *, scenario_id: int, seed: int, base_seed: int) -> str:
    digest = sha1(f"{base_seed}|{scenario_id}|{seed}|gate-f-random".encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(styles)
    return styles[index]


def choose_style(method: str, artifact: dict, scenario: dict, *, scenario_id: int, seed: int, base_seed: int):
    styles = artifact["styles"]
    if method == "neutral":
        return "neutral", None
    if method == "best_fixed":
        return artifact["best_fixed_style"], None
    if method == "random":
        return deterministic_random_style(styles, scenario_id=scenario_id, seed=seed, base_seed=base_seed), None
    phi = opening_state_features(float(scenario["Creditor Target Days"]), float(scenario["Debtor Target Days"]))
    phi_rows = np.repeat(phi[None, :], len(styles), axis=0)
    if method == "geometry":
        scores = score_geometry_selector(artifact["geometry"], phi_rows, artifact["coordinates"])
    elif method == "onehot":
        scores = score_onehot_selector(artifact["onehot"], phi_rows, styles)
    else:
        raise ValueError(f"unknown method {method!r}")
    index = int(np.argmax(scores))
    return styles[index], [float(value) for value in scores]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Gate-F selector on held-out CRAD terminal evaluation")
    parser.add_argument("--config", default="configs/crad_gate_f_16gb.yaml")
    parser.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    parser.add_argument("--selector", default="results/gate_f_selector.npz")
    parser.add_argument(
        "--method",
        choices=["geometry", "onehot", "neutral", "random", "best_fixed", "immediate_search"],
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario-start", type=int, default=None)
    parser.add_argument("--scenario-end", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["experiment"]
    artifact = load_artifact(args.selector)
    frame = load_crad(args.data)
    start = int(args.scenario_start if args.scenario_start is not None else exp["final_start"])
    end = int(args.scenario_end if args.scenario_end is not None else exp["final_end"])
    if not (80 <= start < end <= 100):
        parser.error("final Gate-F evaluation must stay inside CRAD test scenarios 80--99")
    part = frame.iloc[start:end]
    seeds = [int(seed) for seed in (args.seeds if args.seeds else exp["final_seeds"])]
    horizon = int(exp.get("terminal_horizon", 8))
    max_horizon = int(exp.get("terminal_max_horizon", 12))
    base_seed = int(exp.get("evaluation_base_seed", 6242424242))

    if args.dry_run:
        backend = MockLongHorizonBackend(artifact["styles"])
    else:
        model = cfg["model"]
        backend = GLMVoiceBackend(
            model_name=model["name"], quantization=model.get("quantization", "int4"),
            device=model.get("device", "cuda:0"), max_new_tokens=model.get("max_new_tokens", 96),
            temperature=model.get("temperature", 0.2), top_p=model.get("top_p", 0.8),
            audio_vocab_size=model.get("audio_vocab_size", 16384),
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    completed = existing_keys(output)
    expected = len(part) * len(seeds)
    written = 0
    with output.open("a", encoding="utf-8") as handle:
        for scenario_id, series in part.iterrows():
            scenario = series.to_dict()
            for seed in seeds:
                key = (int(scenario_id), int(seed))
                if key in completed:
                    continue
                probes = None
                fallback_used = None
                expected_probe = None
                if args.method == "immediate_search":
                    style, probes, fallback_used = probe_opening_styles(
                        scenario,
                        scenario_id=int(scenario_id),
                        styles=artifact["styles"],
                        branch_seed=int(seed),
                        backend=backend,
                        base_seed=base_seed,
                    )
                    scores = [probe["immediate_utility"] for probe in probes]
                    expected_probe = next(
                        probe for probe in probes if str(probe["style"]) == style
                    )
                else:
                    style, scores = choose_style(
                        args.method, artifact, scenario,
                        scenario_id=int(scenario_id), seed=int(seed), base_seed=base_seed,
                    )
                result = run_long_horizon_branch(
                    scenario, scenario_id=int(scenario_id), style=style, branch_seed=int(seed),
                    backend=backend, horizon=horizon, max_horizon=max_horizon, base_seed=base_seed,
                    expected_opening_probe=expected_probe,
                )
                result["method"] = args.method
                result["selected_style"] = style
                result["selection_scores"] = scores
                if probes is not None:
                    result["opening_probes"] = probes
                    result["opening_probe_fallback_used"] = bool(fallback_used)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed.add(key)
                written += 1
                if written % 10 == 0:
                    print(f"{args.method}: wrote {written} new branches; {len(completed)}/{expected}")

    print(json.dumps({
        "method": args.method, "output": str(output), "new_branches": written,
        "available_branches": len(completed), "expected_branches": expected,
        "complete": bool(len(completed) >= expected), "dry_run": bool(args.dry_run),
    }, indent=2))


if __name__ == "__main__":
    main()
