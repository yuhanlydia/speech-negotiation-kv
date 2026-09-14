#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from speech_negotiation_kv.icassp_eval import bootstrap_preference, choose_alpha, paired_wer_degradation, read_official_metadata


def alpha_tag(alpha: float) -> str:
    return f"alpha_{float(alpha):g}".replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze ParaGeo alpha from development judge + WER only")
    ap.add_argument("--config", default="configs/icassp2027_parageo.yaml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp, root = cfg["experiment"], Path(cfg["outputs"]["root"])
    output = Path(args.output) if args.output else root / "alpha_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    selected, candidates_by_task = {}, {}
    for task in ("static", "composed", "dynamic"):
        baseline_manifest = root / "dev" / task / "prompt_only" / "manifest.jsonl"
        candidates = []
        for alpha in exp["alpha_grid"]:
            tag = alpha_tag(float(alpha))
            candidate_manifest = root / "dev" / task / tag / "manifest.jsonl"
            rows = read_official_metadata(root / "judge" / "dev" / task / tag / "metadata")
            pref = bootstrap_preference(rows, repeats=int(exp["bootstrap_repeats"]), seed=int(exp["bootstrap_seed"]))
            fidelity = paired_wer_degradation(candidate_manifest, baseline_manifest)
            candidates.append({"alpha": float(alpha), "tag": tag,
                               "preference_score": pref["preference_score"], "gain_vs_tie": pref["gain_vs_tie"],
                               "preference_ci": [pref["lower_95"], pref["upper_95"]],
                               "wer_degradation": fidelity["mean_degradation"],
                               "judge_samples": pref["n_samples"], "fidelity_samples": fidelity["n"]})
        best = choose_alpha(candidates, max_wer_degradation=float(exp["max_absolute_text_wer_degradation"]))
        selected[task], candidates_by_task[task] = best, candidates
    payload = {"protocol": "development-only alpha selection; highest pairwise preference subject to frozen WER gate; exact ties choose smallest alpha",
               "selected": selected, "candidates": candidates_by_task}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
