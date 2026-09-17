#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from speech_negotiation_kv.icassp_eval import (
    bootstrap_preference,
    choose_alpha,
    paired_wer_degradation,
    read_manifest,
    read_official_metadata,
)


def alpha_tag(alpha: float) -> str:
    return f"alpha_{float(alpha):g}".replace(".", "p")


def candidate_exclusion_reason(candidate_manifest: Path, baseline_manifest: Path) -> str | None:
    candidate_ids = {str(row["item_id"]) for row in read_manifest(candidate_manifest)}
    baseline_ids = {str(row["item_id"]) for row in read_manifest(baseline_manifest)}
    if candidate_ids == baseline_ids:
        return None
    missing = sorted(baseline_ids - candidate_ids)
    reason = (
        f"incomplete_manifest: candidate has {len(candidate_ids)}/{len(baseline_ids)} "
        f"baseline items"
    )
    if missing:
        reason += f"; missing={','.join(missing)}"
    extra = sorted(candidate_ids - baseline_ids)
    if extra:
        reason += f"; extra={','.join(extra)}"
    return reason


def select_with_status(candidates: list[dict], *, max_wer_degradation: float) -> tuple[dict | None, str]:
    try:
        return choose_alpha(candidates, max_wer_degradation=max_wer_degradation), "selected"
    except ValueError as error:
        if str(error) != "no alpha satisfies the frozen WER gate":
            raise
        return None, "no_alpha_satisfies_frozen_wer_gate"


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze ParaGeo alpha from development judge + WER only")
    ap.add_argument("--config", default="configs/icassp2027_parageo.yaml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp, root = cfg["experiment"], Path(cfg["outputs"]["root"])
    output = Path(args.output) if args.output else root / "alpha_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    selected, selection_status, candidates_by_task, exclusions_by_task = {}, {}, {}, {}
    for task in ("static", "composed", "dynamic"):
        baseline_manifest = root / "dev" / task / "prompt_only" / "manifest.jsonl"
        candidates, exclusions = [], []
        for alpha in exp["alpha_grid"]:
            tag = alpha_tag(float(alpha))
            candidate_manifest = root / "dev" / task / tag / "manifest.jsonl"
            exclusion_reason = candidate_exclusion_reason(candidate_manifest, baseline_manifest)
            if exclusion_reason is not None:
                exclusions.append({"alpha": float(alpha), "tag": tag, "reason": exclusion_reason})
                continue
            rows = read_official_metadata(root / "judge" / "dev" / task / tag / "metadata")
            pref = bootstrap_preference(rows, repeats=int(exp["bootstrap_repeats"]), seed=int(exp["bootstrap_seed"]))
            fidelity = paired_wer_degradation(candidate_manifest, baseline_manifest)
            candidates.append({"alpha": float(alpha), "tag": tag,
                               "preference_score": pref["preference_score"], "gain_vs_tie": pref["gain_vs_tie"],
                               "preference_ci": [pref["lower_95"], pref["upper_95"]],
                               "wer_degradation": fidelity["mean_degradation"],
                               "judge_samples": pref["n_samples"], "fidelity_samples": fidelity["n"]})
        best, status = select_with_status(
            candidates,
            max_wer_degradation=float(exp["max_absolute_text_wer_degradation"]),
        )
        selected[task], selection_status[task] = best, status
        candidates_by_task[task], exclusions_by_task[task] = candidates, exclusions
    payload = {"protocol": "development-only alpha selection; highest pairwise preference subject to frozen WER gate; exact ties choose smallest alpha",
               "selected": selected, "selection_status": selection_status,
               "candidates": candidates_by_task,
               "excluded_candidates": exclusions_by_task}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
