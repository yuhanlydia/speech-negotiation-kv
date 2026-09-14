#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from speech_negotiation_kv.icassp_eval import (
    bootstrap_preference, paired_wer_degradation, preference_by_dimension,
    read_official_metadata, render_ablation_table, render_geometry_table,
    render_main_results_table, render_result_macros,
)

COMMON_ABLATIONS = ["main", "raw_basis", "rank4", "rank8", "rank32", "layers_early", "layers_middle", "layers_late"]
TASK_ABLATIONS = {
    "static": COMMON_ABLATIONS,
    "composed": COMMON_ABLATIONS + ["comp_mean", "comp_normalized"],
    "dynamic": COMMON_ABLATIONS + ["dyn_start", "dyn_end", "dyn_midpoint"],
}


def _judge_summary(path: Path, *, repeats: int, seed: int) -> dict:
    rows = read_official_metadata(path)
    result = bootstrap_preference(rows, repeats=repeats, seed=seed)
    result["by_dimension"] = preference_by_dimension(rows)
    return result


def _manifest(root: Path, split: str, task: str, variant: str) -> Path:
    return root / split / task / variant / "manifest.jsonl"


def _judge(root: Path, split: str, task: str, variant: str) -> Path:
    return root / "judge" / split / task / variant / "metadata"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize frozen ICASSP 2027 ParaGeo experiments")
    ap.add_argument("--config", default="configs/icassp2027_parageo.yaml")
    ap.add_argument("--alpha-selection", default=None); ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp, root = cfg["experiment"], Path(cfg["outputs"]["root"])
    paper_dir = Path(cfg["outputs"]["paper_generated"]); paper_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else root / "summary.json"; output.parent.mkdir(parents=True, exist_ok=True)
    alpha_path = Path(args.alpha_selection) if args.alpha_selection else root / "alpha_selection.json"
    alpha_selection = json.loads(alpha_path.read_text(encoding="utf-8"))
    geometry = json.loads(Path(cfg["parageo"]["basis_summary"]).read_text(encoding="utf-8"))

    tasks, ablations, any_pass = {}, {}, False
    for task in ("static", "composed", "dynamic"):
        main = _judge_summary(_judge(root, "heldout", task, "main"), repeats=int(exp["bootstrap_repeats"]), seed=int(exp["bootstrap_seed"]))
        random = _judge_summary(_judge(root, "heldout", task, "random"), repeats=int(exp["bootstrap_repeats"]), seed=int(exp["bootstrap_seed"]) + 1)
        fidelity = paired_wer_degradation(_manifest(root, "heldout", task, "main"), _manifest(root, "heldout", task, "prompt_only"))
        random_fidelity = paired_wer_degradation(_manifest(root, "heldout", task, "random"), _manifest(root, "heldout", task, "prompt_only"))
        threshold = float(exp["stop_dynamic_gain"] if task == "dynamic" else exp["stop_static_or_composed_gain"])
        score_ok = bool(main.get("gain_vs_tie") is not None and float(main["gain_vs_tie"]) >= threshold)
        fidelity_ok = bool(fidelity.get("mean_degradation") is not None and float(fidelity["mean_degradation"]) <= float(exp["max_absolute_text_wer_degradation"]))
        random_ok = bool(random.get("gain_vs_tie") is not None and float(random["gain_vs_tie"]) < threshold
                         and main.get("preference_score") is not None and random.get("preference_score") is not None
                         and float(main["preference_score"]) > float(random["preference_score"]))
        passes = bool(score_ok and fidelity_ok and random_ok); any_pass = any_pass or passes
        tasks[task] = {"selected_alpha": alpha_selection["selected"][task], "main": main, "random": random,
                       "fidelity": fidelity, "random_fidelity": random_fidelity, "threshold_gain": threshold,
                       "score_gate": score_ok, "fidelity_gate": fidelity_ok, "random_gate": random_ok,
                       "task_passes": passes}
        task_ablations = {}
        baseline_manifest = _manifest(root, "ablation", task, "prompt_only")
        for index, variant in enumerate(TASK_ABLATIONS[task]):
            metadata, manifest = _judge(root, "ablation", task, variant), _manifest(root, "ablation", task, variant)
            if not metadata.exists() or not manifest.exists():
                continue
            pref = _judge_summary(metadata, repeats=int(exp["bootstrap_repeats"]), seed=int(exp["bootstrap_seed"]) + 100 + index)
            pref["fidelity"] = paired_wer_degradation(manifest, baseline_manifest)
            task_ablations[variant] = pref
        ablations[task] = task_ablations

    summary = {"protocol": "ICASSP2027 ParaGeo frozen held-out evaluation", "geometry": geometry,
               "alpha_selection": alpha_selection, "tasks": tasks, "ablations": ablations,
               "paper_gate_passes": bool(any_pass),
               "decision": "continue_to_icassp2027_paper" if any_pass else "archive_parageo"}
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (paper_dir / "main_results.tex").write_text(render_main_results_table(summary), encoding="utf-8")
    (paper_dir / "geometry_table.tex").write_text(render_geometry_table(summary), encoding="utf-8")
    (paper_dir / "ablation_table.tex").write_text(render_ablation_table(summary), encoding="utf-8")
    (paper_dir / "result_macros.tex").write_text(render_result_macros(summary), encoding="utf-8")
    print(json.dumps({"output": str(output), "paper_gate_passes": summary["paper_gate_passes"],
                      "decision": summary["decision"], "paper_generated": str(paper_dir)}, indent=2))


if __name__ == "__main__":
    main()
