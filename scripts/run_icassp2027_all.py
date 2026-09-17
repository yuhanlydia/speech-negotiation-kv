#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

TASKS = ("static", "composed", "dynamic")
MAIN_VARIANTS = {
    "static": ("prompt_only", "main", "random"),
    "composed": ("prompt_only", "main", "random", "full_space"),
    "dynamic": ("prompt_only", "main", "random"),
}
COMMON_ABLATIONS = ("main", "raw_basis", "rank4", "rank8", "rank32", "layers_early", "layers_middle", "layers_late")
TASK_ABLATIONS = {
    "static": COMMON_ABLATIONS,
    "composed": COMMON_ABLATIONS + ("comp_mean", "comp_normalized"),
    "dynamic": COMMON_ABLATIONS + ("dyn_start", "dyn_end", "dyn_midpoint"),
}


def alpha_tag(alpha: float) -> str:
    return f"alpha_{float(alpha):g}".replace(".", "p")


def shell_join(command: list[str]) -> str:
    return shlex.join([str(value) for value in command])


def command_matrix(config: dict, *, config_path: str, stage: str,
                   selected_alphas: dict[str, float] | None = None) -> list[list[str]]:
    py = sys.executable
    root = Path(config["outputs"]["root"])
    parageo, exp = config["parageo"], config["experiment"]
    benchmark_root = os.path.expandvars(str(config["benchmark"]["root"]))
    selected_alphas = selected_alphas or {}
    commands: list[list[str]] = []
    if stage == "geometry":
        commands += [
            [py, "scripts/build_parageo_catalog.py", "--benchmark-root", benchmark_root, "--language", "en", "--output", str(parageo["catalog"])],
            [py, "scripts/collect_parageo_calibration.py", "--config", str(parageo["calibration_config"]), "--catalog", str(parageo["catalog"]), "--output", str(parageo["calibration_records"])],
            [py, "scripts/extract_parageo_kv.py", "--config", str(parageo["calibration_config"]), "--records", str(parageo["calibration_records"]), "--output", str(parageo["calibration_kv"])],
            [py, "scripts/fit_parageo_basis.py", "--config", config_path, "--records", str(parageo["calibration_records"]), "--kv", str(parageo["calibration_kv"]), "--output", str(parageo["basis"]), "--summary", str(parageo["basis_summary"])],
        ]
    elif stage == "dev":
        judge_commands: list[list[str]] = []
        for task in TASKS:
            baseline = root / "dev" / task / "prompt_only"
            commands.append([py, "scripts/run_icassp_generation.py", "--config", config_path, "--task", task, "--split", "dev", "--variant", "prompt_only", "--output-dir", str(baseline)])
            for alpha in exp["alpha_grid"]:
                tag = alpha_tag(float(alpha))
                candidate = root / "dev" / task / tag
                commands.append([py, "scripts/run_icassp_generation.py", "--config", config_path, "--task", task, "--split", "dev", "--variant", "main", "--alpha", str(float(alpha)), "--output-dir", str(candidate)])
                judge_commands.append([py, "scripts/run_official_speechparaling_judge.py", "--benchmark-root", benchmark_root, "--task", task, "--candidate-dir", str(candidate), "--baseline-dir", str(baseline), "--candidate-name", tag, "--baseline-name", "prompt_only", "--output-dir", str(root / "judge" / "dev" / task / tag)])
        commands.extend(judge_commands)
        commands.append([py, "scripts/select_icassp_alpha.py", "--config", config_path, "--output", str(root / "alpha_selection.json")])
    elif stage == "main":
        for task in TASKS:
            alpha = selected_alphas.get(task, f"<SELECTED_ALPHA:{task}>")
            for variant in MAIN_VARIANTS[task]:
                command = [py, "scripts/run_icassp_generation.py", "--config", config_path, "--task", task, "--split", "heldout", "--variant", variant]
                if variant != "prompt_only":
                    command += ["--alpha", str(alpha)]
                command += ["--output-dir", str(root / "heldout" / task / variant)]
                commands.append(command)
    elif stage == "ablations":
        for task in TASKS:
            alpha = selected_alphas.get(task, f"<SELECTED_ALPHA:{task}>")
            commands.append([py, "scripts/run_icassp_generation.py", "--config", config_path, "--task", task, "--split", "ablation", "--variant", "prompt_only", "--output-dir", str(root / "ablation" / task / "prompt_only")])
            for variant in TASK_ABLATIONS[task]:
                commands.append([py, "scripts/run_icassp_generation.py", "--config", config_path, "--task", task, "--split", "ablation", "--variant", variant, "--alpha", str(alpha), "--output-dir", str(root / "ablation" / task / variant)])
    elif stage == "judge":
        for task in TASKS:
            baseline = root / "heldout" / task / "prompt_only"
            for variant in MAIN_VARIANTS[task][1:]:
                commands.append([py, "scripts/run_official_speechparaling_judge.py", "--benchmark-root", benchmark_root, "--task", task, "--candidate-dir", str(root / "heldout" / task / variant), "--baseline-dir", str(baseline), "--candidate-name", variant, "--baseline-name", "prompt_only", "--output-dir", str(root / "judge" / "heldout" / task / variant)])
            if task == "composed":
                commands.append([
                    py, "scripts/run_official_speechparaling_judge.py",
                    "--benchmark-root", benchmark_root,
                    "--task", task,
                    "--candidate-dir", str(root / "heldout" / task / "main"),
                    "--baseline-dir", str(root / "heldout" / task / "full_space"),
                    "--candidate-name", "main",
                    "--baseline-name", "full_space",
                    "--output-dir", str(root / "judge" / "heldout" / task / "main_vs_full_space"),
                ])
            ablation_baseline = root / "ablation" / task / "prompt_only"
            for variant in TASK_ABLATIONS[task]:
                commands.append([py, "scripts/run_official_speechparaling_judge.py", "--benchmark-root", benchmark_root, "--task", task, "--candidate-dir", str(root / "ablation" / task / variant), "--baseline-dir", str(ablation_baseline), "--candidate-name", variant, "--baseline-name", "prompt_only", "--output-dir", str(root / "judge" / "ablation" / task / variant)])
    elif stage == "summarize":
        commands.append([py, "scripts/summarize_icassp2027.py", "--config", config_path, "--alpha-selection", str(root / "alpha_selection.json"), "--output", str(root / "summary.json")])
    else:
        raise ValueError(f"unknown stage: {stage}")
    return commands


def load_selected_alphas(config: dict) -> dict[str, float]:
    path = Path(config["outputs"]["root"]) / "alpha_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is required before main/ablation generation; run --stage dev first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {task: float(payload["selected"][task]["alpha"]) for task in TASKS}


def execute_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    for command in commands:
        print("$", shell_join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the frozen ICASSP 2027 ParaGeo experiment matrix")
    ap.add_argument("--config", default="configs/icassp2027_parageo.yaml")
    ap.add_argument("--stage", default="all", choices=["geometry", "dev", "main", "ablations", "judge", "summarize", "all"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    stages = ["geometry", "dev", "main", "ablations", "judge", "summarize"] if args.stage == "all" else [args.stage]
    selected: dict[str, float] | None = None
    for stage in stages:
        if stage in {"main", "ablations"}:
            selected = None if args.dry_run else load_selected_alphas(cfg)
        execute_commands(command_matrix(cfg, config_path=args.config, stage=stage, selected_alphas=selected), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
