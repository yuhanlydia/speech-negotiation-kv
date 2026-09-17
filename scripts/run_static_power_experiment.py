#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


STAGES = ("dataset", "audio", "generate", "judge", "summarize")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def command_matrix(config: dict, *, stage: str, python: str | None = None,
                   benchmark_root: str = "/root/SpeechParaling-Bench") -> list[list[str]]:
    py = python or sys.executable
    exp, data = config["experiment"], config["dataset"]
    root = Path(config["outputs"]["root"])
    base_config = str(config["parageo"]["base_config"])
    manifest = str(data["manifest"])
    if stage == "dataset":
        return [[py, "scripts/build_static_power_dataset.py", "--output", manifest,
                 "--audio-dir", str(data["input_audio_dir"])]]
    if stage == "audio":
        return [[py, "scripts/synthesize_static_power_prompts.py", "--dataset", manifest,
                 "--audit-output", str(root / "input_audio_audit.jsonl"),
                 "--voice", str(data["tts_voice"]), "--sample-rate", str(data["sample_rate"])]]
    if stage == "generate":
        common = [py, "scripts/run_icassp_generation.py", "--config", base_config,
                  "--task", "static", "--split", "heldout", "--dataset-manifest", manifest]
        commands = [common + ["--variant", "prompt_only", "--output-dir", str(root / "generation" / "prompt_only")],
                    common + ["--variant", "main", "--alpha", str(exp["alpha"]),
                              "--output-dir", str(root / "generation" / "main")]]
        for seed in exp["random_seeds"]:
            tag = f"random_seed_{int(seed)}"
            commands.append(common + ["--variant", "random", "--alpha", str(exp["alpha"]),
                                       "--random-seed", str(int(seed)),
                                       "--output-dir", str(root / "generation" / tag)])
        return commands
    if stage == "judge":
        commands = []
        for seed in exp["random_seeds"]:
            tag = f"random_seed_{int(seed)}"
            commands.append([
                py, "scripts/run_official_speechparaling_judge.py",
                "--benchmark-root", benchmark_root, "--task", "static",
                "--prompt-jsonl", manifest,
                "--candidate-dir", str(root / "generation" / "main"),
                "--baseline-dir", str(root / "generation" / tag),
                "--candidate-name", "main", "--baseline-name", tag,
                "--output-dir", str(root / "judge" / f"main_vs_{tag}"),
            ])
        return commands
    if stage == "summarize":
        return [[py, "scripts/summarize_static_power.py", "--config", "configs/icassp2027_static_power.yaml",
                 "--output", str(root / "summary.json")]]
    raise ValueError(f"unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the balanced static-power experiment")
    parser.add_argument("--config", default="configs/icassp2027_static_power.yaml")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--benchmark-root", default="/root/SpeechParaling-Bench")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        for command in command_matrix(config, stage=stage, benchmark_root=args.benchmark_root):
            print(shell_join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
