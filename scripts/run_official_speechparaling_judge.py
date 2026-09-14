#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys


TASK_SPECS = {
    "static": ("judge_data/judge_code/para_con/para_con_short_sin_en.py", "jsonl_prompt_en/para_con/short_sin.jsonl"),
    "composed": ("judge_data/judge_code/para_con/para_con_short_multi_en.py", "jsonl_prompt_en/para_con/short_multi.jsonl"),
    "dynamic": ("judge_data/judge_code/dyn_var/dyn_var_en.py", "jsonl_prompt_en/dyn_var/dyn_var.jsonl"),
}


def task_spec(task: str) -> tuple[str, str]:
    if task not in TASK_SPECS:
        raise ValueError(f"unknown task: {task}")
    return TASK_SPECS[task]


def wav_names(path: str | Path) -> list[str]:
    return sorted(file.name for file in Path(path).glob("*.wav"))


def validate_pair_dirs(candidate_dir: str | Path, baseline_dir: str | Path) -> list[str]:
    candidate, baseline = wav_names(candidate_dir), wav_names(baseline_dir)
    if not candidate:
        raise ValueError(f"candidate directory has no wav files: {candidate_dir}")
    if candidate != baseline:
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError("candidate/baseline wav sets differ; "
                         f"missing_candidate={missing_candidate[:5]}, missing_baseline={missing_baseline[:5]}")
    return candidate


@contextmanager
def pushd(path: str | Path):
    old = Path.cwd(); os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_upstream_module(benchmark_root: str | Path, task: str):
    root = Path(benchmark_root).resolve()
    module_rel, _ = task_spec(task)
    module_path = root / module_rel
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location(f"speechparaling_official_{task}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream judge module {module_path}")
    module = importlib.util.module_from_spec(spec)
    with pushd(root):
        spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SpeechParaling's official English pairwise judge on fixed WAV sets")
    ap.add_argument("--benchmark-root", default=os.environ.get("SPEECHPARALING_ROOT"))
    ap.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    ap.add_argument("--candidate-dir", required=True); ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--candidate-name", required=True); ap.add_argument("--baseline-name", default="prompt_only")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    if not args.benchmark_root:
        ap.error("--benchmark-root or SPEECHPARALING_ROOT is required")
    validate_pair_dirs(args.candidate_dir, args.baseline_dir)
    root = Path(args.benchmark_root).resolve()
    _, prompt_rel = task_spec(args.task)
    output = Path(args.output_dir).resolve(); judge_json, metadata = output / "judge_json", output / "metadata"
    judge_json.mkdir(parents=True, exist_ok=True); metadata.mkdir(parents=True, exist_ok=True)
    module = load_upstream_module(root, args.task)
    module.PROMPT_JSONL = str(root / prompt_rel)
    module.MODEL_DIRS = {args.baseline_name: str(Path(args.baseline_dir).resolve()),
                         args.candidate_name: str(Path(args.candidate_dir).resolve())}
    module.OUTPUT_DIRS = {args.baseline_name: str(judge_json / args.baseline_name),
                          args.candidate_name: str(judge_json / args.candidate_name)}
    module.METADATA_DIR = str(metadata)
    with pushd(root):
        module.evaluate(args.candidate_name, baseline_name=args.baseline_name)
    print(f"official metadata: {metadata}")


if __name__ == "__main__":
    main()
