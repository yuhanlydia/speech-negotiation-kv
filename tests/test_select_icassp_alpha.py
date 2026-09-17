import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "select_icassp_alpha.py"
    spec = importlib.util.spec_from_file_location("select_icassp_alpha", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, item_ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"item_id": item_id}) + "\n" for item_id in item_ids),
        encoding="utf-8",
    )


def test_incomplete_candidate_is_excluded_with_item_count_reason(tmp_path):
    module = load_module()
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_manifest(baseline, ["a", "b", "c"])
    write_manifest(candidate, ["a", "b"])

    assert module.candidate_exclusion_reason(candidate, baseline) == (
        "incomplete_manifest: candidate has 2/3 baseline items; missing=c"
    )


def test_complete_candidate_is_not_excluded(tmp_path):
    module = load_module()
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_manifest(baseline, ["a", "b"])
    write_manifest(candidate, ["b", "a"])

    assert module.candidate_exclusion_reason(candidate, baseline) is None


def test_no_candidate_passing_wer_gate_is_recorded_without_selection():
    module = load_module()
    candidates = [
        {"alpha": 0.25, "preference_score": 70.0, "wer_degradation": 0.12},
        {"alpha": 0.5, "preference_score": 75.0, "wer_degradation": 0.12},
    ]

    selected, status = module.select_with_status(candidates, max_wer_degradation=0.02)

    assert selected is None
    assert status == "no_alpha_satisfies_frozen_wer_gate"
