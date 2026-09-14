import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_official_speechparaling_judge.py"
spec = importlib.util.spec_from_file_location("judge_wrapper", SCRIPT)
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def test_task_mapping_uses_upstream_english_judges():
    assert judge.task_spec("static")[0].endswith("para_con_short_sin_en.py")
    assert judge.task_spec("composed")[0].endswith("para_con_short_multi_en.py")
    assert judge.task_spec("dynamic")[0].endswith("dyn_var_en.py")


def test_pair_validation_requires_exact_wav_set(tmp_path):
    candidate, baseline = tmp_path / "candidate", tmp_path / "baseline"
    candidate.mkdir(); baseline.mkdir()
    (candidate / "x_001.wav").write_bytes(b"x"); (baseline / "x_001.wav").write_bytes(b"x")
    assert judge.validate_pair_dirs(candidate, baseline) == ["x_001.wav"]
    (candidate / "x_002.wav").write_bytes(b"x")
    try:
        judge.validate_pair_dirs(candidate, baseline)
    except ValueError as error:
        assert "wav sets differ" in str(error)
    else:
        raise AssertionError("mismatched directories should fail")
