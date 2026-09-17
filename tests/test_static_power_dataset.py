import importlib.util
from collections import Counter
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "build_static_power_dataset.py"
    spec = importlib.util.spec_from_file_location("build_static_power_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_is_balanced_unique_and_deterministic():
    module = load_module()
    first = module.build_rows()
    second = module.build_rows()

    assert first == second
    assert len(first) == 180
    assert Counter(row["attribute_key"] for row in first) == {
        key: 10 for key in module.ATTRIBUTE_CONTROLS
    }
    normalized = [module.normalize_target(row["target_text"]) for row in first]
    assert len(normalized) == len(set(normalized))
    assert {row["dimension"] for row in first} == {"Attitude", "Cognitive State"}


def test_rows_have_parseable_prompts_and_stable_ids():
    module = load_module()
    rows = module.build_rows()

    assert rows[0]["item_id"] == "static_power:attitude:polite_tone:00"
    assert Path(rows[0]["audio_path"]).name == "static_power_001.wav"
    assert Path(rows[-1]["audio_path"]).name == "static_power_180.wav"
    for row in rows:
        assert row["target_text"] in row["prompt"]
        assert row["control"] in row["prompt"].lower()
