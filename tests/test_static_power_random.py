import importlib.util
import json
from pathlib import Path

import numpy as np

from speech_negotiation_kv.icassp_variants import norm_matched_random_directions


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "run_icassp_generation.py"
    spec = importlib.util.spec_from_file_location("run_icassp_generation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_external_dataset_manifest_loads_audio_and_static_fields(tmp_path):
    module = load_module()
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(json.dumps({
        "item_id": "static_power:attitude:polite_tone:00",
        "prompt": "Please read this sentence with a polite tone: 'Please sit down.'",
        "dimensions": ["Attitude"],
        "audio_path": str(audio),
    }) + "\n")

    items = module.load_external_items(manifest, task="static")

    assert len(items) == 1
    assert items[0].item_id == "static_power:attitude:polite_tone:00"
    assert items[0].audio_path == str(audio)


def test_random_seed_override_preserves_default_when_absent():
    module = load_module()

    assert module.resolve_random_seed({"random_seed": 11}, None) == 11
    assert module.resolve_random_seed({"random_seed": 11}, 22) == 22


def test_five_random_seeds_change_direction_and_match_norm():
    target = np.arange(12, dtype=np.float64).reshape(3, 4) + 1
    seeds = [15242424242, 16242424242, 17242424242, 18242424242, 19242424242]
    outputs = [norm_matched_random_directions(target, np.random.default_rng(seed)) for seed in seeds]

    for output in outputs:
        np.testing.assert_allclose(np.linalg.norm(output, axis=1), np.linalg.norm(target, axis=1))
    assert len({output.tobytes() for output in outputs}) == 5
