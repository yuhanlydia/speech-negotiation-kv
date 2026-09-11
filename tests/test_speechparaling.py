import json
from pathlib import Path
import subprocess
import sys
from speech_negotiation_kv.speechparaling import (
    extract_target_text,
    parse_static_control,
    parse_dynamic_control,
)


def test_static_control_parser_matches_benchmark_prompt_shape():
    prompt = "Please read this sentence with a very high pitch: 'Ah! There is a mouse!'"
    assert parse_static_control(prompt) == "very high pitch"
    assert extract_target_text(prompt) == "Ah! There is a mouse!"


def test_dynamic_parser_supports_gradual_and_step_transitions():
    gradual = "Please read this sentence starting with a very low pitch and gradually transitioning to a medium pitch: 'Late at night.'"
    assert parse_dynamic_control(gradual) == ("very low pitch", "medium pitch", "linear")
    sudden = "Please read this sentence starting with a high pitch and suddenly dropping to a very low pitch: 'Wait!'"
    assert parse_dynamic_control(sudden) == ("high pitch", "very low pitch", "step")


def test_dynamic_parser_handles_volume_wording():
    prompt = "Please read this sentence starting with a whisper and gradually increasing the volume to a normal volume: 'Did you hear?'"
    assert parse_dynamic_control(prompt) == ("whisper", "normal volume", "linear")


def test_catalog_matching_supports_compositional_multi_control():
    from speech_negotiation_kv.speechparaling import match_catalog_controls
    catalog = {
        "Pitch::very high pitch": {"control": "very high pitch"},
        "Pace::fast pace": {"control": "fast pace"},
        "Pitch::high pitch": {"control": "high pitch"},
    }
    prompt = "Please read this sentence with a very high pitch and a fast pace: 'Look!'"
    result = match_catalog_controls(prompt, catalog)
    assert "Pitch::very high pitch" in result
    assert "Pace::fast pace" in result
    assert "Pitch::high pitch" not in result


def test_word_error_rate_is_zero_for_identical_content():
    from speech_negotiation_kv.speechparaling import word_error_rate
    assert word_error_rate("Hello, world!", "hello world") == 0.0
    assert word_error_rate("one two", "one three") == 0.5


def test_pair_with_audio_uses_filename_sample_indices(tmp_path):
    from speech_negotiation_kv.speechparaling import SpeechParalingItem, pair_with_audio
    items = [SpeechParalingItem(f"i{i}", f"p{i}", ("Pitch",), "static") for i in range(1, 4)]
    for name in ("para_con_short_sin_en_001.wav", "para_con_short_sin_en_003.wav", "para_con_short_sin_en_002.wav"):
        (tmp_path / name).write_bytes(b"")
    paired = pair_with_audio(items, tmp_path)
    by_name = {Path(item.audio_path).name: item.prompt for item in paired}
    assert by_name["para_con_short_sin_en_003.wav"] == "p3"


def test_catalog_covered_selection_rejects_partial_compositions():
    from speech_negotiation_kv.speechparaling import SpeechParalingItem, select_catalog_covered_items
    catalog = {
        "Pitch::very high pitch": {"dimension": "Pitch", "control": "very high pitch"},
        "Pace::fast pace": {"dimension": "Pace", "control": "fast pace"},
    }
    good = SpeechParalingItem("g", "Please read this sentence with a very high pitch and a fast pace: 'Look!'", ("Pitch", "Pace"), "composed")
    bad = SpeechParalingItem("b", "Please read this sentence with a very high pitch and a happy emotion: 'Look!'", ("Pitch", "Emotion"), "composed")
    assert select_catalog_covered_items([bad, good], catalog, task="composed", limit=1) == [good]


def test_deterministic_dev_split_is_disjoint_and_complete():
    from speech_negotiation_kv.speechparaling import SpeechParalingItem, split_pilot_items
    items = [SpeechParalingItem(str(i), str(i), ("Pitch",), "static") for i in range(10)]
    dev = split_pilot_items(items, split="dev", modulus=5, remainder=0)
    held = split_pilot_items(items, split="heldout", modulus=5, remainder=0)
    assert [item.item_id for item in dev] == ["0", "5"]
    assert set(dev).isdisjoint(set(held))
    assert len(dev) + len(held) == len(items)


def test_catalog_builder_defaults_to_all_official_control_dimensions(tmp_path):
    prompt_dir = tmp_path / "jsonl_prompt_en" / "para_con"
    prompt_dir.mkdir(parents=True)
    rows = [
        {
            "prompt": "Please read this sentence with a happy emotion: 'We did it!'",
            "dimensions": ["Emotion"],
        },
        {
            "prompt": "Please read this sentence with a high pitch: 'Listen!'",
            "dimensions": ["Pitch"],
        },
    ]
    (prompt_dir / "short_sin.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    output = tmp_path / "catalog.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_parageo_catalog.py",
            "--benchmark-root",
            str(tmp_path),
            "--language",
            "en",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    catalog = json.loads(output.read_text())
    assert set(catalog) == {"Emotion::happy emotion", "Pitch::high pitch"}


def test_manifest_writer_preserves_unicode_under_ascii_locale(tmp_path, monkeypatch):
    from speech_negotiation_kv.speechparaling import write_manifest

    monkeypatch.setenv("LC_ALL", "C")
    output = tmp_path / "manifest.jsonl"
    write_manifest(output, [{"prompt": "say café — softly"}])
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "prompt": "say café — softly"
    }
