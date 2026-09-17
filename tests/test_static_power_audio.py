import importlib.util
import json
from pathlib import Path
import wave

import pytest


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "synthesize_static_power_prompts.py"
    spec = importlib.util.spec_from_file_location("synthesize_static_power_prompts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, *, sample_rate=22050, seconds=1.0, amplitude=1000):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(int(amplitude).to_bytes(2, "little", signed=True) * int(sample_rate * seconds))


def test_validate_wav_accepts_frozen_format_and_reports_metadata(tmp_path):
    module = load_module()
    path = tmp_path / "valid.wav"
    write_wav(path)

    result = module.validate_wav(path, expected_sample_rate=22050)

    assert result["sample_rate"] == 22050
    assert result["channels"] == 1
    assert result["duration_seconds"] == pytest.approx(1.0)
    assert result["clipped"] is False


def test_validate_wav_rejects_wrong_rate_and_clipping(tmp_path):
    module = load_module()
    wrong_rate = tmp_path / "wrong.wav"
    clipped = tmp_path / "clipped.wav"
    write_wav(wrong_rate, sample_rate=16000)
    write_wav(clipped, amplitude=32767)

    with pytest.raises(ValueError, match="sample rate"):
        module.validate_wav(wrong_rate, expected_sample_rate=22050)
    with pytest.raises(ValueError, match="clipped"):
        module.validate_wav(clipped, expected_sample_rate=22050)


def test_pending_rows_skips_only_valid_existing_audio(tmp_path):
    module = load_module()
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    missing = tmp_path / "missing.wav"
    write_wav(good)
    bad.write_bytes(b"not a wav")
    rows = [{"audio_path": str(path)} for path in (good, bad, missing)]

    pending = module.pending_rows(rows, expected_sample_rate=22050)

    assert pending == rows[1:]


def test_retry_delays_are_bounded_exponential_backoff():
    module = load_module()

    assert module.retry_delays(7) == [2, 4, 8, 16, 30, 30, 30]
