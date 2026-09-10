from pathlib import Path
import pytest
from speech_negotiation_kv.glm_official_waveform import OfficialVoiceAssets, official_audio_instruction_prompt


def test_official_prompt_matches_audio_instruction_protocol():
    prompt = official_audio_instruction_prompt([3, 9])
    assert "speech instruction" in prompt
    assert "13 text token followed by 26 audio tokens" in prompt
    assert "<|audio_3|><|audio_9|>" in prompt


def test_asset_validation_requires_official_decoder_files(tmp_path):
    assets = OfficialVoiceAssets(str(tmp_path), "tokenizer", str(tmp_path / "decoder"))
    with pytest.raises(FileNotFoundError):
        assets.validate()
