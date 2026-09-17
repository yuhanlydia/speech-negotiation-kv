from pathlib import Path
import pytest
import tomli
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


def test_parageo_extra_declares_official_waveform_runtime_dependencies():
    project = Path(__file__).parents[1] / "pyproject.toml"
    dependencies = tomli.loads(project.read_text())["project"]["optional-dependencies"]["parageo"]
    declared = {dependency.split("=")[0].split("<")[0].split(">")[0].lower() for dependency in dependencies}
    assert {
        "omegaconf", "scipy", "einops", "diffusers", "conformer",
        "hydra-core", "lightning", "rich", "gdown", "matplotlib", "wget",
        "setuptools", "soundfile", "google-genai",
    } <= declared
    assert "torch==2.3.0" in dependencies
    assert "torchaudio==2.3.0" in dependencies
