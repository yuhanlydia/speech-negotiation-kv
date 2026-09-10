from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys

from .glm_voice import GLMVoiceBackend, VoiceGeneration, audio_ids_to_prompt


@dataclass(frozen=True)
class OfficialVoiceAssets:
    repo_root: str
    speech_tokenizer_path: str
    decoder_path: str

    def validate(self) -> None:
        root = Path(self.repo_root)
        decoder = Path(self.decoder_path)
        required = [
            root / "flow_inference.py",
            root / "speech_tokenizer",
            decoder / "config.yaml",
            decoder / "flow.pt",
            decoder / "hift.pt",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing official GLM-4-Voice assets: " + ", ".join(missing))


def official_audio_instruction_prompt(audio_ids: list[int]) -> str:
    system = (
        "User will provide you with a speech instruction. Do it step by step. "
        "First, think about the instruction and respond in a interleaved manner, "
        "with 13 text token followed by 26 audio tokens."
    )
    return (
        f"<|system|>\n{system}"
        f"<|user|>\n{audio_ids_to_prompt(audio_ids)}"
        "<|assistant|>streaming_transcription\n"
    )


@dataclass(frozen=True)
class WaveformGeneration:
    generation: VoiceGeneration
    wav_path: str
    sample_rate: int = 22050


class GLMOfficialWaveformBackend(GLMVoiceBackend):
    """GLM-4-Voice backend using the official Whisper-VQ tokenizer and Flow/HiFT decoder.

    The 9B model is still loaded through the existing local Transformers path;
    input waveform tokenization and output audio decoding follow the official
    GLM-4-Voice repository implementation.
    """

    def __init__(self, *, assets: OfficialVoiceAssets, **kwargs):
        super().__init__(**kwargs)
        self.assets = assets
        self._speech_tokenizer_model = None
        self._feature_extractor = None
        self._audio_decoder = None
        self._extract_speech_token = None

    def _ensure_official_components(self) -> None:
        if self._audio_decoder is not None:
            return
        self.assets.validate()
        root = Path(self.assets.repo_root).resolve()
        for path in (root, root / "cosyvoice", root / "third_party" / "Matcha-TTS"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        from transformers import WhisperFeatureExtractor
        from speech_tokenizer.modeling_whisper import WhisperVQEncoder
        from speech_tokenizer.utils import extract_speech_token
        from flow_inference import AudioDecoder

        self._speech_tokenizer_model = WhisperVQEncoder.from_pretrained(
            self.assets.speech_tokenizer_path
        ).eval().to(self.device)
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.assets.speech_tokenizer_path
        )
        decoder = Path(self.assets.decoder_path)
        self._audio_decoder = AudioDecoder(
            config_path=str(decoder / "config.yaml"),
            flow_ckpt_path=str(decoder / "flow.pt"),
            hift_ckpt_path=str(decoder / "hift.pt"),
            device=self.device,
        )
        self._extract_speech_token = extract_speech_token

    def tokenize_waveform(self, audio_path: str | Path) -> list[int]:
        self._ensure_official_components()
        tokens = self._extract_speech_token(
            self._speech_tokenizer_model,
            self._feature_extractor,
            [str(audio_path)],
        )[0]
        result = [int(value) for value in tokens]
        if not result:
            raise RuntimeError(f"official speech tokenizer returned no tokens for {audio_path}")
        return result

    def decode_audio_tokens(self, audio_ids: list[int], output_path: str | Path) -> str:
        if not audio_ids:
            raise ValueError("cannot decode empty audio token sequence")
        self._ensure_official_components()
        import torch
        import torchaudio

        token = torch.tensor(audio_ids, dtype=torch.int64, device=self.device).unsqueeze(0)
        waveform = self._audio_decoder.offline_inference(token).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output), waveform.cpu(), 22050, format="wav")
        return str(output)

    def generate_from_audio_instruction(self, audio_path: str | Path, output_path: str | Path, *,
                                        seed: int, steering=None) -> WaveformGeneration:
        audio_ids = self.tokenize_waveform(audio_path)
        prompt = official_audio_instruction_prompt(audio_ids)
        context = steering if steering is not None else nullcontext()
        with context:
            generation = self.generate(prompt, seed=seed)
        wav_path = self.decode_audio_tokens(generation.audio_token_ids, output_path)
        return WaveformGeneration(generation=generation, wav_path=wav_path)
