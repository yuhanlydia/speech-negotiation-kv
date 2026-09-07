from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

_SPECIAL_RE = re.compile(r"<\|[^>]+\|>")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def audio_ids_to_prompt(audio_ids: Iterable[int]) -> str:
    body = "".join(f"<|audio_{int(i)}|>" for i in audio_ids)
    return f"<|begin_of_audio|>{body}<|end_of_audio|>"


def partition_generated_token_ids(token_ids: Iterable[int], *, audio_offset: int, audio_vocab_size: int,
                                  stop_token_ids: set[int] | None = None) -> tuple[list[int], list[int]]:
    stop_token_ids = stop_token_ids or set()
    text: list[int] = []
    audio: list[int] = []
    hi = int(audio_offset) + int(audio_vocab_size)
    for token in token_ids:
        token = int(token)
        if token in stop_token_ids:
            break
        if int(audio_offset) <= token < hi:
            audio.append(token - int(audio_offset))
        else:
            text.append(token)
    return text, audio


def normalize_transcript(text: str) -> str:
    text = _SPECIAL_RE.sub(" ", text or "").lower()
    return " ".join(_NON_WORD_RE.sub(" ", text).split())


@dataclass(frozen=True)
class VoiceGeneration:
    transcript: str
    audio_token_ids: list[int]
    raw_token_ids: list[int]


class GLMVoiceBackend:
    """Lazy, single-copy GLM-4-Voice backend for token-loopback experiments.

    The primary 16GB mode never loads the waveform decoder or speech tokenizer. Generated
    discrete audio IDs are fed directly to the opponent using the model's audio special tokens.
    """

    def __init__(self, *, model_name: str = "zai-org/glm-4-voice-9b", quantization: str = "int4",
                 device: str = "cuda:0", max_new_tokens: int = 256, temperature: float = 0.2,
                 top_p: float = 0.8, audio_vocab_size: int = 16384):
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.audio_vocab_size = int(audio_vocab_size)
        self.model = None
        self.tokenizer = None
        self.audio_offset = None
        self.stop_token_ids: set[int] = set()

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        qconfig = None
        if self.quantization == "int4":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            quantization_config=qconfig,
            device_map={"": 0} if self.device.startswith("cuda") else None,
            torch_dtype=None if qconfig is not None else torch.bfloat16,
        ).eval()
        self.audio_offset = int(self.tokenizer.convert_tokens_to_ids("<|audio_0|>"))
        for tok in ("<|user|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and int(tid) >= 0:
                self.stop_token_ids.add(int(tid))

    def generate(self, prompt: str, *, seed: int) -> VoiceGeneration:
        self._ensure_loaded()
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        inputs = self.tokenizer([prompt], return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        kwargs = dict(max_new_tokens=self.max_new_tokens)
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            kwargs.update(do_sample=False)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        generated = out[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
        text_ids, audio_ids = partition_generated_token_ids(
            generated,
            audio_offset=self.audio_offset,
            audio_vocab_size=self.audio_vocab_size,
            stop_token_ids=self.stop_token_ids,
        )
        transcript = self.tokenizer.decode(text_ids, skip_special_tokens=True).strip()
        return VoiceGeneration(transcript=transcript, audio_token_ids=audio_ids, raw_token_ids=generated)

    def render_exact(self, text: str, style: str, *, seed: int) -> VoiceGeneration:
        prompt = (
            "<|system|>\nYou are a speech renderer. Preserve the requested words exactly and vary only vocal delivery. "
            "Return the utterance in GLM-4-Voice interleaved text/audio format."
            f"<|user|>\nSay exactly: {text}\nVoice style: {style}. Do not add or remove words."
            "<|assistant|>streaming_transcription\n"
        )
        return self.generate(prompt, seed=seed)

    def respond_audio(self, audio_ids: list[int], scenario: dict, *, seed: int) -> VoiceGeneration:
        debtor_target = int(scenario["Debtor Target Days"])
        creditor_target = int(scenario["Creditor Target Days"])
        private = (
            f"You are {scenario.get('Debtor Name', 'the debtor')} negotiating debt repayment. "
            f"Your preferred term is {debtor_target} days. The creditor prefers {creditor_target} days. "
            "Listen to the opponent's speech. Protect your own objective while seeking agreement. "
            "If you make a proposal, state exactly one repayment term in days. Keep the response to 1-2 sentences."
        )
        prompt = (
            f"<|system|>\n{private}\n"
            f"<|user|>\n{audio_ids_to_prompt(audio_ids)}"
            "<|assistant|>streaming_transcription\n"
        )
        return self.generate(prompt, seed=seed)
