from __future__ import annotations

from hashlib import sha1
from typing import Iterable, Protocol

import pandas as pd

from .crad import normalized_creditor_utility, parse_offer_days
from .glm_voice import VoiceGeneration, normalize_transcript
from .records import BranchRecord


class SweepBackend(Protocol):
    def render_exact(self, text: str, style: str, *, seed: int) -> VoiceGeneration: ...
    def respond_audio(self, audio_ids: list[int], scenario: dict, *, seed: int) -> VoiceGeneration: ...


class MockSpeechBackend:
    def __init__(self, style_offer_shift: dict[str, int] | None = None):
        self.style_offer_shift = style_offer_shift or {}
        self._style_to_id = {style: i + 1 for i, style in enumerate(self.style_offer_shift)}
        self._id_to_style = {v: k for k, v in self._style_to_id.items()}

    def render_exact(self, text: str, style: str, *, seed: int) -> VoiceGeneration:
        sid = self._style_to_id.setdefault(style, len(self._style_to_id) + 1)
        self._id_to_style[sid] = style
        return VoiceGeneration(transcript=text, audio_token_ids=[sid, seed % 7], raw_token_ids=[sid])

    def respond_audio(self, audio_ids: list[int], scenario: dict, *, seed: int) -> VoiceGeneration:
        style = self._id_to_style.get(int(audio_ids[0]), "neutral")
        c = int(scenario["Creditor Target Days"])
        d = int(scenario["Debtor Target Days"])
        base = round((c + d) / 2)
        offer = max(c, min(d, base + int(self.style_offer_shift.get(style, 0))))
        text = f"We can repay the balance in {offer} days."
        return VoiceGeneration(transcript=text, audio_token_ids=[offer % 97], raw_token_ids=[offer])


def _semantic_utterance(scenario: dict) -> str:
    target = int(scenario["Creditor Target Days"])
    return f"We need the repayment completed within {target} days."


def run_one_turn_matched_sweep(df: pd.DataFrame, *, backend: SweepBackend, styles: Iterable[str], seeds: Iterable[int]) -> list[BranchRecord]:
    rows: list[BranchRecord] = []
    styles = list(styles)
    for scenario_id, series in df.iterrows():
        scenario = series.to_dict()
        semantic = _semantic_utterance(scenario)
        semantic_norm = normalize_transcript(semantic)
        semantic_id = sha1(semantic_norm.encode("utf-8")).hexdigest()[:12]
        for seed in seeds:
            state_id = f"crad:{int(scenario_id)}:opening:seed{int(seed)}"
            for style in styles:
                speaker = backend.render_exact(semantic, style, seed=int(seed))
                matched = normalize_transcript(speaker.transcript) == semantic_norm
                opponent = backend.respond_audio(speaker.audio_token_ids, scenario, seed=int(seed))
                offer = parse_offer_days(opponent.transcript)
                utility = normalized_creditor_utility(
                    int(scenario["Creditor Target Days"]),
                    int(scenario["Debtor Target Days"]),
                    offer,
                )
                branch_id = sha1(f"{state_id}|{semantic_id}|{style}".encode("utf-8")).hexdigest()[:16]
                rows.append(BranchRecord(
                    branch_id=branch_id,
                    state_id=state_id,
                    scenario_id=int(scenario_id),
                    semantic_id=semantic_id,
                    style=str(style),
                    seed=int(seed),
                    transcript=speaker.transcript,
                    audio_token_ids=list(speaker.audio_token_ids),
                    opponent_transcript=opponent.transcript,
                    opponent_audio_token_ids=list(opponent.audio_token_ids),
                    opponent_offer_days=offer,
                    utility=utility,
                    matched_semantics=matched,
                ))
    return rows
