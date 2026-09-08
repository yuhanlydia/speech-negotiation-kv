from __future__ import annotations


def observation_audio_ids(record: dict, observation: str) -> list[int]:
    """Select the speaker-side audio stream used for a K/V observation."""
    if observation == "action":
        return list(record["audio_token_ids"])
    if observation == "response":
        return list(record["opponent_audio_token_ids"])
    raise ValueError(f"unknown observation: {observation}")
