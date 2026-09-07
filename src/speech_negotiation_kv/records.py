from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class BranchRecord:
    branch_id: str
    state_id: str
    scenario_id: int
    semantic_id: str
    style: str
    seed: int
    transcript: str
    audio_token_ids: list[int]
    opponent_transcript: str
    opponent_audio_token_ids: list[int]
    opponent_offer_days: Optional[int]
    utility: float
    matched_semantics: bool = True
    score_type: str = "next_offer_proxy"


def write_jsonl(path: str | Path, rows: Iterable[BranchRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
