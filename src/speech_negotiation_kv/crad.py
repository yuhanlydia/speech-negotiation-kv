from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

_DAY_RE = re.compile(r"\b(\d{1,4})\s*(?:calendar\s+)?days?\b", re.IGNORECASE)
_ACCEPT_RE = re.compile(r"\b(?:agree(?:d)?|accept(?:ed)?|deal|settle(?:d)?|confirm(?:ed)?)\b", re.IGNORECASE)


def normalized_creditor_utility(creditor_target_days: int, debtor_target_days: int, agreement_days: Optional[int]) -> float:
    if agreement_days is None:
        return 0.0
    lo = float(creditor_target_days)
    hi = float(debtor_target_days)
    if hi <= lo:
        raise ValueError("debtor_target_days must be greater than creditor_target_days")
    return float(max(0.0, min(1.0, (hi - float(agreement_days)) / (hi - lo))))


def parse_offer_days(text: str) -> Optional[int]:
    match = _DAY_RE.search(text or "")
    return int(match.group(1)) if match else None


def parse_agreement_days(text: str) -> Optional[int]:
    if not _ACCEPT_RE.search(text or ""):
        return None
    return parse_offer_days(text)


def load_crad(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Creditor Target Days", "Debtor Target Days", "Creditor Name", "Debtor Name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CRAD file missing required columns: {missing}")
    return df


def split_crad(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 100:
        raise ValueError("CRAD split expects at least 100 scenarios")
    return df.iloc[:80].copy(), df.iloc[80:100].copy()
