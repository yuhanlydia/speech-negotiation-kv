#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from speech_negotiation_kv.crad import load_crad, split_crad

URL = "https://raw.githubusercontent.com/Yunbo-max/EmoDistill/main/data/credit_recovery_scenarios.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/credit_recovery_scenarios.csv")
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        print(f"Downloading CRAD -> {out}")
        urlretrieve(URL, out)
    df = load_crad(out)
    train, test = split_crad(df)
    print(f"CRAD OK: {len(df)} rows; train={len(train)}, test={len(test)}")


if __name__ == "__main__":
    main()
