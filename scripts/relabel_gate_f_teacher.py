#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech_negotiation_kv.crad import load_crad
from speech_negotiation_kv.records import read_jsonl
from speech_negotiation_kv.short_horizon_selector import relabel_teacher_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reparse one-step Gate-F proposal labels from saved transcripts"
    )
    parser.add_argument("--data", default="data/credit_recovery_scenarios.csv")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.fresh and output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    frame = load_crad(args.data)
    targets = {
        int(index): (
            int(row["Creditor Target Days"]),
            int(row["Debtor Target Days"]),
        )
        for index, row in frame.iterrows()
    }
    source = read_jsonl(args.records)
    corrected = relabel_teacher_rows(source, scenario_targets=targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in corrected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    changed_parseable = sum(
        row["opponent_offer_days"] is not None
        and row["opponent_offer_days"] != row["original_opponent_offer_days"]
        for row in corrected
    )
    newly_unparseable = sum(
        row["opponent_offer_days"] is None
        and row["original_opponent_offer_days"] is not None
        for row in corrected
    )
    total_unparseable = sum(row["opponent_offer_days"] is None for row in corrected)
    print(json.dumps({
        "records": args.records,
        "output": str(output),
        "rows": len(corrected),
        "changed_parseable": changed_parseable,
        "newly_unparseable": newly_unparseable,
        "total_unparseable": total_unparseable,
    }, indent=2))


if __name__ == "__main__":
    main()
