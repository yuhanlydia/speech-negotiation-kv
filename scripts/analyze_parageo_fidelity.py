#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from speech_negotiation_kv.speechparaling import word_error_rate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--output',default=None); args=ap.parse_args()
    rows=[json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    wers=[word_error_rate(row['target_text'],row.get('transcript','')) for row in rows]
    result={'samples':len(rows),'mean_text_channel_wer':float(np.mean(wers)) if wers else None,'median_text_channel_wer':float(np.median(wers)) if wers else None,'p95_text_channel_wer':float(np.quantile(wers,0.95)) if wers else None}
    if args.output: Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
