#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from speech_negotiation_kv.calibration import fit_ridge_opponent_code


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit low-dimensional opponent code from short probes")
    ap.add_argument("--probes", required=True, help="NPZ with C [n,r] and delta_utility [n]")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--output", default="results/opponent_code.npy")
    args = ap.parse_args()
    data = np.load(args.probes)
    w = fit_ridge_opponent_code(data["C"], data["delta_utility"], ridge=args.ridge)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, w.astype(np.float32))
    print(f"saved opponent code dim={len(w)} -> {out}")


if __name__ == "__main__":
    main()
