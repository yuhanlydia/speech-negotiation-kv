# Gate A/B Run Report

Date: 2026-09-08

## Environment

- Repository main: `8719e37e47132033e27ae2067a07ad45ccaffab0`
- GPU: NVIDIA GeForce RTX 3090, 24 GB
- Config: `configs/crad_16gb.yaml`
- Model: `zai-org/glm-4-voice-9b`, local Hub snapshot
  `352e5a64063448d5e46394753fb65bf1dbab975b`
- Quantization: NF4/int4, one model copy, direct audio-token loopback,
  no waveform decoder/tokenizer
- Observed GPU memory: approximately 8.9 GiB at the highest recorded sample;
  no OOM occurred
- Dependency repair: model-compatible `transformers==4.44.1`,
  `accelerate==0.34.2`, and explicit `tiktoken>=0.7` are now declared in
  `pyproject.toml`. The unconstrained latest versions caused model generation
  failures before this repair.

## Preparation checks

- CRAD: 100 rows downloaded; train/test split 80/20
- Unit tests: 9 passed
- Dry-run: 120 branches, `matched_semantics_rate=1.000`,
  `utility_std=0.1281`

## Gate A — matched speech sweep

- Branches: 120 (10 train scenarios × 2 seeds × 6 styles)
- Matched transcript rate: `1.000`
- Creditor generated audio nonempty rate: `1.000`
- Debtor/opponent generated audio nonempty rate: `1.000`
- Parseable opponent-offer rate: `1.000`
- Matched states with utility spread ≥ 0.05: `15/20`
- Matched states with more than one opponent offer across styles: `16/20`
- Utility standard deviation: `0.2508`
- Gate-A interpretation: **positive for this pilot**. Matched speech
  realizations produced nonempty audio and varied the opponent offer/utility
  proxy on a meaningful subset. No systematic creditor/debtor role reversal
  was observed in the transcript/offer audit.

## K/V extraction

- Extracted 120 K/V vectors of dimension 3072
- Selected layers: 16, 20, 24, 28, 32, 36
- Features were pooled and offloaded to CPU as fp16

## Gate B — advantage-KV subspace

| Rank | Explained variance | Scenario-half overlap | Shuffled overlap p95 | Random overlap p95 | Raw-best-KV overlap | Passes shuffled null |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.5888 | 0.1788 | 0.2628 | 0.0022 | 0.2756 | No |
| 8 | 0.8177 | 0.1861 | 0.2276 | 0.0034 | 0.1924 | No |
| 16 | 1.0000 | 0.1928 | 0.2355 | 0.0059 | 0.1712 | No |

Gate-B interpretation: **negative for the stable advantage-KV subspace claim**.
The observed scenario-half overlap is below the shuffled null at all tested
ranks. The random rank-matched null is low, but that alone is insufficient;
the reproducibility gate does not pass.

## Stop decision

Retain the matched-speech negotiation phenomenon as a Gate-A pilot result, but
drop the opponent-stable KV-subspace claim for this setup. Do not run Gate C,
OPSD, RL, or larger sweeps without a new preregistered reason and review.

Raw artifacts remain local and gitignored:

- `results/gateA_glm_sweep.jsonl`
- `results/gateA_kv.npz`
- `results/gateB_r*/basis.npz`

Committed lightweight summaries are the three `results/gateB_r*/summary.json`
files and this report.
