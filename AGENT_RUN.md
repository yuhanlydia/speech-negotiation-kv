# Agent Run Instructions

This is the execution contract for the next GPU agent. Pull `main` and run the gates in order. Do not start OPSD/RL or large sweeps before Gate A and Gate B pass.

## 1. Pull and install

```bash
git pull origin main
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/download_crad.py
pytest -q
pip install -e '.[glm]'
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Use `configs/crad_16gb.yaml` on a 16 GB GPU. It loads one GLM-4-Voice-9B copy in NF4/int4 and uses direct audio-token loopback; do not load waveform decoder/tokenizer in the first run.

## 2. Mandatory dry run

```bash
python scripts/run_sweep.py \
  --config configs/crad_16gb.yaml \
  --split train \
  --limit-scenarios 10 \
  --seeds 0 1 \
  --dry-run \
  --output results/dryrun_sweep.jsonl
```

Expected: 120 branches, `matched_semantics_rate=1.000`, nonzero utility variation. If this fails, stop and fix the pipeline before downloading/running GLM.

## 3. Gate A: real GLM matched speech sweep

```bash
python scripts/run_sweep.py \
  --config configs/crad_16gb.yaml \
  --split train \
  --limit-scenarios 10 \
  --seeds 0 1 \
  --output results/gateA_glm_sweep.jsonl
```

Before proceeding, audit:

- matched transcript rate >= 0.80;
- generated creditor branches contain nonempty audio token IDs;
- debtor responses contain parseable day proposals on a useful fraction of branches;
- creditor/debtor roles are not systematically reversed;
- at least a meaningful subset of matched states shows nonzero utility spread across speech styles.

If matched transcript rate is below 0.80, or audio is empty, stop. Treat this as an implementation/prompting issue, not evidence against the research hypothesis.

If transcripts are matched but opponent responses/utility do not vary across styles, record that as a Gate-A negative result and stop the KV-subspace program for this setup.

## 4. Extract K/V

```bash
python scripts/extract_kv.py \
  --config configs/crad_16gb.yaml \
  --records results/gateA_glm_sweep.jsonl \
  --output results/gateA_kv.npz
```

This extracts selected ChatGLM fused-QKV K/V statistics for the creditor after receiving the debtor's speech, then immediately offloads pooled fp16 features to CPU.

## 5. Gate B: ranks 4/8/16

```bash
for r in 4 8 16; do
  python scripts/fit_subspace.py \
    --records results/gateA_glm_sweep.jsonl \
    --kv results/gateA_kv.npz \
    --rank $r \
    --shuffle-repeats 200 \
    --output-dir results/gateB_r${r}
done
```

Primary Gate-B evidence:

- `scenario_half_overlap`;
- `shuffled_overlap_p95`;
- `random_subspace_overlap_p95`;
- `raw_best_kv_half_overlap`;
- `explained_variance`.

Continue only if the advantage-KV subspace reproducibility beats shuffled and random rank-matched nulls and is not merely reproduced by raw-best-KV PCA.

## 5.5 Debug Gate B′ if the original Gate B is negative

Do not interpret the original negative as proof that no strategic K/V geometry
exists. The original extractor is post-response and full-prompt pooled. Run
the corrected action-side diagnostic before closing the question:

```bash
python scripts/extract_kv.py \
  --config configs/crad_16gb.yaml \
  --records results/gateA_glm_sweep.jsonl \
  --observation action --pooling audio_only \
  --output results/gateA_action_audio_kv.npz
python scripts/fit_gate_b_prime.py \
  --records results/gateA_glm_sweep.jsonl \
  --kv results/gateA_action_audio_kv.npz \
  --shuffle-repeats 200 \
  --output results/debug_gate_b_prime_action_audio_summary.json
```

Gate B′ must report same-scenario seed reproducibility and all unique balanced
5/5 scenario splits (126 for the 10-scenario pilot). Keep action-side,
last-audio, and per-layer controls separate. Do not proceed to Gate C, OPSD,
or RL unless the corrected advantage-specific result beats the shuffled null
with reproducible seed-level directions. See
`results/debug_gate_b_prime_report.md` for the completed pilot record.

## 5.6 Gate D: strategy geometry before value modeling

If B′ shows shared geometry but no stable advantage direction, test style
representation without utility:

```bash
python scripts/run_strategy_geometry.py \
  --records results/gateA_glm_sweep.jsonl \
  --kv results/gateA_action_audio_kv.npz \
  --shuffle-repeats 1000 \
  --output results/gate_d_action_audio_summary.json
```

Gate D uses all 126 balanced 5/5 partitions bidirectionally, a within-state
shuffled-style null, and BH-FDR over all 15 pairwise style directions. Gate D
passes only when held-out style decoding is significant and at least one
positive pairwise direction survives FDR. A pass authorizes long-horizon Gate
E data collection, not Gate F, intervention, distillation, OPSD, or RL.

## 6. What to commit/push after the run

Raw JSONL/NPZ/audio artifacts are intentionally gitignored. Keep them locally. Commit lightweight summaries so the research analysis can be reviewed from GitHub:

```bash
cat > results/GATE_REPORT.md <<'EOF'
# Gate A/B Run Report

- GPU:
- VRAM peak:
- Model revision:
- Config:
- Gate A branch count:
- Matched transcript rate:
- Nonempty audio rate:
- Parseable opponent-offer rate:
- Fraction of states with nonzero utility spread:
- Gate A interpretation:

## Gate B

Paste rank-4/8/16 `summary.json` values here and note whether each beats shuffled/random/raw-KV controls.

## Errors / warnings

Paste any OOM, generation, QKV-hook, or parsing failures here.
EOF

git add results/GATE_REPORT.md results/gateB_r*/summary.json
git commit -m "results: add CRAD speech Gate A/B pilot"
git push origin main
```

Do not commit model weights, raw K/V NPZ, raw sweep JSONL, or audio.

## 7. Stop rule

- Gate A negative under valid matched speech -> do not spend GPU on OPSD/RL.
- Gate A positive, Gate B negative -> retain the speech-negotiation phenomenon but drop the stable-KV-subspace claim.
- Gate A+B positive -> next task is held-out Gate-C steering (correct direction vs sign flip, shuffled, random, orthogonal) and only then OPSD/RL / few-shot opponent adaptation.

The full motivation and formulas are in `README.md` and `docs/superpowers/specs/2026-09-07-speech-negotiation-kv-design.md`.
