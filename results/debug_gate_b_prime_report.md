# Debug Gate B′ Report

Date: 2026-09-08

## Scope and correction

The original Gate-B result is not evidence that a strategic K/V subspace does
not exist. `scripts/extract_kv.py` had extracted the creditor representation
after the debtor response, using `opponent_audio_token_ids` and a full-prompt
mean pool. That feature is already downstream of the style intervention and
the observed offer. It therefore confounds response/outcome representation
with the creditor action representation that the hypothesis intended to test.

Gate B′ corrects this by using the creditor's `audio_token_ids`, isolating
audio-token positions, and retaining per-layer K and V components. It also
enumerates all unique balanced 5/5 scenario splits (126 for the 10-scenario
pilot), reports effective ranks after sample-rank truncation, and measures
same-scenario seed reproducibility.

## Inputs and implementation

- Existing Gate-A records: `results/gateA_glm_sweep.jsonl` (120 matched
  branches, no new GLM sweep).
- New action-side extractions:
  - `gateA_action_audio_kv.npz`: action audio-only pooling;
  - `gateA_action_last_audio_kv.npz`: last-audio-token pooling.
- Layers: 16, 20, 24, 28, 32, 36.
- Components: per-layer K, V, K/V and all-layer K/V.
- Ranks: requested 4, 8, 16.
- Primary shuffled null: 200 within-state utility shuffles for every split.
- Layerwise shuffled null: 50 within-state utility shuffles for every split.

The raw NPZ/JSONL files remain local and gitignored. The corresponding
lightweight summaries are committed beside this report.

## Same-scenario reproducibility

Only 7 of 10 scenarios had usable utility variation in both seeds for the
Spearman diagnostic.

| Action pooling | Seed utility Spearman median | Advantage-direction cosine median |
|---|---:|---:|
| audio-only | 0.4330 (n=7) | 0.1743 (n=7) |
| last-audio | 0.4330 (n=7) | 0.1264 (n=7) |

The low direction alignment means the advantage label is not yet a stable
within-scenario object. This limits the power of any cross-scenario subspace
claim.

## Primary exhaustive split result

The table reports the median observed overlap and the median per-split 95th
percentile of the shuffled null over all 126 balanced splits. The final column
is the number of splits where observed overlap exceeded that split's shuffled
95th percentile.

### Action audio-only

| Requested rank | Effective full rank | Observed median | Shuffled p95 median | Passing splits |
|---:|---:|---:|---:|---:|
| 4 | 4 | 0.3385 | 0.4267 | 0/126 |
| 8 | 8 | 0.2717 | 0.3277 | 0/126 |
| 16 | 16 | 0.2772 | 0.3432 | 0/126 |

### Action last-audio

| Requested rank | Effective full rank | Observed median | Shuffled p95 median | Passing splits |
|---:|---:|---:|---:|---:|
| 4 | 4 | 0.2520 | 0.3859 | 0/126 |
| 8 | 8 | 0.2686 | 0.3321 | 0/126 |
| 16 | 16 | 0.2777 | 0.3508 | 0/126 |

For individual half fits, requested rank 16 was often truncated to 6–10
directions because each half contains only the informative state directions.
The summaries report those effective half ranks explicitly. The old Gate-B
rank-16 summary now does the same; its two half ranks are 9 and 7.

## Layerwise result

Layerwise K/V geometry is non-random and often has higher raw overlap than the
all-layer concatenation, but it does not establish advantage specificity. The
best rank-4 layerwise result was only 16/126 passing splits for audio-only
layer-16 V, and 13/126 for last-audio layer-16 K/V. No configuration passed in
a majority of splits.

Full per-layer/rank results are in:

- `results/debug_gate_b_prime_action_audio_layerwise_summary.json`
- `results/debug_gate_b_prime_action_last_audio_layerwise_summary.json`

## Scientific decision

The corrected estimator removes the major action/response confound, but this
pilot still does **not** pass the stable advantage-specific KV gate. The
appropriate conclusion is:

> Action-side audio K/V contains shared geometry, but the current 10-scenario,
> two-seed utility labels do not recover a reproducible cross-scenario
> advantage-bearing subspace.

This is not the old claim that “no KV subspace exists.” It is a corrected
negative for this estimator and sample size, with weak same-scenario seed
reproducibility as the principal limitation. Do not proceed to Gate C, OPSD,
or RL from this result. A future continuation needs either more independent
seeds/episodes and a stronger long-horizon outcome label, or a preregistered
analysis that explicitly targets scenario-specific strategy.
