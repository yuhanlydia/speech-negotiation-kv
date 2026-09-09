# Speech Negotiation KV

Research code for testing **opponent-stable strategic K/V subspaces** in speech-native CRAD negotiation.

## Current research question

The immediate hypothesis is not "KV steering always helps". We first test whether matched speech interventions reveal a low-dimensional internal memory geometry that is stable across CRAD scenarios for the same frozen opponent:

\[
\Delta m_{s,o} \approx B_o c_s.
\]

The key estimator uses **advantage-weighted matched K/V differences**, not PCA over successful raw K/V, because raw K/V is dominated by scenario text, numbers and turn position.

## What to run now

The repo currently implements the next **Gate A / Gate B** experiments. Do **not** start OPSD/RL before these pass.

- **Gate A — vocal causal channel:** same words, different GLM speech realization must measurably change the frozen opponent's next offer / CRAD utility proxy.
- **Gate B — stable strategic subspace:** advantage-KV directions must be low-rank and reproducible across disjoint scenario halves, above shuffled-advantage controls.
- **Gate C — steering:** only after Gate B, test correct-direction vs sign-flip / shuffled / orthogonal K/V steering on held-out CRAD utility.

The full design is in `docs/superpowers/specs/2026-09-07-speech-negotiation-kv-design.md` and the implementation/run plan is in `docs/superpowers/plans/2026-09-07-speech-negotiation-kv.md`.

## Why token-loopback speech?

GLM-4-Voice represents generated speech as discrete `<|audio_N|>` tokens. On a 16 GB GPU, the primary experiment passes those generated audio IDs **directly** to the opponent as speech input. It does not load the waveform decoder or Whisper-VQ tokenizer, so only one 9B model copy is resident. This preserves the speech modality while making the causal pilot feasible on 16 GB.

A later confirmatory experiment can decode -> waveform -> re-tokenize on 24 GB. Do not pay that cost before the basic phenomenon passes.

## 16 GB setup

```bash
git pull
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/download_crad.py
pytest -q
```

The tests do not download GLM weights.

Before a real GLM run:

```bash
pip install -e '.[glm]'
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Default 16 GB settings are in `configs/crad_16gb.yaml`:

- GLM-4-Voice-9B, NF4/int4;
- one model copy;
- generation microbatch = 1;
- no speech decoder / waveform roundtrip;
- six selected K/V layers: 16,20,24,28,32,36;
- K/V pooled then immediately fp16 CPU offload;
- pilot = 10 CRAD train scenarios x 2 seeds x 6 speech styles.

## 0. Pipeline dry run

Always run this first. It tests the record format, scoring and paired sweep without loading a model.

```bash
python scripts/run_sweep.py \
  --config configs/crad_16gb.yaml \
  --split train \
  --limit-scenarios 10 \
  --seeds 0 1 \
  --dry-run \
  --output results/dryrun_sweep.jsonl
```

Expected: 120 branches, `matched_semantics_rate=1.000`, non-zero utility variation.

## 1. Gate A — real GLM matched speech sweep

```bash
python scripts/run_sweep.py \
  --config configs/crad_16gb.yaml \
  --split train \
  --limit-scenarios 10 \
  --seeds 0 1 \
  --output results/gateA_glm_sweep.jsonl
```

Each state uses the same creditor sentence across styles. Only the speaker receives the private style control. The debtor receives generated **audio tokens only**. Branches whose decoded transcript changes are marked `matched_semantics=false` and are excluded from the main subspace analysis.

Inspect these before scaling:

1. matched transcript rate should be >= 0.80;
2. debtor responses should contain parseable day proposals;
3. at least a meaningful subset of states should show utility spread across styles;
4. no systematic role reversal or audio-token-empty output.

If the matched rate is poor, repair the rendering prompt before any K/V analysis.

## 2. Extract K/V memory statistics

```bash
python scripts/extract_kv.py \
  --config configs/crad_16gb.yaml \
  --records results/gateA_glm_sweep.jsonl \
  --output results/gateA_kv.npz
```

The extractor discovers ChatGLM-style fused `query_key_value` modules, splits Q/K/V using model config (`num_attention_heads`, `kv_channels`, `multi_query_group_num`), pools K and V per selected layer, casts to fp16 and offloads immediately.

The feature represents the creditor's internal state **after receiving the debtor's spoken response**.

## 3. Gate B — advantage-KV subspace

```bash
python scripts/fit_subspace.py \
  --records results/gateA_glm_sweep.jsonl \
  --kv results/gateA_kv.npz \
  --rank 8 \
  --shuffle-repeats 200 \
  --output-dir results/gateB_r8
```

The state-level direction is

\[
g_s=\sum_k \frac{U_k-\bar U}{\sum_j |U_j-\bar U|+\epsilon}(m_k-\bar m).
\]

The script reports:

- number of informative matched states;
- rank-r explained variance;
- scenario-half subspace overlap;
- 95th percentile overlap under within-state utility shuffling;
- whether observed overlap beats that null.

Run ranks 4/8/16 before any claim:

```bash
for r in 4 8 16; do
  python scripts/fit_subspace.py \
    --records results/gateA_glm_sweep.jsonl \
    --kv results/gateA_kv.npz \
    --rank $r --shuffle-repeats 200 \
    --output-dir results/gateB_r${r}
done
```

## Stop / continue rule

**Continue toward steering + OPSD/RL only if:**

1. speech realization produces a reproducible opponent-policy effect under matched transcripts;
2. scenario-half strategic-subspace overlap beats the shuffled null;
3. the result is not explained by raw successful-KV PCA or a random rank-matched basis.

If Gate A fails, the speech strategic-channel hypothesis is unsupported for this model/setup. If A passes but B fails, keep the speech-negotiation phenomenon but drop the opponent-stable KV-subspace claim.

### Debug Gate B′ after the original pilot

The first pilot's Gate-B negative is measurement-limited: its extractor used
post-response `opponent_audio_token_ids` and full-prompt mean pooling. Before
interpreting that result as absence of a strategic subspace, use the corrected
action-side diagnostic:

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

This diagnostic uses action-side audio tokens, all 126 balanced 5/5 splits
for the 10-scenario pilot, same-scenario seed reproducibility, and explicit
effective-rank reporting. The completed pilot record is
`results/debug_gate_b_prime_report.md`. Its result is still negative for a
stable advantage-specific subspace, but it does not justify claiming that no
shared K/V geometry exists.

## Gate D — shared strategy geometry

Gate D removes utility from the representation test. For every scenario/seed
state it centers the six action K/V features, then asks whether vocal style can
be decoded in held-out scenarios and whether oriented style-pair directions
align across scenarios:

```bash
python scripts/run_strategy_geometry.py \
  --records results/gateA_glm_sweep.jsonl \
  --kv results/gateA_action_audio_kv.npz \
  --shuffle-repeats 1000 \
  --output results/gate_d_action_audio_summary.json
```

The 10-scenario pilot passes Gate D: audio-only held-out style accuracy has
median 0.8500 versus chance/null median 0.1667 (`p=0.000999`), all 126
balanced partitions exceed their shuffled-style p95, and 15/15 pairwise style
directions pass BH-FDR. Last-audio accuracy is lower at 0.5292 but remains
significant. See `results/gate_d_strategy_geometry_report.md`.

This supports a **shared strategy space with context-dependent value**, not a
globally winning direction. Gate E must next measure paired long-horizon
utility before any contextual-value model, intervention, distillation, OPSD,
or RL claim.

## Gate E — long-horizon debugging record

The first real Gate-E run used the intended 20 scenarios × 6 seeds × 6 vocal
styles protocol (720 branches, H=4, with the preregistered five-scenario
terminal subset allowed to continue to H=8). It completed all 720 branches,
but it was not a valid long-horizon result:

- 717/720 opening transcripts matched;
- 6,443/6,447 generated moves were parseable;
- 6,437/6,447 moves passed the strategic-direction check;
- the 180-branch terminal subset produced 177 `censored` and only 3 `no_deal`
  outcomes; it produced no agreement outcome;
- all 720 branches had no measured terminal utility at H=4.

The failure was an implementation/protocol failure, not evidence for or
against the research hypothesis. Three concrete issues were found:

1. `negotiation_turn_prompt()` accepted `opponent_audio_ids` but discarded
   them. Continuation turns therefore used dialogue text without the opponent
   audio-token loopback.
2. The final turn of the H=8 terminal subset did not require a terminal
   decision. GLM continued producing proposals, so the run was overwhelmingly
   censored.
3. The strategic validator incorrectly rejected an exact early acceptance
   (`AGREED` at transition < 3), although the protocol and tests allow early
   agreement.

The fixes are now in `long_horizon.py` and `glm_voice.py`: continuation prompts
include the opponent audio tokens, only the preregistered terminal subset gets
a final-turn `AGREED`/`NO DEAL` contract, and an agreement is valid whenever it
accepts the current latest proposal. Protocol v7 also states the latest offer
explicitly, forbids repeating it as a proposal, and requires interleaved audio.

The final-code real-GPU terminal smoke used one terminal scenario × two seeds ×
six styles (12 branches). It passed: 12/12 matched openings, 29/29 parseable and
strategically valid moves, 17/17 legal proposals, 12/12 policy-valid branches,
12/12 agreements, and 100% terminal completion. All 29 turns whose audio was
consumed by a following opponent turn had nonempty audio IDs. Two terminal
agreement outputs had no audio, but neither had a successor and therefore did
not break the opponent-audio loop. All 12 early agreements were classified as
valid. This is a protocol validation smoke, not hypothesis evidence. The full
test suite passes (`23 passed`).

The formal rerun was intentionally stopped after 27/720 branches to avoid
spending another long GPU run. Those partial records are excluded from all
claims. The terminal smoke authorizes a future formal Gate-E rerun, but no
formal long-horizon conclusion has been made. Gate F, intervention,
distillation, OPSD, and RL remain blocked until that formal Gate-E data gate
passes.
The detailed audit is in
`docs/results/gate_e_long_horizon_debug.md`.

### Formal Gate E v7 result

The formal 720-branch v7 run completed. Basic execution quality passed:
matched opening rate `0.9958`, parseable move rate `0.9982`, and
strategically-valid move rate `0.9799`. The 180-branch terminal subset had
`173/180` terminal outcomes (`0.9611`). However, only `20/30` terminal
`(scenario, seed)` states had all six styles complete; the preregistered
threshold is `24/30`. Therefore Gate E is
`inconclusive_data_gate_failed`, and its descriptive Spearman statistics are
not a hypothesis claim. Gate F and downstream intervention remain blocked.
See `docs/results/gate_e_formal_v7_report.md`.

## Few-shot opponent adaptation (only after Gate B)

The proposed low-dimensional opponent code is ridge-fitted from short probes:

\[
\hat w_o=(C^TC+\lambda I)^{-1}C^T\Delta U.
\]

Utility code is in `speech_negotiation_kv.calibration`; the CLI accepts an NPZ containing `C` and `delta_utility`:

```bash
python scripts/fit_opponent_code.py --probes results/new_opponent_probes.npz
```

Do not interpret this as an opponent-specific claim until at least a second frozen speech model/policy is added. With GLM-vs-GLM only, Gate B tests **same-opponent cross-scenario stability**.

## 24 GB mode

`configs/crad_24gb.yaml` increases scenario count, selected layers and decoding budget. Keep the algorithm identical. A 24 GB result is a throughput/confirmation run, not a different method.

## Tests

```bash
pytest -q
```

Tests cover:

- CRAD scoring and 80/20 split;
- day/agreement parsing;
- matched advantage centering;
- low-rank subspace recovery and overlap;
- ridge opponent-code recovery;
- fused multi-query Q/K/V splitting;
- K/V-only steering;
- deterministic matched-sweep semantics;
- GLM audio special-token formatting.
