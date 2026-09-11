# ParaGeo: Compositional Paralinguistic Geometry for Speech Language Models

This repository now studies **content-invariant paralinguistic geometry for controllable speech generation**. The original CRAD speech-negotiation experiments are retained as the discovery path that motivated ParaGeo, but negotiation is no longer the main benchmark.

## Current scientific position

The project has four relevant findings:

- **Vocal causal channel — PASS.** Holding words fixed while changing vocal delivery changes a frozen speech agent's behavior.
- **Shared vocal-strategy geometry — STRONG PASS.** In GLM-4-Voice, matched vocal styles occupy transferable action-side K/V directions across held-out CRAD scenarios (median six-way decoding accuracy 0.8500 vs chance 0.1667; 126/126 scenario splits above their shuffled p95; 15/15 style-pair directions BH-FDR significant).
- **Immediate-to-terminal alignment — STRONG CONFIRMATORY PASS.** On fresh CRAD scenarios 20--29, one-step opponent utility predicts terminal utility (mean within-state Spearman 0.6586, bootstrap 95% CI [0.5677, 0.7642], tie-aware best-style agreement 0.950, median regret 0).
- **Negotiation method recovery — NEGATIVE.** Both the frozen pre-response selector (Gate F) and six-probe immediate search (Gate F2) failed their preregistered held-out terminal-utility gates. These negative results remain part of the record and are not overwritten.

The new hypothesis focuses only on the strongest successful branch:

\[
\boxed{\text{paralinguistic behavior occupies a low-dimensional, content-invariant, compositional geometry that can be directly controlled during speech generation}}
\]

## Why SpeechParaling-Bench

[SpeechParaling-Bench](https://github.com/Northern-byte-bit/SpeechParaling-Bench) provides exactly the tasks that should benefit from this geometry:

- **Paralanguage Control** — 691 English + 691 Chinese examples;
- **Dynamic Variation** — 120 English + 120 Chinese examples;
- **Situational Adaptation** — 190 English + 190 Chinese examples;
- 100+ paralinguistic features and 1001 prompts per language;
- public baseline outputs and pairwise LALM judge / score-calculation code.

The first ParaGeo gate uses only English Paralanguage Control and Dynamic Variation. Full-benchmark expansion is blocked until the pilot passes.

## ParaGeo method

### 1. Content-invariant geometry

For matched lexical content `x` rendered with attribute `a`:

\[
\tilde h(x,a)=h(x,a)-\frac{1}{|A|}\sum_{a'}h(x,a').
\]

Fit:

\[
\tilde h(x,a)\approx Bc_a.
\]

The first catalog uses all reusable single-control dimensions published in
SpeechParaling's `short_sin.jsonl`. The current English catalog contains 80 controls
across 12 dimensions, which leaves enough fully covered official items for the frozen
40-item compositional pilot.

### 2. Semantic-orthogonal control

Estimate a semantic/content subspace `S` from content means and project it out:

\[
B_{\rm para}=\operatorname{qr}((I-SS^\top)B).
\]

All steering uses `B_para` to reduce lexical drift.

### 3. Static control

\[
\Delta h=\alpha B_{\rm para}c_a.
\]

### 4. Compositional control

Single-attribute coordinates are added without fitting the combination:

\[
c_{a_1+\cdots+a_m}=\sum_i c_{a_i}.
\]

The public `short_multi` subset tests whether this gives unseen combination control.

### 5. Dynamic control

For an instruction that transitions from attribute `a` to `b`:

\[
c_t=(1-\lambda_t)c_a+\lambda_t c_b.
\]

Gradual prompts use a linear path; prompts containing `suddenly` use a step trajectory. `ScheduledFusedQKVSteerer` applies the corresponding K/V direction once per generation forward.

### 6. Random-direction control

Norm-matched random directions are mandatory. A gain that random steering reproduces is not a ParaGeo result.

## Official GLM-4-Voice waveform path

The old negotiation pilot used direct discrete audio-token loopback for 16 GB efficiency. **The SpeechParaling benchmark path no longer does this for output evaluation.** It uses the official GLM-4-Voice components:

1. `GLM-4-Voice-Tokenizer` / Whisper-VQ for benchmark input waveform tokenization;
2. `GLM-4-Voice-9B` for interleaved text/audio generation;
3. `GLM-4-Voice-Decoder` (Flow + HiFT) for 22.05-kHz output WAV decoding.

The adapter follows the public official implementation in [zai-org/GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice). Primary pilot generation uses temperature 0.8 and top-p 0.8.

## Current pilot

English only:

| Subset | Role | Samples |
|---|---|---:|
| `para_con/short_sin` | static single-attribute control | 80 |
| `para_con/short_multi` | compositional control | 40 |
| `dyn_var` | intra-utterance dynamic control | 60 |

Methods:

- `prompt_only`
- `parageo_static`
- `parageo_composed`
- `parageo_dynamic`
- `random`

The runner first selects only items whose requested controls are fully covered by the frozen catalog. A deterministic 20% development slice (`catalog-covered index % 5 == 0`) is used only to choose `alpha` from `{0.25, 0.5, 1.0, 1.5}`; the runner exposes `--split dev` and `--split heldout`. Exact dev-score ties choose the smallest alpha. The remaining 80% is the held-out pilot.

## Frozen stop rule

Continue only if text-channel semantic fidelity is preserved and at least one held-out result satisfies:

- static/compositional SpeechParaling score gain >= **+5.0 points** over prompt-only; or
- Dynamic Variation score gain >= **+8.0 points** over prompt-only.

Random steering must not reproduce the gain. `analyze_parageo_pilot.py` enforces the score, WER, and random-control gates jointly. If this gate fails, stop ParaGeo; do not rescue it with RL, LoRA, favorable seeds, or post-hoc schedules.

## Quick start

```bash
git pull origin main
source .venv/bin/activate
pip install -e '.[dev,glm,parageo]'
pytest -q
```

External assets:

```bash
export GLM_VOICE_REPO=/absolute/path/to/GLM-4-Voice
export GLM_VOICE_DECODER=/absolute/path/to/glm-4-voice-decoder
export SPEECHPARALING_ROOT=/absolute/path/to/SpeechParaling-Bench
```

Then follow **[`PARAGEO_RUN.md`](PARAGEO_RUN.md)** exactly.

## Main files

### ParaGeo core

- `src/speech_negotiation_kv/parageo.py` — centering, SVD basis, semantic orthogonalization, coordinates, composition, dynamic schedules.
- `src/speech_negotiation_kv/parageo_steering.py` — scheduled fused-QKV intervention.
- `src/speech_negotiation_kv/parageo_eval.py` — dev-only scale selection and frozen score/WER/random stop gate.
- `src/speech_negotiation_kv/speechparaling.py` — SpeechParaling prompt parser, catalog coverage, filename-index pairing, deterministic dev/held-out split, fidelity utilities.
- `src/speech_negotiation_kv/glm_official_waveform.py` — official Whisper-VQ + Flow/HiFT waveform adapter.

### Calibration

- `scripts/build_parageo_catalog.py`
- `scripts/collect_parageo_calibration.py`
- `scripts/extract_parageo_kv.py`
- `scripts/fit_parageo_basis.py`

### Benchmark

- `scripts/run_speechparaling_pilot.py`
- `scripts/select_parageo_scale.py`
- `scripts/analyze_parageo_fidelity.py`
- `scripts/analyze_parageo_pilot.py`
- `configs/parageo_speechparaling_pilot.yaml`

### Design / run contract

- `PARAGEO_RUN.md`
- `docs/superpowers/specs/2026-09-10-parageo-speechparaling-design.md`
- `docs/superpowers/plans/2026-09-10-parageo-speechparaling.md`

## Expansion after a pilot pass only

1. Full 1001-item English SpeechParaling-Bench.
2. Full 1001-item Chinese benchmark.
3. Situational Adaptation via context -> ParaGeo coordinate routing.
4. Qwen-Omni replication.
5. Cross-model geometry rank / composition analysis.
6. External S2S-Arena or WildSpeech-Bench generalization.

## Historical negotiation results

The older CRAD scripts/results remain in the repository for reproducibility. They document the causal discovery, Gate-D geometry, Gate-E2 alignment, and the negative Gate-F/Gate-F2 method attempts. They are no longer the execution path for new experiments.
