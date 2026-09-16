# ParaGeo — ICASSP 2027

**Compositional Paralinguistic Geometry for Controllable Speech Generation**

Current experiment version: **`icassp2027-r2-reviewer-controls`**.

This repository is frozen around the ICASSP 2027 four-page submission. The paper asks whether a pretrained speech language model contains a shared, content-invariant geometry for broad paralinguistic behavior, and whether that geometry supports training-free **unseen attribute composition** and **intra-utterance dynamic control**.

The older CRAD speech-negotiation experiments remain only as discovery history. Gate F/F2 were negative and should not be rerun for favorable seeds.

## Paper claim

For matched lexical content `x` and paralinguistic control `a`, ParaGeo centers the internal speech representation within content,

\[
\tilde h(x,a)=h(x,a)-\mathbb E_{a'}[h(x,a')],
\]

fits a low-rank shared basis, and removes a semantic content subspace. Each vocal behavior is represented by a coordinate `c_a` in the resulting basis.

The ICASSP paper tests:

1. **Static control**
   \[
   \Delta h=\alpha B_{\rm para}c_a.
   \]
2. **Unseen composition**
   \[
   c_{a_1+\cdots+a_m}=\sum_i c_{a_i}.
   \]
3. **Dynamic control**
   \[
   c_t=(1-\lambda_t)c_a+\lambda_t c_b.
   \]

Activation steering itself is not claimed as novel. The contribution is the **shared content-invariant geometry**, plus its compositional and time-varying use.

## Current empirical status

Historical discovery evidence:

- vocal causal channel: **PASS**;
- shared action-side vocal-strategy K/V geometry: **STRONG PASS** (held-out six-way decoding median `0.8500`, chance `0.1667`; `126/126` scenario splits above shuffled p95; `15/15` pairwise style directions BH-FDR significant);
- immediate-to-terminal alignment in CRAD: **STRONG PASS**;
- negotiation selector/search Gate F/F2: **NEGATIVE**.

Real GLM-4-Voice waveform development runs also established a safe steering regime. The composition dev subset has **zero additional text-channel WER for every tested alpha**; dynamic has safe settings at `0.25`, `1.0`, and `1.5`; static is safest at `1.5`. These are fidelity results, not yet a held-out control-score claim.

## ICASSP r2 reviewer controls

Two reviewer-critical controls are now part of the default pipeline.

### 1. Shuffled geometry null

During geometry fitting, attribute labels are **independently permuted within each lexical content**. This preserves each sentence's marginal activation structure while destroying cross-content attribute identity. The summary reports real vs null:

- leave-one-content-out (LOCO) attribute decoding;
- same-attribute cross-content cosine;
- permutation p-values.

Default: `1000` permutations with frozen seed `15242424242`.

### 2. Norm-matched full-space composition baseline

For composition only, the pipeline now evaluates ordinary full-dimensional mean-difference vector addition:

\[
d^{\rm full}_{a+b}=d^{\rm full}_a+d^{\rm full}_b.
\]

The resulting vector is **norm-matched per sample** to the corresponding ParaGeo direction before intervention. Therefore the comparison tests directional structure rather than injection magnitude.

Held-out composition now compares:

- prompt-only;
- ParaGeo;
- norm-matched random direction;
- **norm-matched full-space vector addition**.

The final summary reports ParaGeo's preference advantage over this full-space baseline.

## Backbone and benchmark

- Backbone: `zai-org/glm-4-voice-9b`, NF4/int4.
- Official waveform path: benchmark WAV -> Whisper-VQ -> GLM-4-Voice-9B -> Flow/HiFT -> 22.05-kHz WAV.
- Benchmark: SpeechParaling-Bench English.
- Main tasks:
  - `para_con/short_sin`: static single-attribute control;
  - `para_con/short_multi`: unseen multi-attribute composition;
  - `dyn_var`: dynamic variation.
- Current catalog: 80 reusable controls across 12 dimensions.

## Frozen protocol

After catalog filtering:

```text
dev:     eligible index % 5 == 0
heldout: all remaining eligible items
```

Development selects alpha independently for each task from:

```text
0.25, 0.5, 1.0, 1.5
```

using the official SpeechParaling pairwise judge subject to mean text-channel WER degradation `<= 0.02`. Held-out data never change hyperparameters.

Main methods:

- `prompt_only`
- `main` ParaGeo: semantic-orthogonal rank-16 basis, all six layers
- `random`: norm-matched random direction
- `full_space`: **composition only**, norm-matched raw full-dimensional vector addition

Fixed ablations (first 24 benchmark-index-sorted held-out eligible items per task):

- no semantic orthogonalization (`raw_basis`)
- rank `4/8/16/32`
- early `[16,20]`, middle `[24,28]`, late `[32,36]`, all six layers
- composition: sum / mean / unit-normalized sum
- dynamic: scheduled trajectory / static start / static end / static midpoint

## Evaluation

The wrapper calls the official SpeechParaling English pairwise judge and changes only runtime file paths. It does not rewrite the upstream evaluation prompt.

Candidate win/tie/loss is mapped to `1 / 0.5 / 0`, giving a 0--100 preference score where 50 is parity. The pipeline reports sample-level 10,000-repeat bootstrap 95% CIs, per-dimension preference, and paired text-channel WER degradation.

A core task passes the paper gate only if:

- static/composition gain is at least `+5`, or dynamic gain at least `+8`;
- mean WER degradation is `<= 0.02`;
- norm-matched random steering stays below the threshold and below ParaGeo.

For the **composition geometry claim**, also report ParaGeo minus full-space preference. If full-space vector addition matches ParaGeo, do not claim that the low-rank geometry itself is necessary.

## One-command run

```bash
git pull origin main
source .venv/bin/activate
pip install -e '.[dev,glm,parageo]'
pytest -q
python -m compileall -q src scripts

export GLM_VOICE_REPO=/absolute/path/to/GLM-4-Voice
export GLM_VOICE_DECODER=/absolute/path/to/glm-4-voice-decoder
export SPEECHPARALING_ROOT=/absolute/path/to/SpeechParaling-Bench

python scripts/run_icassp2027_all.py --stage all --dry-run
python scripts/run_icassp2027_all.py --stage all
```

Stages remain resumable:

```bash
python scripts/run_icassp2027_all.py --stage geometry
python scripts/run_icassp2027_all.py --stage dev
python scripts/run_icassp2027_all.py --stage main
python scripts/run_icassp2027_all.py --stage ablations
python scripts/run_icassp2027_all.py --stage judge
python scripts/run_icassp2027_all.py --stage summarize
```

## Outputs

After `summarize`:

```text
results/parageo_basis_summary.json              # includes shuffled geometry null
results/icassp2027/summary.json                 # includes full-space composition baseline
paper/generated/main_results.tex
paper/generated/geometry_table.tex
paper/generated/ablation_table.tex
paper/generated/result_macros.tex
```

Key implementation files:

- `src/speech_negotiation_kv/icassp_reviewer_controls.py` — shuffled null + norm-matched full-space composition;
- `src/speech_negotiation_kv/icassp_variants.py` — rank/raw/layer/composition/dynamic variants;
- `src/speech_negotiation_kv/icassp_eval.py` — judge aggregation and LaTeX output;
- `scripts/fit_parageo_basis.py` — geometry fit + null test;
- `scripts/run_icassp_generation.py` — main/ablation/full-space generation;
- `scripts/run_icassp2027_all.py` — one-command orchestrator;
- `scripts/summarize_icassp2027.py` — final JSON + paper tables.

The supplied ICASSP template permits four content pages plus a fifth references-only page; keep all experimental analysis inside the four content pages.
