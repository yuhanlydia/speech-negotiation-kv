# ParaGeo — ICASSP 2027

**ParaGeo: Content-Invariant Paralinguistic Geometry for Compositional and Dynamic Speech Control**

This repository is now frozen around the **ICASSP 2027 four-page submission path**. The current paper studies whether a pretrained speech language model contains a shared low-dimensional geometry for broad paralinguistic behavior, and whether that geometry supports training-free **unseen attribute composition** and **intra-utterance dynamic control**.

The older CRAD speech-negotiation experiments remain in the repository as the discovery history. Their Gate F and Gate F2 method attempts were negative and should not be rerun for favorable seeds.

## Paper claim

For matched lexical content `x` and paralinguistic control `a`, ParaGeo centers the internal speech representation within content:

\[
\tilde h(x,a)=h(x,a)-\mathbb E_{a'}[h(x,a')],
\]

fits a low-rank shared basis, and removes a semantic content subspace. Each vocal behavior is represented only by a coordinate `c_a` in the resulting basis.

The ICASSP paper tests three operations:

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

Activation steering itself is not claimed as novel. The contribution is the **content-invariant shared geometry across broad paralinguistic attributes**, plus its compositional and time-varying use.

## Current empirical status

### Historical discovery evidence

- Vocal causal channel: **PASS**.
- Shared action-side vocal-strategy K/V geometry: **STRONG PASS** (held-out six-way decoding median `0.8500`, chance `0.1667`; `126/126` scenario splits above shuffled p95; `15/15` pairwise style directions BH-FDR significant).
- Immediate-to-terminal alignment in CRAD: **STRONG PASS**.
- Negotiation selector/search methods Gate F and F2: **NEGATIVE**.

### ParaGeo development fidelity

Real GLM-4-Voice waveform experiments have already established a safe steering regime. Most notably, the development composition subset has **zero additional text-channel WER for every tested alpha**. Dynamic control has safe settings at alpha `0.25`, `1.0`, and `1.5`; static control is safest at alpha `1.5`. These are fidelity results only; the ICASSP code now selects alpha using the official SpeechParaling pairwise judge subject to the frozen WER gate.

## Backbone and benchmark

- Backbone: `zai-org/glm-4-voice-9b`, NF4/int4.
- Official waveform path: benchmark WAV -> Whisper-VQ -> GLM-4-Voice-9B -> Flow/HiFT -> 22.05-kHz WAV.
- Benchmark: SpeechParaling-Bench English.
- Main tasks:
  - `para_con/short_sin`: static single-attribute control;
  - `para_con/short_multi`: unseen multi-attribute composition;
  - `dyn_var`: dynamic variation.

The current attribute catalog contains 80 reusable controls across 12 dimensions.

## Frozen ICASSP protocol

### Development / held-out

After catalog-coverage filtering:

```text
dev:     eligible index % 5 == 0
heldout: all other eligible examples
```

Development selects alpha independently for each task from:

```text
0.25, 0.5, 1.0, 1.5
```

using the official SpeechParaling pairwise judge, subject to mean text-channel WER degradation `<= 0.02`. Held-out data never change hyperparameters.

### Main methods

- `prompt_only`
- `main` ParaGeo: semantic-orthogonal rank-16 basis, all six layers
- `random`: norm-matched random direction

### Fixed ablations

Ablations use the first 24 benchmark-index-sorted held-out eligible examples per task.

Common:

- no semantic orthogonalization (`raw_basis`)
- rank `4/8/16/32`
- early `[16,20]`, middle `[24,28]`, late `[32,36]`, all six layers

Composition:

- coordinate sum (main)
- mean
- unit-normalized sum

Dynamic:

- scheduled trajectory (main)
- static start
- static endpoint
- static midpoint

Scale sensitivity is the frozen development alpha grid.

## Evaluation

The wrapper calls the **official SpeechParaling English pairwise judge scripts** and changes only runtime file paths. It does not rewrite the upstream evaluation prompt.

Candidate win/tie/loss is mapped to `1 / 0.5 / 0`, giving a 0--100 preference score where 50 denotes parity. The pipeline reports sample-level 10,000-repeat bootstrap 95% CIs, per-dimension preference, and paired text-channel WER degradation.

A core task passes the paper gate only if:

- static/composition preference gain is at least `+5` points, or dynamic gain is at least `+8` points;
- mean WER degradation is `<= 0.02`;
- norm-matched random steering stays below the same threshold and below ParaGeo.

If no held-out task passes, archive the project rather than rescuing it post hoc.

## One-command run

Read **[`ICASSP2027_RUN.md`](ICASSP2027_RUN.md)** first.

Setup:

```bash
git pull origin main
source .venv/bin/activate
pip install -e '.[dev,glm,parageo]'
pytest -q
python -m compileall -q src scripts
```

Required paths:

```bash
export GLM_VOICE_REPO=/absolute/path/to/GLM-4-Voice
export GLM_VOICE_DECODER=/absolute/path/to/glm-4-voice-decoder
export SPEECHPARALING_ROOT=/absolute/path/to/SpeechParaling-Bench
```

Inspect the entire command graph without GPU/API calls:

```bash
python scripts/run_icassp2027_all.py --stage all --dry-run
```

Run the complete experiment matrix:

```bash
python scripts/run_icassp2027_all.py --stage all
```

Stages can also be resumed independently:

```bash
python scripts/run_icassp2027_all.py --stage geometry
python scripts/run_icassp2027_all.py --stage dev
python scripts/run_icassp2027_all.py --stage main
python scripts/run_icassp2027_all.py --stage ablations
python scripts/run_icassp2027_all.py --stage judge
python scripts/run_icassp2027_all.py --stage summarize
```

## Paper outputs

After `summarize`:

```text
results/icassp2027/summary.json
paper/generated/main_results.tex
paper/generated/geometry_table.tex
paper/generated/ablation_table.tex
paper/generated/result_macros.tex
```

Draft paper files:

```text
paper/ICASSP2027_ParaGeo_draft.tex
paper/icassp2027_parageo_refs.bib
```

The draft contains multiple title, abstract, and introduction options plus the recommended Experimental Setup. Generated result macros/tables can be included without manually copying numbers.

## Important ICASSP page rule

The supplied ICASSP 2027 template permits **four content pages plus a fifth page containing references only**. Therefore ablations/mechanism analysis must fit inside the four content pages; do not place experimental appendix material on the fifth page.

## Main implementation files

- `configs/icassp2027_parageo.yaml` — frozen ICASSP matrix.
- `src/speech_negotiation_kv/parageo.py` — geometry and mechanism diagnostics.
- `src/speech_negotiation_kv/icassp_variants.py` — rank/raw/layer/composition/dynamic variants.
- `src/speech_negotiation_kv/icassp_eval.py` — official-judge aggregation, bootstrap, WER, LaTeX rendering.
- `scripts/run_icassp_generation.py` — official-waveform main/ablation generation.
- `scripts/run_official_speechparaling_judge.py` — path-only wrapper around the upstream judge.
- `scripts/select_icassp_alpha.py` — development-only alpha selection.
- `scripts/summarize_icassp2027.py` — final decision + paper tables.
- `scripts/run_icassp2027_all.py` — one-command orchestrator.

## Historical reproducibility

Older CRAD and exploratory ParaGeo reports remain under `results/` and `docs/results/`. They are preserved for provenance but are not the current execution path.
