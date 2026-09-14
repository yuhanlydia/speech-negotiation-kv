# ICASSP 2027 ParaGeo — Complete Run Contract

This is the frozen experiment path for the four-page ICASSP 2027 ParaGeo submission. Do not tune on held-out judge results. The historical CRAD Gate F/F2 experiments are not part of this run.

## External assets

Clone/download the official dependencies outside this repository:

```bash
git clone --recurse-submodules https://github.com/zai-org/GLM-4-Voice.git
git clone https://huggingface.co/THUDM/glm-4-voice-decoder
git clone https://github.com/Northern-byte-bit/SpeechParaling-Bench.git
cd SpeechParaling-Bench
python script/download_data.py
python script/download_baseline.py
cd ..
```

Export absolute paths:

```bash
export GLM_VOICE_REPO=/absolute/path/to/GLM-4-Voice
export GLM_VOICE_DECODER=/absolute/path/to/glm-4-voice-decoder
export SPEECHPARALING_ROOT=/absolute/path/to/SpeechParaling-Bench
```

SpeechParaling's official judge uses `judge_data/config.py`. Configure its API key/base URL exactly as required by the upstream repository before judge stages. ParaGeo does not replace or modify the upstream judge prompt.

## Environment and verification

```bash
git pull origin main
source .venv/bin/activate
pip install -e '.[dev,glm,parageo]'
pytest -q
python -m compileall -q src scripts
```

The generation path is mandatory:

```text
benchmark WAV -> Whisper-VQ -> GLM-4-Voice-9B -> Flow/HiFT -> output WAV
```

Primary generation is NF4/int4, batch 1, temperature 0.8, top-p 0.8.

## Inspect before running

```bash
python scripts/run_icassp2027_all.py \
  --config configs/icassp2027_parageo.yaml \
  --stage all \
  --dry-run
```

This prints geometry/calibration, four-alpha development generation and judging, held-out main/random generation, fixed-subset ablations, official judging, and LaTeX summarization without GPU/API work.

## One-command full run

```bash
python scripts/run_icassp2027_all.py \
  --config configs/icassp2027_parageo.yaml \
  --stage all
```

Generation resumes from existing manifests/WAVs, and the upstream judge skips existing metadata. To resume stage by stage:

```bash
python scripts/run_icassp2027_all.py --stage geometry
python scripts/run_icassp2027_all.py --stage dev
python scripts/run_icassp2027_all.py --stage main
python scripts/run_icassp2027_all.py --stage ablations
python scripts/run_icassp2027_all.py --stage judge
python scripts/run_icassp2027_all.py --stage summarize
```

`main` and `ablations` require `results/icassp2027/alpha_selection.json`, produced by `dev`.

## Frozen scientific protocol

### Geometry calibration

- Attribute catalog: all reusable single-control items in English `short_sin.jsonl`.
- Lexical calibration: identical eight fixed content sentences for every attribute.
- K/V layers: `[16,20,24,28,32,36]`.
- Main rank: 16; maximum saved rank: 32.
- Semantic subspace rank: 8.

`results/parageo_basis_summary.json` reports rank-16 explained variance, 95%-energy rank, leave-one-content-out attribute decoding, chance, and same-attribute cross-content cosine.

### Development and alpha

After catalog coverage filtering:

```text
dev:     eligible index % 5 == 0
heldout: all other eligible items
```

For each task independently, development selects alpha from `0.25, 0.5, 1.0, 1.5` by the official pairwise judge subject to mean text-channel WER degradation `<= 0.02`. Exact preference ties choose the smallest alpha. Held-out outcomes never change alpha.

### Main held-out tasks

1. `static`: English `para_con/short_sin`.
2. `composed`: English `para_con/short_multi`.
3. `dynamic`: English `dyn_var`.

Methods:

- `prompt_only`
- `main`: semantic-orthogonal rank-16 ParaGeo, all six layers
- `random`: norm-matched random steering

Composition uses coordinate sum. Dynamic uses a scheduled linear path for gradual instructions and a midpoint step for instructions containing `suddenly`.

### Fixed-subset ablations

Ablations use only the first 24 benchmark-index-sorted held-out eligible items per task.

Common:

- `main`
- `raw_basis` (no semantic orthogonalization)
- `rank4`, `rank8`, `rank32`
- `layers_early=[16,20]`
- `layers_middle=[24,28]`
- `layers_late=[32,36]`

Composition only:

- `comp_mean`
- `comp_normalized`

Dynamic only:

- `dyn_start`
- `dyn_end`
- `dyn_midpoint`

Scale sensitivity is the development alpha grid and is never repeated on held-out data.

## Evaluation and statistics

All candidates are paired with prompt-only GLM using the **official SpeechParaling English judge scripts**. The wrapper changes runtime paths only.

Decision mapping:

```text
candidate win  = 1
tie            = 0.5
candidate loss = 0
```

Preference score is `100 * mean(pairwise score)`, so 50 denotes parity. The summarizer computes sample-level 10,000-repeat bootstrap 95% CIs, per-dimension preferences, and paired text-channel WER degradation.

## Frozen paper gate

A core task passes only when all three conditions hold:

1. ParaGeo preference gain over 50 is at least +5 points for static/composition or +8 points for dynamic.
2. Mean text-channel WER degradation vs prompt-only is `<= 0.02`.
3. Norm-matched random steering remains below the same gain threshold and below ParaGeo.

If no held-out task passes, archive the paper. Do not rescue it with new seeds, post-hoc layers, schedules, RL, or LoRA.

## Paper outputs

After `summarize`:

```text
results/icassp2027/summary.json
paper/generated/main_results.tex
paper/generated/geometry_table.tex
paper/generated/ablation_table.tex
paper/generated/result_macros.tex
```

Draft:

```text
paper/ICASSP2027_ParaGeo_draft.tex
paper/icassp2027_parageo_refs.bib
```

## ICASSP page-budget rule

The supplied ICASSP 2027 template permits four content pages plus a references-only fifth page. Therefore any experiment “appendix” must be integrated compactly into the four content pages; do not put experiments on page five.
