# ICASSP 2027 ParaGeo Experiment Design

Date: 2026-09-14

## Goal

Turn the current ParaGeo pilot into a compact, reproducible ICASSP 2027 paper package. The paper makes only three claims that can fit in four content pages:

1. pretrained speech LMs contain a content-invariant low-dimensional paralinguistic geometry;
2. single-attribute coordinates compose at inference time to improve unseen multi-attribute control;
3. time-varying coordinate trajectories improve intra-utterance paralinguistic variation.

Activation steering itself is not claimed as novel. The novelty is the shared geometry across many paralinguistic dimensions and its compositional/dynamic use on SpeechParaling-Bench.

## Backbone and benchmark

- Backbone: `zai-org/glm-4-voice-9b`, int4/NF4, official Whisper-VQ input tokenizer and Flow/HiFT waveform decoder.
- Benchmark: SpeechParaling-Bench English.
- Main tasks:
  - `para_con/short_sin`: static single-attribute control;
  - `para_con/short_multi`: unseen multi-attribute composition;
  - `dyn_var`: intra-utterance dynamic variation.
- Situational adaptation, Chinese evaluation, and second-model replication are out of scope for the four-page submission.

## Calibration and geometry

Build the reusable attribute catalog from all single-control items in `short_sin.jsonl`. Render every catalog attribute on the same eight fixed lexical calibration sentences. Extract action-side K/V from layers `[16,20,24,28,32,36]`.

For feature `h(x,a)` with lexical content `x` and control `a`, use within-content centering:

`h_tilde(x,a) = h(x,a) - mean_a h(x,a)`.

Fit the raw paralinguistic basis with SVD. Estimate a semantic basis from per-content means and remove it from the steering basis. The main rank is 16; rank ablations are 4, 8, 16, and 32.

Report three mechanism diagnostics:

- cumulative explained variance of the raw centered representation;
- leave-one-content-out nearest-centroid attribute accuracy and chance level;
- mean same-attribute cross-content cosine similarity.

These diagnostics are computed from calibration only and never use benchmark judge outcomes.

## Main inference methods

All methods use identical benchmark prompts, generation seeds, GLM sampling parameters, and official waveform decoding.

### Prompt-only

No activation modification.

### ParaGeo static

For requested attribute `a`, inject

`delta_h = alpha * B_para c_a`.

### ParaGeo composition

For an unseen multi-attribute instruction requesting controls `a_1...a_m`, use the additive coordinate

`c = sum_i c_{a_i}`

without fitting any multi-attribute example.

### ParaGeo dynamic

For a prompt transitioning from `a` to `b`, construct a time-varying coordinate. Gradual instructions use linear interpolation; prompts containing `suddenly` use a step transition at the frozen midpoint:

`c_t = (1-lambda_t)c_a + lambda_t c_b`.

### Random control

Use a norm-matched random direction under the same scale and layer set. A result reproduced by random steering is not counted as a ParaGeo gain.

## Data split and alpha selection

Catalog-covered benchmark items are deterministically split by benchmark index:

- development: `eligible_index % 5 == 0`;
- held-out: all remaining items.

Development is used only to select `alpha` from `{0.25, 0.5, 1.0, 1.5}` independently for static, composition, and dynamic control. Selection maximizes pairwise judge preference subject to text-channel WER degradation <= 0.02. Exact score ties choose the smallest alpha.

No hyperparameter is changed after held-out judge outcomes are observed.

## Evaluation

Use the official SpeechParaling pairwise LALM-judge prompt and candidate-vs-baseline protocol. ParaGeo variants are compared directly against prompt-only GLM on exactly matched files.

For each judge metadata file, map candidate win / tie / loss to `1 / 0.5 / 0`. Report:

- candidate preference score = 100 * mean(pairwise score);
- paired bootstrap 95% CI for preference score minus the neutral 50-point tie baseline;
- text-channel WER and degradation vs prompt-only;
- per-dimension preference when available.

Main table reports prompt-only, ParaGeo, and random for the three tasks.

## Ablations

Ablations use a deterministic fixed subset of the held-out data to limit LALM-judge cost. Select the first 24 held-out catalog-covered items for each applicable task after benchmark-index sorting. No ablation changes the main held-out results.

Run:

1. **Semantic orthogonalization:** orthogonal ParaGeo vs raw unprojected basis.
2. **Rank:** 4, 8, 16, 32.
3. **Layer location:** early `[16,20]`, middle `[24,28]`, late `[32,36]`, all `[16,20,24,28,32,36]`. The same full geometry direction is chunk-selected; the basis is not refit per layer set.
4. **Composition operator:** sum, mean, unit-normalized sum.
5. **Dynamic trajectory:** scheduled trajectory, static start, static endpoint, static midpoint.
6. **Scale sensitivity:** `{0.25,0.5,1.0,1.5}` on development only; never use held-out for scale analysis.
7. **Random direction:** norm-matched random steering for every main task.

## Automatic paper outputs

The experiment pipeline must export:

- `results/icassp2027/summary.json`;
- `paper/generated/main_results.tex`;
- `paper/generated/geometry_table.tex`;
- `paper/generated/ablation_table.tex`;
- `paper/generated/result_macros.tex`.

The LaTeX files contain only numbers generated from result JSON/metadata and are safe to `\input{}` into the paper.

## Stop / paper rule

The ICASSP submission proceeds only if at least one held-out core result is clearly positive while preserving content:

- static or composition preference gain >= +5 points over prompt-only, or
- dynamic preference gain >= +8 points over prompt-only,
- text-channel WER degradation <= 0.02,
- norm-matched random control does not achieve the same gain.

If the held-out results fail this rule, archive the project instead of rescuing it with post-hoc seeds, layers, schedules, RL, or fine-tuning.

## Paper scope

ICASSP allows four content pages plus a references-only fifth page under the supplied `spconf` template. The paper should therefore contain a compact Introduction, Method, Experimental Setup, one main table, one mechanism/ablation table or figure, and Conclusion. Additional ablation material must fit in the four content pages; do not rely on a fifth-page appendix.