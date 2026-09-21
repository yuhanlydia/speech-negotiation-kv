# Reproducing ParaGeo measurements

[Back to ParaGeo](../README.md) · [Results and provenance](PARAGEO_RESULTS.md)

There are three different tasks: **inspect released summaries**, **reanalyze locally saved features**, and **run new synthesis/judging**. Only the first works from the public files alone. The examples below do not claim that GPU or external-judge runs were rerun during repository preparation.

## A. Inspect the public evidence without a model

From the repository root:

```bash
python -m json.tool results/parageo_basis_summary.json
python -m json.tool results/icassp2027/summary.json
python -m json.tool results/icassp2027_static_power/summary.json
```

The [results guide](PARAGEO_RESULTS.md) explains the fields, denominators, and arm statuses. A summary inspection reproduces a displayed number from its saved source; it is not a rerun of the statistical experiment.

## B. Install the numerical package and run its geometry tests

Use a separate Python environment. The project declares Python 3.10 or later; **Python 3.10 is the conservative choice for the pinned waveform extras below**.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_parageo.py
```

This focused test file exercises centering, low-rank fitting, prototype coordinates, cross-content classification, composition, schedules, and projection on synthetic fixtures. It does not download GLM or validate the paper's empirical scores. The broader suite contains torch-dependent tests; install the appropriate extras before running all of it.

For the original experiment implementation, use a separate worktree or clone at [`b71c6cc`](https://github.com/yuhanlydia/speech-negotiation-kv/tree/b71c6cc5a1faf51f3e95245068b4c9c9241f8c0e). Current `main` additionally contains this public-facing documentation.

## C. Refit geometry from existing local artifacts

Required local files:

- `results/parageo_calibration.jsonl`: original matched calibration records and branch IDs;
- `results/parageo_calibration_kv.npz`: recorded K/V with matching IDs and layer metadata;
- the checked-in [experiment configuration](../configs/icassp2027_parageo.yaml).

Write new outputs rather than overwriting the recorded result:

```bash
mkdir -p results/local_reanalysis
python scripts/fit_parageo_basis.py \
  --config configs/icassp2027_parageo.yaml \
  --records results/parageo_calibration.jsonl \
  --kv results/parageo_calibration_kv.npz \
  --output results/local_reanalysis/parageo_basis.npz \
  --summary results/local_reanalysis/parageo_basis_summary.json
```

This uses CPU numerical operations and saved features; it does not regenerate speech. The original basis is fitted on the full calibration pool before the centroid leave-one-content-out loop. Retain that protocol when reproducing the archived statistic. A fold-wise basis refit is a different experiment.

## D. New synthesis and waveform evaluation

### Dependencies and assets

The optional dependencies in [`pyproject.toml`](../pyproject.toml) pin the original stack, including Transformers 4.44.1, Accelerate 0.34.2, and the waveform dependencies. Use a clean environment instead of mixing them into another project's installation:

```bash
python -m pip install -e '.[dev,glm,parageo]'
```

Obtain the external assets under their respective terms:

- [GLM-4-Voice implementation](https://github.com/zai-org/GLM-4-Voice), including required submodules;
- [GLM-4-Voice model](https://huggingface.co/zai-org/glm-4-voice-9b), speech tokenizer, and waveform decoder specified in the configuration;
- [SpeechParaling-Bench](https://github.com/Northern-byte-bit/SpeechParaling-Bench), with the English prompt and audio files required by its download instructions;
- a configured upstream audio-judge service for evaluation.

Set real absolute paths:

```bash
export GLM_VOICE_REPO=/absolute/path/to/GLM-4-Voice
export GLM_VOICE_DECODER=/absolute/path/to/glm-4-voice-decoder
export SPEECHPARALING_ROOT=/absolute/path/to/SpeechParaling-Bench
```

The recorded waveform path is benchmark WAV → speech tokenization → GLM-4-Voice → Flow/HiFT decoding. The [original run contract](../ICASSP2027_RUN.md) contains the legacy asset commands. Follow current upstream asset-access instructions and retain exact revisions in a new run manifest.

### Inspect commands before running anything costly

```bash
python scripts/run_icassp2027_all.py \
  --config configs/icassp2027_parageo.yaml \
  --stage all --dry-run
```

This is a command preview. Actual generation needs local model assets, suitable GPU memory, the basis, and the requested dataset. Actual judging invokes the configured external service and may incur costs.

### Important: do not blindly execute the archived all-stage command

The completed experiment selected **no composition alpha**. The frozen orchestrator's selected-alpha loader expects numeric selections for all tasks and does not handle the saved `composed: null` as a successful full-run path. The completed dynamic 0.25 arm also differs from its incomplete selected 1.5 arm. Preserving those facts is part of reproducing the experiment; do not silently fill missing selections or report a new arm as the archived one.

For a new, explicitly scoped static comparison, the existing per-task runner can be invoked directly after assets and geometry are present:

```bash
python scripts/run_icassp_generation.py \
  --config configs/icassp2027_parageo.yaml \
  --task static --split heldout --variant prompt_only \
  --output-dir results/new_static_run/prompt_only --fresh

python scripts/run_icassp_generation.py \
  --config configs/icassp2027_parageo.yaml \
  --task static --split heldout --variant main --alpha 0.5 \
  --output-dir results/new_static_run/main --fresh
```

Then judge the matched WAV sets:

```bash
python scripts/run_official_speechparaling_judge.py \
  --benchmark-root "$SPEECHPARALING_ROOT" --task static \
  --candidate-dir results/new_static_run/main \
  --baseline-dir results/new_static_run/prompt_only \
  --candidate-name main --baseline-name prompt_only \
  --output-dir results/new_static_run/judge
```

These are new-run examples, not a guarantee of bitwise regeneration. Keep output directories, model revisions, upstream benchmark revision, judge model/service identity, command, and environment together. Do not commit API keys or credential-bearing configurations. Random-control resumption also requires consistent item traversal, not just the same RNG seed.

## Artifact and measurement notes

### Included in the public repository

Source modules, experiment configurations, tests, the control catalog, summary JSON files, and historical reports are tracked. Earlier manuscript sources and generated tables remain under `paper/`; see its [status note](../paper/README.md).

### External or not bundled

Model/tokenizer/decoder weights and benchmark assets come from upstream providers. Original local K/V arrays, synthesized WAVs, complete calibration/generation JSONL manifests, the full stored spectrum, and exact historical judge-service metadata are not supplied by the public summary release. The [.gitignore](../.gitignore) excludes these artifact classes by default. Aggregate results cannot reconstruct them.

### What each measurement means

- **Label:** exact requested control phrase; not independently annotated realized delivery.
- **Feature:** audio-token-pooled K/V from a common replay pass; not cached synthesis-time K/V.
- **Projection:** shared rank-16 basis with a sentence-mean content proxy; not proof of complete semantic erasure.
- **Probe:** held-out label centroids under a globally fitted calibration transform.
- **Fidelity:** WER of the emitted text channel, not independent ASR of the WAV.
- **Layer ablation:** selected chunks of one calibrated direction; removed energy is not redistributed.
- **Random control:** a Gaussian orientation matched to the selected vector's global norm; dynamic norms follow the target schedule with fixed orientation per item.
- **Uncertainty:** item-level intervals and conditional permutation statistics, with distinct units for repeated controls and overlapping scenario splits.

## Source map

| Step | File |
|---|---|
| Fixed calibration sentences | [`configs/parageo_speechparaling_pilot.yaml`](../configs/parageo_speechparaling_pilot.yaml) |
| Synthesis and retention | [`scripts/collect_parageo_calibration.py`](../scripts/collect_parageo_calibration.py) |
| Audio replay and extraction | [`scripts/extract_parageo_kv.py`](../scripts/extract_parageo_kv.py) |
| Hook positions and pooling | [`src/speech_negotiation_kv/kv_hooks.py`](../src/speech_negotiation_kv/kv_hooks.py) |
| SVD, prototypes, and diagnostics | [`src/speech_negotiation_kv/parageo.py`](../src/speech_negotiation_kv/parageo.py) |
| Basis artifact construction | [`scripts/fit_parageo_basis.py`](../scripts/fit_parageo_basis.py) |
| Configuration variants | [`src/speech_negotiation_kv/icassp_variants.py`](../src/speech_negotiation_kv/icassp_variants.py) |
| Generation | [`scripts/run_icassp_generation.py`](../scripts/run_icassp_generation.py) |
| Development selection | [`scripts/select_icassp_alpha.py`](../scripts/select_icassp_alpha.py) |
| Upstream judge wrapper | [`scripts/run_official_speechparaling_judge.py`](../scripts/run_official_speechparaling_judge.py) |
| Aggregation | [`scripts/summarize_icassp2027.py`](../scripts/summarize_icassp2027.py) |

## Verification status of the documentation update

This update changes presentation, citation metadata, and navigation, not experiment code or results. Its JSON-field example is checked against the recorded geometry summary. It does not claim a new CUDA installation, a full test-suite run, external-judge validation, or reproduction from unreleased raw artifacts.
