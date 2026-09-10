# ParaGeo × SpeechParaling-Bench Run Contract

This is the current main research path. The CRAD negotiation experiments are retained as the discovery history; Gate F and Gate F2 remain negative and should not be rerun to search for favorable seeds.

## 0. External repositories and model assets

Clone the official repos outside this repository:

```bash
git clone --recurse-submodules https://github.com/zai-org/GLM-4-Voice.git
# download the official decoder checkpoint as instructed by GLM-4-Voice
git clone https://huggingface.co/THUDM/glm-4-voice-decoder

git clone https://github.com/Northern-byte-bit/SpeechParaling-Bench.git
cd SpeechParaling-Bench
python script/download_data.py
python script/download_baseline.py
cd ..
```

Set:

```bash
export GLM_VOICE_REPO=/absolute/path/to/GLM-4-Voice
export GLM_VOICE_DECODER=/absolute/path/to/glm-4-voice-decoder
export SPEECHPARALING_ROOT=/absolute/path/to/SpeechParaling-Bench
```

The official GLM repo requires its own dependencies/submodules. Install them in the same environment or use the official GLM docker image.

## 1. Repo setup and verification

```bash
git pull origin main
source .venv/bin/activate
pip install -e '.[dev,glm,parageo]'
pytest -q
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The official output path is now mandatory: benchmark input WAV -> Whisper-VQ tokenizer -> GLM-4-Voice-9B -> official Flow/HiFT decoder -> output WAV.

## 2. Build reusable SpeechParaling attribute catalog

```bash
python scripts/build_parageo_catalog.py \
  --benchmark-root "$SPEECHPARALING_ROOT" \
  --language en \
  --dimensions Pitch Timbre Pace Volume Rhythm Age \
  --output results/parageo_attribute_catalog.json
```

The catalog is derived from public `para_con/short_sin.jsonl`; no benchmark labels are invented.

## 3. Collect matched-content calibration actions

```bash
python scripts/collect_parageo_calibration.py \
  --config configs/parageo_speechparaling_pilot.yaml \
  --catalog results/parageo_attribute_catalog.json \
  --fresh \
  --output results/parageo_calibration.jsonl
```

Every reusable control is rendered on the same eight lexical calibration sentences. Unmatched transcripts are retained for audit but excluded from basis fitting.

## 4. Extract action-side K/V and fit ParaGeo

```bash
python scripts/extract_parageo_kv.py \
  --config configs/parageo_speechparaling_pilot.yaml \
  --records results/parageo_calibration.jsonl \
  --output results/parageo_calibration_kv.npz

python scripts/fit_parageo_basis.py \
  --config configs/parageo_speechparaling_pilot.yaml \
  --records results/parageo_calibration.jsonl \
  --kv results/parageo_calibration_kv.npz \
  --output results/parageo_basis.npz \
  --summary results/parageo_basis_summary.json
```

Before benchmark steering, inspect `cross_content_centroid_accuracy`. If it is near chance, stop: the original Gate-D phenomenon did not transfer to the richer attribute catalog.

## 5. Pilot subsets

Use English only for the first gate.

### Static single

Prompts:

```text
$SPEECHPARALING_ROOT/jsonl_prompt_en/para_con/short_sin.jsonl
```

Audio:

```text
$SPEECHPARALING_ROOT/audio_dataset_en/para_con/con_short_sin/
```

Generate 80 eligible items.

### Compositional

Use `short_multi.jsonl` + `con_short_multi/`, 40 items whose requested controls exist in the frozen catalog.

### Dynamic

Use `dyn_var/dyn_var.jsonl` + `audio_dataset_en/dyn_var/`, 60 items with catalog-covered endpoints.

## 6. Scale selection: dev only

The development split is deterministic: catalog-covered item index `% 5 == 0`. Tune only `alpha in {0.25, 0.5, 1.0, 1.5}` on that 20% slice. Freeze one scale per task family before generating the remaining 80%.

The runner enforces this directly with `--split dev` and `--split heldout`. Do not create an alternative split and do not inspect held-out judge scores when selecting alpha. After judging the four dev scales, freeze alpha with:

```bash
python scripts/select_parageo_scale.py \
  --score 0.25=DEV_ALPHA_025_SCORE.json \
  --score 0.5=DEV_ALPHA_05_SCORE.json \
  --score 1.0=DEV_ALPHA_10_SCORE.json \
  --score 1.5=DEV_ALPHA_15_SCORE.json \
  --output results/parageo_static_scale.json
```

Exact score ties select the smallest alpha.

## 7. Example generation commands

Prompt-only static baseline:

```bash
python scripts/run_speechparaling_pilot.py \
  --config configs/parageo_speechparaling_pilot.yaml \
  --prompt-jsonl "$SPEECHPARALING_ROOT/jsonl_prompt_en/para_con/short_sin.jsonl" \
  --audio-dir "$SPEECHPARALING_ROOT/audio_dataset_en/para_con/con_short_sin" \
  --task static --method prompt_only --limit 80 --split heldout \
  --output-dir results/parageo_outputs/prompt_only/output_en/para_con/con_short_sin
```

Static ParaGeo:

```bash
python scripts/run_speechparaling_pilot.py \
  --config configs/parageo_speechparaling_pilot.yaml \
  --prompt-jsonl "$SPEECHPARALING_ROOT/jsonl_prompt_en/para_con/short_sin.jsonl" \
  --audio-dir "$SPEECHPARALING_ROOT/audio_dataset_en/para_con/con_short_sin" \
  --task static --method parageo_static --limit 80 --split heldout --scale FROZEN_STATIC_ALPHA \
  --output-dir results/parageo_outputs/parageo/output_en/para_con/con_short_sin
```

Compositional ParaGeo:

```bash
python scripts/run_speechparaling_pilot.py \
  --config configs/parageo_speechparaling_pilot.yaml \
  --prompt-jsonl "$SPEECHPARALING_ROOT/jsonl_prompt_en/para_con/short_multi.jsonl" \
  --audio-dir "$SPEECHPARALING_ROOT/audio_dataset_en/para_con/con_short_multi" \
  --task composed --method parageo_composed --limit 40 --split heldout --scale FROZEN_COMPOSED_ALPHA \
  --output-dir results/parageo_outputs/parageo/output_en/para_con/con_short_multi
```

Dynamic ParaGeo:

```bash
python scripts/run_speechparaling_pilot.py \
  --config configs/parageo_speechparaling_pilot.yaml \
  --prompt-jsonl "$SPEECHPARALING_ROOT/jsonl_prompt_en/dyn_var/dyn_var.jsonl" \
  --audio-dir "$SPEECHPARALING_ROOT/audio_dataset_en/dyn_var" \
  --task dynamic --method parageo_dynamic --limit 60 --split heldout --scale FROZEN_DYNAMIC_ALPHA \
  --output-dir results/parageo_outputs/parageo/output_en/dyn_var
```

Run the matched `random` method on every reported subset with the same frozen scale.

## 8. Semantic fidelity audit

Each output directory contains `parageo_manifest.jsonl`.

```bash
python scripts/analyze_parageo_fidelity.py \
  --manifest results/parageo_outputs/parageo/output_en/dyn_var/parageo_manifest.jsonl \
  --output results/parageo_dynamic_fidelity.json
```

Reject steering if held-out mean text-channel WER worsens by more than 0.02 absolute versus prompt-only.

## 9. Official SpeechParaling judge

Copy/symlink generated WAV folders into the benchmark's expected `api_models/MODEL_NAME/output_en/...` structure, configure the benchmark judge exactly as documented upstream, then run:

```bash
cd "$SPEECHPARALING_ROOT"
python judge_data/run_all_evaluations.py
python judge_data/run_all_calculate.py
```

Do not edit the judge prompts or score calculator.

## 10. Frozen pilot decision

After obtaining held-out static/compositional and dynamic scores, run:

```bash
python scripts/analyze_parageo_pilot.py \
  --baseline-static STATIC_PROMPT_SCORE.json \
  --ours-static STATIC_PARAGEO_SCORE.json \
  --random-static STATIC_RANDOM_SCORE.json \
  --baseline-static-fidelity results/static_prompt_fidelity.json \
  --ours-static-fidelity results/static_parageo_fidelity.json \
  --baseline-composed COMPOSED_PROMPT_SCORE.json \
  --ours-composed COMPOSED_PARAGEO_SCORE.json \
  --random-composed COMPOSED_RANDOM_SCORE.json \
  --baseline-composed-fidelity results/composed_prompt_fidelity.json \
  --ours-composed-fidelity results/composed_parageo_fidelity.json \
  --baseline-dynamic DYNAMIC_PROMPT_SCORE.json \
  --ours-dynamic DYNAMIC_PARAGEO_SCORE.json \
  --random-dynamic DYNAMIC_RANDOM_SCORE.json \
  --baseline-dynamic-fidelity results/dynamic_prompt_fidelity.json \
  --ours-dynamic-fidelity results/dynamic_parageo_fidelity.json \
  --static-min-gain 5 --dynamic-min-gain 8 --max-wer-degradation 0.02 \
  --output results/parageo_pilot_decision.json
```

The analyzer only passes a task when the score threshold is met, mean text-channel WER degrades by <=0.02 absolute, and the matched random-direction control does not itself meet the same gain threshold. Continue if at least one held-out task passes.

If the gate fails, stop this repo. Do not add RL, LoRA, new seeds, or post-hoc schedules.

## 11. Expansion after a pass only

1. Full English 1001.
2. Full Chinese 1001.
3. Situational Adaptation router: context -> ParaGeo coordinate.
4. Qwen-Omni replication.
5. Cross-model geometry comparison.
6. External S2S-Arena / WildSpeech-Bench generalization.
