# ParaGeo SpeechParaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ParaGeo pilot that learns content-invariant paralinguistic K/V geometry and evaluates static, compositional, and dynamic steering on SpeechParaling-Bench using the official GLM-4-Voice waveform pipeline.

**Architecture:** Pure NumPy geometry/parsing code is separated from the GPU backend. Calibration first produces matched audio-token actions and K/V tensors, then a CPU fitter writes a semantic-orthogonal basis and attribute coordinates. The benchmark runner consumes the frozen basis and outputs official 22.05-kHz WAV files directly into SpeechParaling-compatible directories.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Transformers 4.44.1, torchaudio, official GLM-4-Voice repository, SpeechParaling-Bench.

**Spec:** `docs/superpowers/specs/2026-09-10-parageo-speechparaling-design.md`

## Global Constraints

- Keep Gate F and Gate F2 negative results unchanged.
- Primary model: `zai-org/glm-4-voice-9b`, int4, one 9B model copy.
- Official output decoder and input Whisper-VQ tokenizer are mandatory for benchmark audio.
- Primary temperature `0.8`, top-p `0.8`.
- Calibration and benchmark raw WAV/JSONL/NPZ stay gitignored.
- Tune steering scale only on the deterministic development slice.
- No RL/LoRA/post-training before the frozen pilot gate passes.

---

### Task 1: ParaGeo math and benchmark parsing

**Files:**
- Create: `src/speech_negotiation_kv/parageo.py`
- Create: `src/speech_negotiation_kv/speechparaling.py`
- Test: `tests/test_parageo.py`
- Test: `tests/test_speechparaling.py`

**Interfaces:**
- Produces content centering, low-rank basis fitting, semantic orthogonalization, composition, dynamic trajectories, prompt parsing, catalog matching, and text-channel WER.

- [x] Write failing tests for content centering, low-rank recovery, composition, dynamic interpolation, benchmark static/dynamic prompt parsing, and WER.
- [x] Run the focused tests and observe expected failures before implementation.
- [x] Implement the minimal pure-NumPy/parser functions.
- [x] Run focused tests to green.

### Task 2: Dynamic K/V steering primitive

**Files:**
- Create: `src/speech_negotiation_kv/parageo_steering.py`
- Test: `tests/test_parageo_steering.py`

**Interfaces:**
- Produces `ScheduledFusedQKVSteerer`, accepting `[generation_steps, concatenated_KV_dimension]` directions.

- [x] Write a failing fake-ChatGLM test proving one schedule step per model forward and saturation after schedule end.
- [x] Implement the scheduled hook using existing `apply_kv_delta`.
- [x] Verify static and changing directions on the fake model.

### Task 3: Official GLM waveform adapter

**Files:**
- Create: `src/speech_negotiation_kv/glm_official_waveform.py`
- Test: `tests/test_glm_official_waveform.py`

**Interfaces:**
- Produces `OfficialVoiceAssets` and `GLMOfficialWaveformBackend`.
- Uses official Whisper-VQ input tokenization and Flow/HiFT output decoding.

- [x] Write tests for official prompt construction and required asset validation.
- [x] Implement lazy official-repo imports and offline decoder/tokenizer wrappers.
- [x] Keep optional dependencies out of import-time execution.

### Task 4: Calibration and basis fitting pipeline

**Files:**
- Create: `scripts/build_parageo_catalog.py`
- Create: `scripts/collect_parageo_calibration.py`
- Create: `scripts/extract_parageo_kv.py`
- Create: `scripts/fit_parageo_basis.py`
- Create: `configs/parageo_speechparaling_pilot.yaml`

**Interfaces:**
- Produces `results/parageo_attribute_catalog.json`, `results/parageo_calibration.jsonl`, `results/parageo_calibration_kv.npz`, `results/parageo_basis.npz`, and a lightweight summary JSON.

- [x] Build reusable attribute catalog from SpeechParaling short-single prompts.
- [x] Cross fixed calibration sentences with all selected reusable attributes using matched lexical rendering.
- [x] Extract audio-only action K/V at frozen layers.
- [x] Fit SVD basis, semantic orthogonalization, coordinates, and cross-content centroid diagnostic.

### Task 5: SpeechParaling generation runner

**Files:**
- Create: `scripts/run_speechparaling_pilot.py`
- Create: `scripts/analyze_parageo_fidelity.py`

**Interfaces:**
- Supports `prompt_only`, `parageo_static`, `parageo_composed`, `parageo_dynamic`, and `random`.
- Writes WAV outputs plus an auditable `parageo_manifest.jsonl`.

- [x] Implement static, composed, dynamic, and random steering selection.
- [x] Decode all outputs with the official Flow/HiFT decoder.
- [x] Record target text, selected coordinates, steering scale, output transcript, and token count.
- [x] Add text-channel WER audit.

### Task 6: Frozen pilot decision and documentation

**Files:**
- Create: `PARAGEO_RUN.md`
- Create: `scripts/analyze_parageo_pilot.py`
- Modify: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- The run contract defines calibration, dev-scale search, held-out generation, official SpeechParaling judge invocation, fidelity check, and the frozen continue/stop decision.

- [x] Encode +5 static/composed or +8 dynamic score threshold.
- [x] Document the deterministic 20% dev split and 80% held-out pilot.
- [x] Make ParaGeo the current repo path while preserving the negotiation discovery and negative Gate-F history.
- [x] Verify focused unit tests and Python compilation before pushing.
