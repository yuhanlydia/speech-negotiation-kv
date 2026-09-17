# Static Attribute Power Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a balanced 180-item, five-random-seed direct ParaGeo-vs-random static evaluation.

**Architecture:** Add a deterministic expanded-dataset builder, explicit random-seed generation support, an expanded experiment orchestrator, and a hierarchical analyzer. Reuse the frozen r2 geometry, official GLM waveform backend, official judge wrapper, and existing resumable manifests.

**Tech Stack:** Python 3.10, NumPy, PyTorch/GLM-4-Voice, edge-tts-compatible neutral TTS, Google Gemini judge API, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-static-power-expansion-design.md`

## Global Constraints

- Exactly 18 attributes × 10 unique lexical targets.
- Static alpha is 0.5; r2 basis, layers, generation seed, and judge rubric remain frozen.
- Random seeds are exactly 15242424242, 16242424242, 17242424242, 18242424242, and 19242424242.
- Primary result is direct ParaGeo-vs-random with hierarchical attribute/utterance bootstrap.
- Persistent failures are reported, never hidden by seed or item substitution after judging.

---

### Task 1: Balanced expanded dataset

**Files:**
- Create: `scripts/build_static_power_dataset.py`
- Create: `tests/test_static_power_dataset.py`
- Create: `configs/icassp2027_static_power.yaml`

**Interfaces:**
- Produces JSONL rows containing `item_id`, `prompt`, `target_text`, `dimension`, `control`, `attribute_key`, and `audio_path`.

- [ ] Write failing tests for 18×10 balance, normalized-text uniqueness, and deterministic output.
- [ ] Run `pytest -q tests/test_static_power_dataset.py` and confirm the missing builder failure.
- [ ] Implement deterministic templates and collision validation.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit the dataset builder and configuration.

### Task 2: Neutral prompt-audio synthesis and audit

**Files:**
- Create: `scripts/synthesize_static_power_prompts.py`
- Create: `tests/test_static_power_audio.py`

**Interfaces:**
- Consumes Task 1 JSONL and produces one 22.05 kHz mono WAV per item plus an audited manifest.

- [ ] Write failing tests for command construction, resumability, and WAV validation.
- [ ] Run the focused tests and confirm failure.
- [ ] Implement the TTS adapter with engine metadata and deterministic filenames.
- [ ] Synthesize and audit all 180 prompts; fail if any item is missing, clipped, empty, or undecodable.
- [ ] Commit code and the small text manifest; do not commit WAV files.

### Task 3: Explicit five-seed random generation

**Files:**
- Modify: `scripts/run_icassp_generation.py`
- Modify: `src/speech_negotiation_kv/icassp_variants.py`
- Create: `tests/test_static_power_random.py`

**Interfaces:**
- Adds `--random-seed` and external dataset inputs while preserving all existing defaults.
- Produces `prompt_only`, `main`, and `random_seed_<seed>` manifests.

- [ ] Write failing tests that different fixed seeds change direction and preserve per-item norm.
- [ ] Run tests and confirm failure for the missing CLI/data path.
- [ ] Implement the minimal seed/data extensions without changing r2 defaults.
- [ ] Run focused and full tests.
- [ ] Commit generation support.

### Task 4: Direct judge matrix

**Files:**
- Create: `scripts/run_static_power_experiment.py`
- Create: `tests/test_static_power_orchestrator.py`

**Interfaces:**
- Emits generation commands for 180 prompt/main and 5×180 random items, then five direct judge commands `main` vs `random_seed_<seed>`.

- [ ] Write a failing matrix test asserting exact variants, seeds, paths, and generation-before-judge ordering.
- [ ] Run the test and confirm failure.
- [ ] Implement dry-run and resumable stage execution.
- [ ] Run dry-run and focused/full tests.
- [ ] Commit the orchestrator.

### Task 5: Hierarchical analysis

**Files:**
- Create: `scripts/summarize_static_power.py`
- Create: `src/speech_negotiation_kv/static_power_eval.py`
- Create: `tests/test_static_power_eval.py`

**Interfaces:**
- Produces `results/icassp2027_static_power/summary.json` with per-seed, per-attribute, macro, hierarchical CI, WER, missingness, and decision fields.

- [ ] Write failing synthetic tests for equal attribute weighting and two-level bootstrap.
- [ ] Run tests and confirm failure.
- [ ] Implement deterministic analysis and failure inventory.
- [ ] Run focused/full tests.
- [ ] Commit the analyzer.

### Task 6: Execute and verify

**Files:**
- Generate: `results/icassp2027_static_power/summary.json`
- Generate: `paper/generated/static_power_results.tex`

**Interfaces:**
- Consumes all prior artifacts and produces final publishable evidence.

- [ ] Build and audit 180 prompt inputs.
- [ ] Run all seven generation variants with resumable manifests.
- [ ] Run five direct judges and secondary fidelity checks.
- [ ] Generate summary and TeX output.
- [ ] Validate counts, all JSON, WAV integrity, `pytest -q`, and `compileall`.
- [ ] Commit results and push `main`.
