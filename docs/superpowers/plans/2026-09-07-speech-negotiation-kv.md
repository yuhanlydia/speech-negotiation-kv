# Speech Negotiation KV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 16GB-first, reproducible CRAD + GLM-4-Voice pipeline that can test opponent-stable advantage-KV subspaces before expensive OPSD/RL training.

**Architecture:** Keep the benchmark/scoring, speech-model adapter, matched-sweep records, KV extraction, and subspace analysis separate. Primary speech transport is direct GLM discrete audio-token loopback; CPU-side analysis never requires the model to be resident. The first runnable target is a one-turn matched counterfactual sweep plus offline KV/subspace analysis; long-horizon rollouts extend the same record format.

**Tech Stack:** Python 3.10+, NumPy, pandas, PyYAML, PyTorch/Transformers/bitsandbytes only for real GLM runs, pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-speech-negotiation-kv-design.md`

## Global Constraints

- CRAD only; 80 train / 20 held-out test.
- One GLM-4-Voice-9B model copy on GPU.
- 16GB default uses NF4/int4 and token-loopback; no waveform decoder/tokenizer.
- Matched counterfactuals fix scenario/state/semantic content and pair decoding seeds.
- KV summaries are detached and offloaded immediately.
- No OPSD/RL scaling until the phenomenon and subspace stop gates pass.

---

### Task 1: Benchmark core and tests

**Files:** `src/speech_negotiation_kv/crad.py`, `tests/test_core.py`, `scripts/download_crad.py`.

**Interfaces:** `normalized_creditor_utility`, `parse_offer_days`, `parse_agreement_days`, `load_crad`, `split_crad`.

- [x] Add scorer/parser/split tests.
- [x] Implement CRAD utilities and fixed 80/20 split.
- [x] Add downloader and split smoke check.

### Task 2: Advantage subspace core

**Files:** `src/speech_negotiation_kv/subspace.py`, `scripts/fit_subspace.py`, `tests/test_core.py`.

**Interfaces:** `advantage_memory_directions(records)`, `fit_low_rank_subspace(G, rank)`, `subspace_overlap(a,b)`.

- [x] Test content-offset cancellation and low-rank recovery.
- [x] Implement advantage-centered directions and SVD estimator.
- [x] Add scenario-half overlap, shuffled-advantage null, raw-best-KV PCA and random rank-matched controls.

### Task 3: Fused-QKV primitives

**Files:** `src/speech_negotiation_kv/kv_hooks.py`, `tests/test_core.py`.

**Interfaces:** `split_fused_qkv`, `apply_kv_delta`, `FusedQKVRecorder`, `FusedQKVSteerer`.

- [x] Test multi-query fused QKV splitting and K/V-only edit.
- [x] Add ChatGLM `query_key_value` module discovery.
- [x] Add selected-layer recorder with fp16 CPU offload.
- [x] Add rank-direction steering context manager for Gate C.

### Task 4: GLM token-loopback backend

**Files:** `src/speech_negotiation_kv/glm_voice.py`, `tests/test_core.py`.

**Interfaces:** `audio_ids_to_prompt`, `partition_generated_token_ids`, `GLMVoiceBackend.generate`, `render_exact`, `respond_audio`.

- [x] Test GLM audio special-token formatting without model weights.
- [x] Implement lazy NF4/int4 loader using `trust_remote_code=True`.
- [x] Implement same-transcript private style rendering and audio-token-only opponent input.

### Task 5: Matched sweep and record format

**Files:** `src/speech_negotiation_kv/records.py`, `src/speech_negotiation_kv/sweep.py`, `scripts/run_sweep.py`, `tests/test_core.py`.

**Interfaces:** JSONL rows keyed by `branch_id`, `state_id`, `scenario_id`, `semantic_id`, `style`, `seed`, `utility`, `transcript`, `audio_token_ids`.

- [x] Add deterministic dry-run backend.
- [x] Enforce same semantic ID and paired seed across style branches.
- [x] Mark transcript mismatches rather than silently pooling them.
- [x] Add real GLM and dry-run CLI.

### Task 6: KV extraction and Gate-B analysis

**Files:** `scripts/extract_kv.py`, `scripts/fit_subspace.py`.

- [x] Build creditor observation prompts from debtor audio tokens.
- [x] Run a single forward pass per matched branch and save compressed NPZ K/V vectors.
- [x] Report explained variance, scenario-half overlap, shuffled null, raw-best-KV control and random-subspace null.

### Task 7: Few-shot opponent calibration

**Files:** `src/speech_negotiation_kv/calibration.py`, `scripts/fit_opponent_code.py`, `tests/test_core.py`.

- [x] Implement and test ridge opponent-code recovery.
- [x] Add CLI for 4-16 short-probe calibration after Gate B.

### Task 8: Run order and next stop gate

- [x] Document 16GB and 24GB configs in README.
- [x] Verify local unit tests and a 120-branch dry-run.
- [x] Verify the Gate-B CLI on a synthetic known low-rank K/V signal.
- [ ] Run the first real 10-scenario GLM Gate-A sweep on a CUDA host.
- [ ] Extract real K/V and run ranks 4/8/16.
- [ ] Stop if Gate A or Gate B fails; only then implement full held-out steering and OPSD/RL.
