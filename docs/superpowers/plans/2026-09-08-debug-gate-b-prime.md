# Debug Gate B' Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the Gate-B measurement confounds and run a reproducible Debug Gate B′ using action-side audio KV features, auditable pooling, seed diagnostics, and all 126 balanced scenario splits.

**Architecture:** Extend the KV recorder to expose token-selective pooling and record the creditor action stream rather than the post-response stream. Extend subspace analysis with actual-rank reporting, within-scenario seed diagnostics, and exhaustive 5/5 scenario split evaluation. Keep old commands/results backward compatible and write a separate B′ report.

**Tech Stack:** Python, PyTorch, NumPy, pytest, existing GLM-4-Voice NF4 runtime and local Gate-A JSONL artifacts.

**Spec:** User-provided Debug Gate B′ requirements in the current task message.

## Global Constraints

- Do not alter the target utility, shuffled null, or scientific thresholds.
- Do not relabel the old post-response Gate-B result; report it as a measurement limitation.
- Use existing Gate-A records and local model cache; no new GLM sweep is required for B′.
- Preserve raw artifacts as gitignored; commit only code, tests, summaries, and the experiment report.

### Task 1: Add auditable action-side and token-selective KV extraction

**Files:**
- Modify: `src/speech_negotiation_kv/kv_hooks.py`
- Modify: `scripts/extract_kv.py`
- Test: `tests/test_core.py`

- [x] Add recorder pooling modes for all tokens, audio-only tokens, and last-audio token positions.
- [x] Add action/response observation selection in extraction; action uses `audio_token_ids`, response retains `opponent_audio_token_ids`.
- [x] Emit layer metadata and pooling/observation metadata in NPZ.
- [x] Add tests for token-selective pooling and extraction prompt selection helpers.

### Task 2: Add robust B′ statistical diagnostics

**Files:**
- Modify: `src/speech_negotiation_kv/subspace.py`
- Modify: `scripts/fit_subspace.py`
- Test: `tests/test_core.py`

- [x] Enumerate every unique balanced 5/5 scenario split.
- [x] Compute same-scenario seed utility Spearman correlation and direction cosine where pairs are identifiable.
- [x] Compute observed and shuffled overlap distributions over all splits.
- [x] Record requested rank and actual effective ranks after sample-rank truncation.
- [x] Add layer/K/V/pooling selection and action-side metadata to summary output.

### Task 3: Run and record Debug Gate B′

**Files:**
- Create: `results/debug_gate_b_prime_report.md`
- Create: `results/debug_gate_b_prime_summary.json`

- [x] Run tests and static checks.
- [x] Extract action-side audio-only and last-audio K/V using existing Gate-A records.
- [x] Run layerwise K, V, and concatenated K/V analyses plus same-seed and exhaustive split diagnostics.
- [x] Write an evidence-based interpretation without declaring the KV direction valid unless the diagnostics support it.

### Task 4: Verify, commit, and push

**Files:**
- Modify: `docs/superpowers/plans/2026-09-08-debug-gate-b-prime.md`

- [x] Run the full test suite, inspect generated summaries, and run `git diff --check`.
- [ ] Commit lightweight artifacts and push the verified branch to `origin/main`.
