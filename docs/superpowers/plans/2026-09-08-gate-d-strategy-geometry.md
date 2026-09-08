# Gate D Strategy Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether action-side vocal-strategy representations transfer across CRAD scenarios independently of utility.

**Architecture:** Convert each action K/V feature into a within-state centered residual, then evaluate cross-scenario style decoding over all 126 balanced 5/5 splits with a within-state shuffled-style null. Independently measure all 15 oriented pairwise style directions across scenarios and control their permutation p-values with Benjamini-Hochberg FDR.

**Tech Stack:** Python 3.10, NumPy, pytest, existing action-side K/V NPZ artifacts.

**Spec:** User-provided “Shared Strategy Space + Context-Dependent Value” and Experiment-D requirements in the current task message.

## Global Constraints

- Gate D does not use utility labels.
- Primary representation is action-side audio-only all-layer K/V; last-audio and layerwise K/V are controls.
- Use all 126 unique balanced 5/5 scenario splits.
- Chance accuracy is exactly `1/6` for six balanced styles.
- Shuffle style labels independently within each matched state.
- Do not authorize Gate E unless primary D1 is significant and D2 has FDR-controlled supporting evidence.
- Do not reinterpret style geometry as advantage geometry.

---

### Task 1: State-centered style geometry primitives

**Files:**
- Create: `src/speech_negotiation_kv/strategy_geometry.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: feature matrix `X`, state/scenario/style arrays.
- Produces: `center_within_states`, `cosine_centroid_accuracy`, `shuffle_labels_within_states`, `pairwise_style_directions`, and `benjamini_hochberg`.

- [x] **Step 1: Write failing synthetic tests**

```python
def test_state_centered_style_decoder_transfers_across_scenarios():
    centered = center_within_states(features, state_ids)
    assert cosine_centroid_accuracy(centered, styles, scenarios, {0, 1}) == 1.0

def test_pairwise_style_direction_is_stable_after_state_centering():
    result = pairwise_style_directions(centered, styles, scenarios, state_ids)
    assert result[("a", "b")]["cross_scenario_mean_cosine"] > 0.99
```

- [x] **Step 2: Run the focused tests and confirm missing-import failures**

Run: `.venv/bin/pytest -q tests/test_core.py -k 'style_geometry or pairwise_style'`

- [x] **Step 3: Implement the minimal NumPy primitives**

Use state-wise means, row L2 normalization, train-only style centroids, independent within-state permutations, oriented style differences, and monotone BH adjusted p-values.

- [x] **Step 4: Run focused and full tests**

Run: `.venv/bin/pytest -q tests/test_core.py`.

### Task 2: Exhaustive Gate-D runner

**Files:**
- Create: `scripts/run_strategy_geometry.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: Gate-A JSONL and 4D action-side K/V NPZ.
- Produces: JSON summaries for all-layer and per-layer K, V, and K/V configurations.

- [x] **Step 1: Add failing tests for 126-split decoding and within-state null preservation**

```python
assert len(balanced_scenario_splits(range(10))) == 126
assert sorted(shuffled[state]) == sorted(original[state])
```

- [x] **Step 2: Implement the CLI**

The CLI records held-out accuracy distributions, permutation p-values, splitwise null thresholds, confusion matrices, pairwise cosine statistics, raw/BH-adjusted p-values, observation/pooling metadata, and an explicit gate decision.

- [x] **Step 3: Verify CLI help and synthetic behavior**

Run: `.venv/bin/python scripts/run_strategy_geometry.py --help` and the full test suite.

### Task 3: Run Gate D and document the decision

**Files:**
- Create: `results/gate_d_strategy_geometry_report.md`
- Create: `results/gate_d_action_audio_summary.json`
- Create: `results/gate_d_action_last_audio_summary.json`
- Modify: `README.md`
- Modify: `AGENT_RUN.md`

**Interfaces:**
- Consumes: local gitignored action-side K/V artifacts.
- Produces: lightweight reviewable evidence and the D→E stop/continue decision.

- [x] **Step 1: Run the primary audio-only analysis**

Use 1,000 style-label permutations for all-layer K/V and 200 for layerwise controls.

- [x] **Step 2: Run the last-audio control**

Use the same split and null definitions.

- [x] **Step 3: Write the scientific report**

Separate stable style representation evidence from any utility/value claim and state whether Gate E is authorized.

- [x] **Step 4: Update run instructions**

Document Gate D before any long-horizon Gate E run.

### Task 4: Verify, commit, and push

**Files:**
- Modify: `docs/superpowers/plans/2026-09-08-gate-d-strategy-geometry.md`

- [x] **Step 1: Run verification**

Run `.venv/bin/pytest -q`, `python -m py_compile`, JSON schema assertions, and `git diff --check`.

- [ ] **Step 2: Commit and push lightweight results**

Keep raw JSONL/NPZ local; push code, tests, reports, and summary JSON only.
