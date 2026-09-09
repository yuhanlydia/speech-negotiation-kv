# Gate F/G Short-Horizon Strategy Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a one-step-supervised geometry-aware vocal-strategy selector, plus matched non-geometry and held-out-strategy controls.

**Architecture:** Reuse the existing one-turn matched speech backend to collect Gate-F teacher rows on CRAD 30--59. Fit lightweight NumPy selectors using fixed Gate-D strategy coordinates and deterministic CRAD state features. Evaluate on scenario-held-out validation/robustness splits, then provide a terminal-evaluation runner for frozen methods on CRAD 80--99.

**Tech Stack:** Python 3.10, NumPy, pandas, PyYAML, existing GLM-4-Voice backend, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-short-horizon-strategy-distillation-design.md`

## Global Constraints

- Do not use terminal utility to train the Gate-F selector.
- Gate F train scenarios are 30--59; validation 60--69; robustness 70--79; final test 80--99.
- Keep the existing six vocal styles unchanged.
- Gate-D style coordinates come from action-side audio-only K/V prototypes.
- Selector fitting is CPU-only and GLM weights remain frozen.
- Final test scenarios 80--99 must not be used for hyperparameter choice.
- Raw JSONL/NPZ/audio artifacts remain gitignored.

---

### Task 1: Selector primitives and teacher targets

**Files:**
- Create: `src/speech_negotiation_kv/short_horizon_selector.py`
- Create: `tests/test_short_horizon_selector.py`

**Interfaces:**
- Produces `teacher_distribution`, `state_spread_weights`, `opening_state_features`, `fit_geometry_selector`, `fit_onehot_selector`, `score_geometry_selector`, `score_onehot_selector`, `evaluate_selector_states`, `paired_bootstrap_mean_delta`.

- [ ] **Step 1: Write failing tests** for tie-aware soft teachers, zero-spread weights, geometry scoring, matched one-hot scoring, state-level Top-1/regret, and paired bootstrap deltas.
- [ ] **Step 2: Run focused tests and confirm RED** because `short_horizon_selector` does not exist.
- [ ] **Step 3: Implement minimal NumPy functions** with train-only standardization and ridge-regularized least squares on centered one-step utility advantages.
- [ ] **Step 4: Run focused tests and confirm GREEN.**

### Task 2: One-step Gate-F teacher collection

**Files:**
- Create: `configs/crad_gate_f_16gb.yaml`
- Create: `scripts/run_gate_f_teacher.py`

**Interfaces:**
- Consumes CRAD scenarios 30--79, existing `GLMVoiceBackend`, `run_one_turn_matched_sweep`.
- Produces resumable JSONL rows containing matched style, immediate opponent offer, and immediate utility.

- [ ] **Step 1:** Add CLI arguments `--scenario-start`, `--scenario-end`, `--seeds`, `--output`, `--dry-run`, `--fresh`.
- [ ] **Step 2:** Enforce exact scenario ranges and append/resume keys `(scenario_id, seed, style)`.
- [ ] **Step 3:** Add a dry-run validation path that requires the expected branch count and matched semantics.

### Task 3: Formal Gate-F CPU fit and validation

**Files:**
- Create: `scripts/fit_gate_f_selector.py`

**Interfaces:**
- Consumes train and validation teacher JSONL, Gate-D records + action-audio K/V, CRAD CSV.
- Produces selector NPZ and JSON summary for `style_lookup`, `onehot`, and `geometry` models.

- [ ] **Step 1:** Derive fixed rank-5 Gate-D style coordinates from centered style prototypes.
- [ ] **Step 2:** Build per-row CRAD opening-state features without terminal information.
- [ ] **Step 3:** Fit ridge candidates on train scenarios and select ridge from `[0.01, 0.1, 1, 10, 100]` by validation mean within-state Spearman, breaking ties by Top-1.
- [ ] **Step 4:** Save frozen geometry/one-hot coefficients and train standardization metadata.
- [ ] **Step 5:** Report validation Spearman, tie-aware Top-1, immediate utility, regret, and parameter count.

### Task 4: Gate G1/G2 controls

**Files:**
- Create: `scripts/run_gate_g_controls.py`

**Interfaces:**
- Consumes Gate-F train + robustness one-step rows and frozen Gate-D coordinates.
- Produces G1 matched geometry-vs-onehot results and six leave-one-style-out G2 results.

- [ ] **Step 1:** Refit geometry and one-hot models on identical train rows and evaluate on scenarios 70--79.
- [ ] **Step 2:** For each held-out style, remove it from all training rows, fit the geometry model, and evaluate all six styles on robustness states.
- [ ] **Step 3:** Use the one-hot fallback score equal to the state mean over seen styles for the unseen style.
- [ ] **Step 4:** Report held-out-style correlation, pairwise sign accuracy, all-six Top-1, regret, and parameter counts.

### Task 5: Final terminal method runner

**Files:**
- Create: `scripts/run_gate_f_terminal_eval.py`
- Create: `scripts/analyze_gate_f_terminal.py`

**Interfaces:**
- `run_gate_f_terminal_eval.py` consumes a frozen selector and runs CRAD 80--99 with one selected opening style followed by the existing neutral continuation policy.
- `analyze_gate_f_terminal.py` joins method JSONL files by `(scenario_id, seed)` and reports paired terminal utility deltas with 10,000-bootstrap 95% intervals.

- [ ] **Step 1:** Support methods `geometry`, `neutral`, `random`, and `best_fixed` with deterministic paired seeds.
- [ ] **Step 2:** Preserve existing Gate-E2 H=8/H=12 terminal protocol and timeout provenance.
- [ ] **Step 3:** Analyze terminal utility, agreement/no-deal, rounds, and paired deltas relative to each baseline.
- [ ] **Step 4:** Mark Gate F pass only when geometry beats best-fixed with paired 95% CI lower bound > 0.

### Task 6: Run contract and documentation

**Files:**
- Create: `GATE_F_RUN.md`
- Modify: `README.md`

- [ ] **Step 1:** Document exact train/validation/robustness/final-test commands.
- [ ] **Step 2:** State that E2 supports short-horizon value sufficiency and the next method claim is not yet empirically established.
- [ ] **Step 3:** Keep G4 steering, OPSD, RL, cross-model and cross-domain work explicitly blocked until final held-out terminal Gate F succeeds.

### Task 7: Verification and push

- [ ] **Step 1:** Run focused selector tests.
- [ ] **Step 2:** Run full `pytest -q` in the experiment environment.
- [ ] **Step 3:** Run `python -m py_compile` on all new scripts/modules and `git diff --check`.
- [ ] **Step 4:** Push code, tests, configs, docs, and lightweight metadata only.
