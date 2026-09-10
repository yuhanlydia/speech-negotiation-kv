# Gate F2 Immediate-Search Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a preregistered six-probe immediate-search policy against the frozen best-fixed baseline on new Gate-F2 seeds.

**Architecture:** Add a reusable opening-probe selector to the long-horizon protocol, then extend the terminal runner to replay and continue only the selected probe. Keep Gate F2 analysis separate from the original Gate F analyzer so the old negative result and threshold remain immutable.

**Tech Stack:** Python 3.11, NumPy, PyYAML, pytest, GLM-4-Voice-9B NF4/int4.

**Spec:** `docs/superpowers/specs/2026-09-09-gate-f2-immediate-search-design.md`

## Global Constraints

- Preserve the original Gate F negative result and files.
- Use CRAD scenarios 80--99 and new seeds 40--45.
- Keep the six registered styles and their existing order for deterministic ties.
- Run one model copy with microbatch 1 and token loopback.
- Gate F2 passes only if the paired bootstrap 95% CI lower bound for immediate-search minus best-fixed terminal utility is strictly greater than zero.
- Never change seeds, tie-breaking, metric, or threshold after observing Gate-F2 terminal outcomes.

---

### Task 1: Opening probe selection and deterministic replay

**Files:**
- Modify: `src/speech_negotiation_kv/long_horizon.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: `probe_opening_styles(scenario, scenario_id, styles, branch_seed, backend, base_seed) -> (selected_style, probes, fallback_used)`.
- Extends: `run_long_horizon_branch(..., expected_opening_probe=None)` with exact replay verification.

- [x] **Step 1: Write failing tests** proving that the mock backend selects the highest immediate-utility eligible style, preserves fixed-order tie-breaking, falls back to neutral when all candidates are invalid, and rejects a replay mismatch.

- [x] **Step 2: Run the focused tests** with `pytest tests/test_core.py -k 'probe or replay' -q` and verify failure because the new interface does not exist.

- [x] **Step 3: Implement the minimal probe record, eligibility logic, deterministic selector, and optional replay assertion** in `long_horizon.py`.

- [x] **Step 4: Run focused and full tests** with `pytest tests/test_core.py -k 'probe or replay' -q` and `pytest -q`.

### Task 2: Gate-F2 terminal runner and frozen configuration

**Files:**
- Modify: `scripts/run_gate_f_terminal_eval.py`
- Create: `configs/crad_gate_f2_16gb.yaml`
- Test: `tests/test_gate_f2.py`

**Interfaces:**
- Adds terminal method `immediate_search`.
- Writes per-state probe summaries, selected style, fallback provenance, and `opening_probe_replay_verified`.

- [x] **Step 1: Write a failing CLI dry-run test** for two states that requires `immediate_search`, six probe records per state, no fallback, and verified replay.

- [x] **Step 2: Run `pytest tests/test_gate_f2.py -q`** and verify that the unsupported method fails.

- [x] **Step 3: Extend the runner** to probe six styles, select one, replay it exactly, and persist auditable probe metadata. Add the frozen seeds 40--45 config with a distinct evaluation base seed.

- [x] **Step 4: Run the focused test and a CLI dry-run** for scenarios 80--81 and seeds 40--41, then run the full test suite.

### Task 3: Independent Gate-F2 analyzer

**Files:**
- Create: `scripts/analyze_gate_f2_terminal.py`
- Test: `tests/test_gate_f2.py`

**Interfaces:**
- Consumes: `immediate_search=PATH` and `best_fixed=PATH` terminal JSONL files.
- Produces: coverage, method summaries, probe quality, paired bootstrap delta, and `gate_f2_passes`.

- [x] **Step 1: Write failing analyzer tests** for a complete positive fixture, incomplete coverage, and replay-invalid records.

- [x] **Step 2: Run the focused tests** and verify failure because the analyzer does not exist.

- [x] **Step 3: Implement the analyzer** with the exact 120-state coverage and lower-CI-greater-than-zero rule from the spec.

- [x] **Step 4: Run focused and full tests**, compile changed Python files, and run `git diff --check`.

### Task 4: Freeze implementation and run Gate F2

**Files:**
- Modify: `README.md`
- Create after results: `docs/results/gate_f2_terminal_report.md`
- Create after results: `results/gate_f2_terminal_summary.json`

**Interfaces:**
- Produces local raw files `results/gate_f2_immediate_search.jsonl` and `results/gate_f2_best_fixed.jsonl`.

- [ ] **Step 1: Commit and push code/config/tests before GPU evaluation** so implementation and seeds are frozen independently of outcomes.

- [ ] **Step 2: Run immediate-search on GPU** for scenarios 80--99 and seeds 40--45, resuming the same file after interruptions.

- [ ] **Step 3: Run best-fixed on the identical paired states**, then analyze with 10,000 bootstrap repeats.

- [ ] **Step 4: Diagnose and repair only implementation/data-validity failures**, rerunning affected records without changing the scientific rule. If valid coverage produces a negative result, report it as negative.

- [ ] **Step 5: Write the result report, rerun verification, commit lightweight summaries/docs, and push main.**
