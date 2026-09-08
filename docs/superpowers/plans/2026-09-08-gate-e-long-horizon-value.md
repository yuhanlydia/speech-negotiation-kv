# Gate E Long-Horizon Value Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the first-turn influence of a matched vocal intervention predicts a paired multi-turn or explicit terminal negotiation outcome.

**Architecture:** Run 20 CRAD scenarios × 6 styles × 6 paired seeds with one styled creditor opening followed by a shared neutral policy. Record four debtor-response transitions for every branch, preserve unresolved trajectories as censored, and continue a preregistered five-scenario subset to at most eight transitions for explicit agreement/no-deal outcomes.

**Tech Stack:** Python 3.10, GLM-4-Voice-9B NF4/int4, direct audio-token loopback, JSONL resumable artifacts, NumPy/pandas analysis, pytest.

**Spec:** User-provided Experiment-E “Immediate Influence–Long-Horizon Value Gap” requirements in the current task message.

## Global Constraints

- Dataset is CRAD train scenarios 0–19.
- Styles are the existing six matched vocal interventions.
- Seeds are `[0, 1, 2, 3, 4, 5]`, paired across all styles in a scenario.
- Total styled branches are exactly 720.
- H=4 means four debtor-response transitions, including the response to the styled opening.
- After the styled opening, every creditor action uses the same neutral/base policy prompt.
- The debtor policy prompt and per-turn random seed are identical across style branches.
- Explicit agreement or no-deal is terminal; unresolved H=4 branches are censored and receive no fabricated terminal utility.
- Scenarios `[0, 4, 8, 12, 16]` are selected before outcomes and continue to at most H=8.
- Agreement utility uses the existing normalized CRAD target-day scorer; no-deal utility is 0.
- Raw trajectories are gitignored and written incrementally for resume safety.
- Gate F is not authorized unless terminal coverage is adequate for a held-out contextual-value comparison.

---

### Task 1: Structured negotiation moves and terminal scoring

**Files:**
- Create: `src/speech_negotiation_kv/long_horizon.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Produces: `NegotiationMove`, `parse_negotiation_move`, `paired_turn_seed`, and `score_terminal_outcome`.

- [ ] **Step 1: Write failing parser/scoring tests**

```python
assert parse_negotiation_move("AGREED: 45 days", latest_offer=45).kind == "agreement"
assert parse_negotiation_move("NO DEAL", latest_offer=45).kind == "no_deal"
assert parse_negotiation_move("PROPOSE: 60 days", latest_offer=45).days == 60
assert score_terminal_outcome("agreement", 45, 30, 120) == 5 / 6
```

- [ ] **Step 2: Verify RED**

Run `.venv/bin/pytest -q tests/test_core.py -k negotiation_move` and confirm missing implementation.

- [ ] **Step 3: Implement exact marker parsing with conservative fallbacks**

Accept only explicit agreement/no-deal language as terminal. A day-bearing nonterminal response is a proposal; an unparseable response is invalid and remains nonterminal.

- [ ] **Step 4: Verify GREEN**

Run focused and full tests.

### Task 2: Neutral multi-turn policy and resumable rollout

**Files:**
- Modify: `src/speech_negotiation_kv/glm_voice.py`
- Modify: `src/speech_negotiation_kv/long_horizon.py`
- Create: `scripts/run_long_horizon.py`
- Create: `configs/crad_gate_e_16gb.yaml`
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: CRAD rows, style bank, paired seeds, speech backend.
- Produces: one JSON-serializable trajectory per branch with turn-level transcripts/audio IDs, immediate offer, terminal/censoring status, rounds, and utility.

- [ ] **Step 1: Write failing mock-rollout tests**

Test that only the opening receives a non-neutral style, all later actions use neutral prompts, paired turn seeds are style-independent, early agreement terminates, and H=4 unresolved output is censored.

- [ ] **Step 2: Implement role-conditioned neutral turn prompts**

Require each response to begin with `AGREED: N days`, `NO DEAL`, or `PROPOSE: N days`; include compact transcript history and latest opponent audio.

- [ ] **Step 3: Implement incremental JSONL resume**

Derive branch IDs from scenario/seed/style/protocol version, skip completed IDs, flush and fsync after each branch, and never duplicate rows on resume.

- [ ] **Step 4: Run the 720-branch mock dry-run**

Require exact branch count, complete paired groups, and valid outcome fields.

### Task 3: Gate-E analysis

**Files:**
- Create: `scripts/analyze_long_horizon.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: trajectory JSONL.
- Produces: summary JSON with parseability, terminal/censoring rates, next-offer versus terminal ranking correlations, style effects, and paired bootstrap confidence intervals.

- [ ] **Step 1: Write failing synthetic analysis tests**

Construct trajectories where next-offer ranking reverses at terminal and require negative within-state Spearman correlation.

- [ ] **Step 2: Implement paired state-level statistics**

Report terminal coverage before testing the gap. Never impute censored utility as zero. Compare style rankings only in states with all six terminal outcomes; report partial-state sensitivity separately.

- [ ] **Step 3: Verify synthetic and full tests**

Run pytest and CLI help checks.

### Task 4: Real Gate-E run and decision

**Files:**
- Create: `results/gate_e_long_horizon_report.md`
- Create: `results/gate_e_summary.json`
- Modify: `README.md`
- Modify: `AGENT_RUN.md`

- [ ] **Step 1: Run a real two-scenario smoke**

Audit role adherence, markers, matched opening transcripts, GPU memory, and resume behavior before the formal run.

- [ ] **Step 2: Run the formal 720-branch H=4 sweep**

Use one model copy, microbatch 1, and incremental output.

- [ ] **Step 3: Continue the preregistered five-scenario subset to H=8**

Only unresolved selected branches continue; preserve all H=4 states.

- [ ] **Step 4: Analyze and record the result**

Classify Gate E as valid, inconclusive due to terminal/parse coverage, or negative/positive for immediate-to-long-horizon concordance. Do not start Gate F unless explicitly authorized by the preregistered coverage rule.

### Task 5: Verify, commit, and push

**Files:**
- Modify: `docs/superpowers/plans/2026-09-08-gate-e-long-horizon-value.md`

- [ ] **Step 1: Run full verification**

Run pytest, py_compile, dry-run artifact assertions, real-summary assertions, and `git diff --check`.

- [ ] **Step 2: Commit and push lightweight evidence**

Push code, config, report, and summary only; keep raw trajectory/audio artifacts local.
