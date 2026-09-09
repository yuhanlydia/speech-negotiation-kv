# Formal Gate E v7 report

Date: 2026-09-09

Local raw records: `results/gate_e_formal_v7.jsonl` (720 branches, gitignored)

## Protocol

The run used the fixed GLM-4-Voice-9B int4 configuration, 20 train scenarios,
6 paired seeds, and 6 opening voice styles. Every branch ran at least H=4;
scenarios `[0, 4, 8, 12, 16]` formed the 180-branch H=8 terminal subset.

## Data-quality gate

| metric | result | threshold | status |
|---|---:|---:|---|
| branches | 720/720 | 720 | pass |
| matched opening rate | 0.9958 | >= 0.80 | pass |
| parseable move rate | 0.9982 | >= 0.90 | pass |
| strategically valid move rate | 0.9799 | >= 0.90 | pass |
| H=4 terminal completion | 680/720 = 0.9444 | diagnostic | — |
| H=8 terminal-subset completion | 173/180 = 0.9611 | >= 0.80 | pass |
| complete terminal states | 20/30 = 0.6667 | >= 0.80 (24/30) | **fail** |

Terminal-subset outcomes were 173 agreements and 7 censored branches. The six
incomplete `(scenario, seed)` states were:

- `scenario 12, seed 2`: neutral and hesitant censored;
- `scenario 12, seed 5`: hesitant censored;
- `scenario 16, seed 1`: firm/assertive censored;
- `scenario 16, seed 5`: hesitant censored;
- `scenario 8, seed 1`: firm/assertive censored;
- `scenario 8, seed 3`: empathetic/warm censored.

## Hypothesis statistics

The analyzer reports terminal per-state Spearman mean `0.6933`, median
`0.7746`, and 10,000-bootstrap 95% CI `[0.5476, 0.8286]`. The best-style
agreement rate is `0.95`. These values are descriptive only: because the
complete-state data gate failed, the preregistered interpretation is
`inconclusive_data_gate_failed`, and `gate_f_authorized=false`.

## Decision

Formal Gate E is **inconclusive**, not positive or negative. The run is large
enough to show that the protocol mostly executes, but it does not meet the
pre-registered requirement of at least 24/30 complete terminal states. Do not
start Gate F, intervention, distillation, OPSD, or RL from this artifact.

The next valid action is a targeted protocol/data-coverage decision for the six
incomplete terminal states. Any rerun must preserve the fixed thresholds and
must be recorded separately from this result.
