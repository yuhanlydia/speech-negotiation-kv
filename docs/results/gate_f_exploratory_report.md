# Exploratory Gate F report

Date: 2026-09-09

## Scope and limitations

This is an exploratory CPU analysis performed after formal Gate E failed its
complete-terminal-state data gate. It uses the 20 complete terminal states
available in `results/gate_e_formal_v7.jsonl`; it does not impute or repair the
six incomplete states. `exploratory=true` and `paper_pass_authorized=false`.

Formal Gate-E completion remains `20/30` complete terminal states, below the
pre-registered `24/30` threshold.

Formal Gate-E records contain no K/V captures. Therefore this analysis does
not claim to fit a formal Gate-D-to-Gate-E per-branch K/V model. It uses style
prototype coordinates `c_z` learned from the Gate-D action-audio K/V artifact,
then maps the six style labels in the complete Gate-E states to those fixed
coordinates.

## Evaluation

The split is leave-one-terminal-scenario-out over the five terminal scenarios.
The contextual model uses a ridge regression over the interaction features
`phi(s) tensor c_z`, where `phi(s)` contains targets, bargaining gap, opening
offer, previous concession, and round fraction. Results are state-level
Spearman ranking and Top-1 style selection.

| model | Spearman n | mean | median | Top-1 |
|---|---:|---:|---:|---:|
| style lookup | 18 | -0.0340 | 0.0309 | 0.40 |
| global strategy | 18 | -0.0340 | 0.0309 | 0.40 |
| state-only | 0 finite rankings | — | — | 0.35 |
| contextual bilinear | 18 | 0.0281 | 0.0655 | 0.45 |

The contextual model is directionally better than the global/style lookup
baseline on this incomplete sample, but the uplift is small, has no paired
bootstrap confidence interval here, and is not a formal Gate-F result.

## Decision

This exploratory result does not authorize intervention, distillation, OPSD,
or RL. It is a hypothesis-generating signal only. A formal Gate F requires
passing the Gate-E terminal coverage rule and obtaining complete, held-out
terminal states with the preregistered comparisons.
