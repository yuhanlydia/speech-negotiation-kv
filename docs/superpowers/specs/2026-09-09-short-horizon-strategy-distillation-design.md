# Short-Horizon Strategy Distillation Design

Date: 2026-09-09

## Motivation

Gate D established a cross-scenario vocal-strategy geometry. Gate E2 then established that the immediate opponent reaction to a matched vocal intervention strongly predicts terminal CRAD utility. The method phase should therefore exploit **one-step supervision** rather than pay for long-horizon rollouts during training.

## Core hypothesis

For state `s` and vocal strategy `z`, let `u1(s,z)` be first-response CRAD utility and `c_z` be the Gate-D strategy coordinate. Train a state-conditioned scorer

\[
q_\theta(s,z)=\phi(s)^\top W c_z
\]

from one-step labels only. The selector chooses `argmax_z q_theta(s,z)` at inference. The frozen GLM-4-Voice model is not fine-tuned in this stage.

The teacher is tie-aware. Within each matched state,

\[
A_1(s,z)=u_1(s,z)-\frac1K\sum_j u_1(s,z_j),
\qquad
p_T(z|s)=\operatorname{softmax}(A_1/\tau).
\]

A state is weighted by its immediate utility spread so states with no measurable strategy effect contribute little.

## Data split

- Gate E2 confirmation remains untouched: CRAD train scenarios 20--29.
- Gate F train: CRAD train scenarios 30--59.
- Gate F validation / model selection: scenarios 60--69.
- Gate G robustness: scenarios 70--79.
- Final held-out benchmark: CRAD test scenarios 80--99.

Default training seeds are 20--25. The six existing vocal styles remain unchanged.

## Gate F components

1. Generate one-step matched speech teacher data only; no terminal rollout is used to create training labels.
2. Fit four CPU selectors from the same data:
   - global per-style lookup;
   - non-geometry state x style one-hot ridge head;
   - geometry-aware bilinear ridge scorer using Gate-D coordinates;
   - immediate oracle, reported only as an upper-bound teacher at evaluation time.
3. Persist selector artifacts as lightweight NPZ + JSON metadata.
4. Evaluate selector quality on held-out scenarios with within-state Spearman, tie-aware Top-1, expected immediate utility, and regret to one-step oracle.
5. Run the selected opening strategy on final CRAD 80--99 under the existing neutral continuation protocol and compare terminal utility with neutral, random, and best-fixed-style baselines using paired bootstrap intervals.

## Gate G components

### G1: geometry necessity

Keep the same state features, train split, labels, ridge selection protocol, and held-out scenarios. Compare the geometry-aware bilinear scorer against the state x style one-hot scorer. Geometry is useful only if it improves held-out strategy ranking / Top-1 or data efficiency; parameter-count differences must be reported.

### G2: unseen-strategy generalization

For each of the six styles, remove that style from all selector-training rows. The geometry model still receives the held-out style coordinate `c_z` at evaluation. Evaluate on held-out scenarios where all six styles are observed. Report:

- held-out-style utility prediction correlation across states;
- pairwise sign accuracy of held-out style versus each seen style;
- all-six Top-1 accuracy and regret.

The one-hot class head has no learned held-out-style parameter and uses its pre-registered fallback: the mean score over seen styles for that state. This is an explicit structural baseline, not a claim that the classifier supports zero-shot classes.

## State representation for this phase

Use deterministic CRAD opening-state features only:

- normalized creditor target days;
- normalized debtor target days;
- normalized bargaining gap;
- normalized midpoint;
- normalized target ratio;
- constant bias handled by the regression routine.

Features are standardized on training rows only. No terminal information and no test-scenario statistics enter training.

## Formal evaluation rules

All model selection uses scenarios 60--69. Scenarios 80--99 stay untouched until the selector and hyperparameters are frozen.

Primary method metric is **terminal CRAD utility** on scenarios 80--99 with paired `(scenario, seed)` comparisons. Secondary metrics are agreement rate, no-deal rate, rounds, regret to one-step oracle, generated branches/tokens, and wall-clock cost when available.

A method-level Gate F pass requires geometry-aware selection to beat best-fixed-style terminal utility with paired bootstrap 95% CI lower bound above zero. A stronger result is near-parity with a terminal-trained selector at materially lower rollout cost.

## Resource constraints

- GLM-4-Voice-9B NF4/int4, one model copy.
- Generation microbatch 1.
- Direct audio-token loopback.
- CPU fitting for Gate F/G selectors.
- Raw speech / JSONL remains gitignored; commit lightweight summaries and model metadata only.

## Non-goals

This phase does not implement K/V causal steering, OPSD, GRPO, LoRA/SFT, cross-model transfer, or cross-domain transfer. Those are gated on a successful held-out terminal selector result.
