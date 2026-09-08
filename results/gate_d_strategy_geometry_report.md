# Gate D — Shared Strategy Geometry Report

Date: 2026-09-08

## Hypothesis

Gate D separates strategy representation from strategy value:

\[
m_{s,z}=\mu_s+B c_z+\epsilon,
\qquad
A(s,z)=w(s)^\top c_z+\eta.
\]

It asks whether the action-side representation of the six vocal strategies is
stable across scenarios. It does not use utility labels and therefore does not
test whether any strategy is globally advantageous.

## Protocol

- Input: the existing 120 action-side K/V tensors from 10 CRAD scenarios,
  2 seeds, and 6 matched-transcript styles.
- Primary representation: all selected layers (16, 20, 24, 28, 32, 36),
  concatenated K/V, audio-token-only pooling.
- Control: the same representation at the last audio token only.
- State removal: subtract the six-style mean separately inside every
  scenario/seed state.
- D1 decoder: parameter-free cosine nearest-style-centroid classifier.
- Generalization: all 126 unique 5/5 scenario partitions, evaluated in both
  A-to-B and B-to-A directions and averaged per partition.
- D1 null: 1,000 independent style-label permutations within each state.
- D2: all 15 oriented style-pair directions, comparing normalized direction
  cosines across different scenarios.
- D2 null: 1,000 within-state style permutations; 15 pairwise p-values are
  corrected with Benjamini-Hochberg FDR at 0.05.
- Gate rule: D1 median held-out accuracy above 1/6 with permutation
  p<=0.05, plus at least one positive D2 pair passing BH-FDR.

## D1 — Cross-scenario style decoding

| Representation | Chance | Held-out median | Held-out p05–p95 | Null median | Permutation p | Splits above own null p95 |
|---|---:|---:|---:|---:|---:|---:|
| Action audio-only | 0.1667 | 0.8500 | 0.8000–0.8833 | 0.1667 | 0.000999 | 126/126 |
| Action last-audio | 0.1667 | 0.5292 | 0.3792–0.6500 | 0.1667 | 0.000999 | 126/126 |

The full audio trajectory transfers substantially better than a single final
audio token. The primary result is not driven by one arbitrary 5/5 split.

The audio-only aggregate confusion audit shows that firm/assertive and
hesitant/uncertain are perfectly decoded across the repeated bidirectional
tests. Neutral and urgent are also highly stable. Calm/confident is less
separable and is often confused with neutral, urgent, or empathetic styles;
the six labels should therefore remain separate in Gate E rather than being
post-hoc merged.

## D2 — Pairwise style directions

- Action audio-only: 15/15 style pairs have positive cross-scenario direction
  alignment and pass BH-FDR.
- Action last-audio: 14/15 pairs pass BH-FDR.
- Strongest audio-only pair: firm/assertive versus hesitant/uncertain,
  cross-scenario mean cosine 0.8033.
- Weakest audio-only pair: calm/confident versus neutral, mean cosine 0.1369,
  still significant after FDR (q=0.003996).
- The only non-significant last-audio pair is calm/confident versus
  empathetic/warm (mean cosine 0.0171, q=0.1668).

## Decision

**Gate D passes.** The same vocal strategies occupy transferable internal
directions across held-out CRAD scenarios even though Gate B′ could not find a
fixed utility-bearing direction. Together, the current evidence supports the
pilot-level **Strategy–Value Gap**:

> Strategy representation is stable across context, while the direction of
> strategic value is not captured by a single context-independent vector.

This result authorizes Gate E, which must replace next-offer utility with a
paired long-horizon outcome. It does not authorize Gate F, intervention,
distillation, OPSD, or RL until the long-horizon data are collected and the
context-conditioned value comparison is evaluated.

Raw action K/V remains local and gitignored. Machine-readable results are in:

- `results/gate_d_action_audio_summary.json`
- `results/gate_d_action_last_audio_summary.json`
