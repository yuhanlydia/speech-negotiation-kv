# Formal Gate F/G — held-out terminal evaluation

Date: 2026-09-09  
Run contract: `GATE_F_RUN.md`  
Selector: `results/gate_f_selector.npz` (local, gitignored)

## Completed protocol

- Train teacher: scenarios 30--59, six seeds, 1080 one-step branches.
- Validation teacher: scenarios 60--69, six seeds, 360 one-step branches.
- Robustness teacher: scenarios 70--79, six seeds, 360 one-step branches.
- Final held-out terminal test: scenarios 80--99, seeds 30--35, 120 states
  per method.
- All later turns used the frozen neutral continuation policy and the Gate-E2
  H=8/H=12 terminal protocol.

Teacher collection quality was complete for all requested keys. Matched
semantics were 1.000 on train and validation; parseable immediate offers were
0.9954 and 0.9917 respectively. The selector was fit on CPU, with ridge
selection using validation scenarios only.

## Selector and robustness checks

On validation scenarios 60--69, geometry and one-hot both reached Top-1
agreement 0.7833; geometry selected ridge 100.0. On Gate-G robustness
scenarios 70--79, both reached Top-1 agreement 0.7500. G1 therefore does not
show a geometry necessity advantage. G2 was run for all six leave-one-style-out
conditions and is included in `results/gate_g_controls_summary.json`.

## Final held-out results

| Method | Mean terminal utility | Agreements | NO DEAL | Forced timeout |
|---|---:|---:|---:|---:|
| Geometry | 0.6270 | 109 | 11 | 11 |
| One-hot | 0.6270 | 109 | 11 | 11 |
| Best fixed | 0.6321 | 109 | 11 | 11 |
| Neutral | 0.6395 | 114 | 6 | 6 |
| Random | 0.6058 | 106 | 14 | 14 |

The pre-registered primary comparison is geometry minus best-fixed:

\[
\Delta U=-0.00512,\qquad CI_{95\%}=[-0.04713, 0.03889].
\]

The formal Gate-F rule requires the paired bootstrap lower bound to be greater
than zero. It is not met, so `gate_f_passes=false`.

## Interpretation and stop rule

Gate E2 supports short-horizon value sufficiency as a phenomenon, but this
first frozen one-step selector did not convert that phenomenon into a held-out
terminal improvement. Geometry also did not beat the matched one-hot control.
The result is therefore a negative method result, not evidence for a broken
runtime or an incomplete test: all five methods have 120/120 paired states.

Per the run contract, do not start G4 K/V steering, OPSD, GRPO, LoRA/SFT,
cross-model transfer, or cross-domain experiments from this result.
