# Formal Gate E2 — fresh long-horizon confirmation

Date: 2026-09-09  
Protocol commit: `932c8d9`  
Result summary: `results/gate_e2_formal_v3_summary.json`

## Question

Does the first-round immediate utility of a matched vocal intervention predict
its terminal CRAD utility after the rest of the negotiation continues under
the neutral/base policy?

This is a confirmatory alignment test. It is not a pre-registered test of an
immediate-influence/long-horizon *gap*.

## Protocol

- Model: GLM-4-Voice-9B, int4, token-loopback, one model copy, batch 1.
- CRAD split: train scenarios 20--29.
- Seeds: 10--15.
- Styles: neutral; calm and confident; firm and assertive; empathetic and
  warm; urgent but controlled; hesitant and uncertain.
- Total: `10 x 6 x 6 = 360` matched branches.
- All branches ran through H=8 and then up to H=12.
- At H=12, an unresolved branch is explicitly recorded as protocol-forced
  `NO DEAL`, with `terminal_forced_no_deal=true`. This is a timeout outcome,
  not a fabricated model agreement.

## Fixed data gate

| Check | Requirement | Result |
|---|---:|---:|
| Branch key coverage | 360/360 exact | 360/360 |
| Matched semantics | >= 0.95 | 1.000 |
| Parseable moves | >= 0.95 | 1.000 |
| Strategically valid moves | >= 0.95 | 0.985 |
| Terminal completion | >= 0.95 | 1.000 (360/360) |
| Complete terminal states | >= 54/60 | 60/60 |

The data gate therefore passes. Terminal outcomes were 339 agreements and 21
NO DEAL outcomes, of which 20 were explicit H=12 protocol timeouts.

## Confirmatory metrics

For every complete `(scenario, seed)` state, the six styles were ranked by
`immediate_offer_utility` and terminal `utility`.

| Metric | Result |
|---|---:|
| Informative states for Spearman | 43/60 |
| Mean Spearman | 0.6586 |
| Median Spearman | 0.7464 |
| Bootstrap 95% CI for mean | [0.5677, 0.7642] |
| Tie-aware immediate/terminal best-style agreement | 0.950 |
| Median regret | 0.0000 |
| Mean regret | 0.0534 |

The pre-registered positive-alignment condition (bootstrap lower bound > 0),
strong-alignment lower bound (> 0.30), Top-1 threshold (> 0.50), and median
regret threshold (< 0.05) all pass. The analyzer therefore reports
`supports_strong_immediate_predicts_terminal`.

## Protocol debugging before v3

The first fresh run was not silently reused. E2-v1 had 18 invalid branches
concentrated in scenario 29 because the model said `six` where the opening
sentence contained `6`; the semantic normalizer was repaired and the complete
360 branches were rerun as v2. E2-v2 then had 339 agreements, one NO DEAL, and
20 unresolved H=12 branches. Those branches exposed that the terminal prompt
was advisory rather than an enforced terminal rule. The runner was repaired
to record an unresolved terminal-subset branch as protocol-forced NO DEAL,
with an explicit provenance field, and all 360 branches were rerun as v3.

## Sensitivity and limitations

The primary E2 gate treats terminal agreement/NO DEAL as the terminal outcome
and gates strategic validity separately, as specified in the protocol. A
stricter sensitivity analysis that requires every branch in a state to also
have `policy_valid=true` leaves 47/60 complete states; its mean Spearman is
0.6705 with Top-1 agreement 0.9787. This is reported, not used to change the
pre-registered gate.

The 20 timeout outcomes should remain visible in future method comparisons.
They are valid terminal protocol outcomes but are not evidence that the model
itself chose NO DEAL. Gate E2 establishes immediate-to-terminal alignment for
this setup; it does not yet establish a distilled selector, causal improvement
over neutral, or superiority over another method.
