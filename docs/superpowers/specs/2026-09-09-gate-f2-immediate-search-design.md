# Gate F2 Immediate-Search Recovery Design

Date: 2026-09-09

## Status and motivation

The original Gate F remains a valid negative result. Its frozen pre-response
geometry selector reached mean terminal utility 0.6385 versus 0.6321 for the
best fixed style, but the paired 95% confidence interval crossed zero. Debugging
showed that the selector only observes deterministic scenario targets while the
best immediate style varies substantially across sampling seeds.

Gate E2 provides an independent mechanism result: once the first opponent
reaction is observed, immediate utility strongly predicts terminal utility.
On the 60 Gate-E2 states, a deterministic immediate-best replay policy has mean
terminal utility 0.8452 versus 0.7521 for the strongest fixed style in that
dataset, a paired delta of 0.0930 with bootstrap 95% CI [0.0232, 0.1651]. Gate
F2 tests this actionable policy prospectively rather than continuing to tune a
pre-response selector that cannot observe reaction noise.

## Method

For each CRAD state and paired seed, render the same creditor opening in all six
registered vocal styles. Feed each audio realization to the frozen debtor for
exactly one response. Parse the six proposals and choose the style with maximum
normalized creditor immediate utility. Ties are broken by the existing fixed
style order. Unmatched, audio-empty, or unparseable candidates are ineligible;
if all six are ineligible, use neutral and record the fallback.

Only the selected branch continues under the existing neutral/base policy to
H=8, then H=12 with the existing forced-NO-DEAL provenance rule. The selected
opening and first response are regenerated with the identical deterministic
seed schedule before continuation. The runner must assert that the regenerated
transcripts, audio token IDs, and parsed offer exactly match the selected probe;
otherwise the branch is invalid rather than silently continuing a different
trajectory.

This method uses six short probes plus one selected long rollout. It is a
test-time search/control result, not a successful distillation result and not a
claim that K/V geometry is necessary.

## Confirmatory replication

- Model: GLM-4-Voice-9B, int4/NF4, one model copy, microbatch 1.
- Scenarios: CRAD 80--99.
- New seeds: 40, 41, 42, 43, 44, 45.
- Styles: the existing six registered styles.
- Method states: 20 x 6 = 120.
- Probe branches: 20 x 6 x 6 = 720 one-step candidates.
- Baseline: best-fixed style from the frozen corrected Gate-F artifact.
- Pairing key: `(scenario_id, seed)`.

Scenarios 80--99 were used by the original Gate F, so this is explicitly a
new-seed replication, not a new-scenario pristine test. Gate F2 has no learned
or tuned scenario-specific parameters; its selection rule was frozen from the
Gate-E2 mechanism before these seeds are run.

## Gates and metrics

Data validity requires 120/120 terminal states for both immediate-search and
best-fixed, matched opening semantics, nonempty audio for the selected branch,
successful deterministic replay, and explicit agreement or NO DEAL outcomes.
Coverage failure makes Gate F2 inconclusive and triggers debugging/rerun without
changing the scientific threshold.

The primary metric is paired terminal utility:

`utility(immediate_search) - utility(best_fixed)`.

Gate F2 passes only when the 10,000-repeat paired bootstrap 95% confidence
interval has lower bound strictly greater than zero. Secondary metrics are mean
utility, agreement/NO-DEAL rate, forced timeout rate, selected-style counts,
probe parseability, fallback rate, rounds, and delta versus neutral/random if
those optional replication arms are run.

## Stop rule

If Gate F2 passes, the supported method is short-horizon test-time strategy
search. The original distilled geometry Gate F remains negative. Distillation
may be revisited only as a separate method that reproduces the search teacher.
If Gate F2 fails after valid coverage and implementation verification, stop
claiming that one-step search improves this benchmark; do not weaken the gate,
change tie-breaking, or select favorable seeds after observing outcomes.
