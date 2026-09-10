# Gate F2 — immediate-search new-seed replication

Date: 2026-09-10

Frozen implementation commit: `8c8a485`

Protocol: `GATE_F2_RUN.md`

Summary: `results/gate_f2_terminal_summary.json`

## Question and protocol

Gate F2 tested whether six cheap one-step vocal-strategy probes can improve
terminal negotiation utility over the frozen best-fixed style. For each CRAD
80--99 state and each new seed 40--45, the runner generated all six matched
openings and one opponent response per style, selected maximum immediate
creditor utility with fixed config-order tie-breaking, and continued only the
selected branch under the existing neutral H=8/H=12 policy.

The primary preregistered comparison was paired terminal utility:

\[
U_{\rm immediate\ search}-U_{\rm best\ fixed}.
\]

PASS required a 10,000-repeat paired-bootstrap 95% CI lower bound strictly
greater than zero. CRAD 80--99 had been used by the original Gate F, so this was
explicitly registered as a new-seed replication rather than a pristine
new-scenario test.

## Data gate

| Check | Result |
|---|---:|
| Search terminal states | 120/120 |
| Best-fixed terminal states | 120/120 |
| Common paired states | 120/120 |
| One-step probes | 720/720 |
| Eligible probe rate | 1.000 |
| Selected-probe eligible rate | 1.000 |
| Exact replay verification | 1.000 |
| Fallback states | 0 |

The data gate passes.

As an additional implementation audit, the empathetic/warm candidate saved in
each search state's probe set was compared with the separately executed
best-fixed branch. Opening transcript, opening audio token IDs, first opponent
transcript, first opponent audio token IDs, and parsed first offer all matched
exactly in 120/120 states. The paired seed schedule and replay mechanism are
therefore operating as intended.

## Formal result

| Method | Mean terminal utility | Median | Agreement | NO DEAL / forced timeout | Mean rounds |
|---|---:|---:|---:|---:|---:|
| Immediate search | 0.6787 | 0.8444 | 105 | 15 | 3.3833 |
| Best fixed | 0.6685 | 0.8444 | 109 | 11 | 3.0000 |

The paired result is:

\[
\Delta U=+0.0102039,\qquad CI_{95\%}=[-0.0569606,\ 0.0745645].
\]

The lower bound is not greater than zero, so `gate_f2_passes=false`.

## Failure analysis

This failure is not explained by malformed probes or nondeterministic replay.
The immediate signal was highly tied on this replication:

- 102/120 states had more than one immediate-best style;
- 50/120 states had identical immediate utility for all six styles;
- only 25/120 states had an immediate-search offer strictly better than the
  frozen empathetic/warm best-fixed offer.

Across all states, search improved terminal utility in 30 states, tied in 70,
and was worse in 20. It also produced four more forced NO-DEAL timeouts than
best-fixed. The resulting positive point estimate is too small and uncertain
to support a method claim.

Gate E2 remains valid evidence that observed one-step utility and terminal
utility align on scenarios 20--29. Gate F2 shows that this does not reliably
translate into a held-out method gain when many one-step style outcomes are
tied and downstream continuation remains stochastic.

## Stop rule

The preregistered stop rule applies. Do not rerun favorable seeds, change
tie-breaking after observing these outcomes, or call F3/F4 authorized from this
result. Any future safe-tie, immediate-accept, larger-sample, second-opponent,
or second-domain experiment must be registered as a new hypothesis and must
not overwrite the negative Gate F or Gate F2 conclusions.
