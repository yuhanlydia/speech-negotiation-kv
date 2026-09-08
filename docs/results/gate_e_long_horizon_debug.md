# Gate E debugging record

Date: 2026-09-08

## Scope

The planned run was 20 CRAD scenarios × 6 seeds × 6 vocal styles = 720
branches. Every branch used one styled creditor opening, then the frozen
neutral continuation policy. H=4 was retained for all branches; scenarios
`[0, 4, 8, 12, 16]` were the terminal subset allowed to continue to H=8.

## Invalid first formal run

Artifact: `results/gate_e_long_horizon.jsonl` (local, gitignored)

| quantity | result |
|---|---:|
| branches | 720/720 |
| matched openings | 717/720 |
| generated moves | 6,447 |
| parseable moves | 6,443/6,447 |
| strategically valid moves | 6,437/6,447 |
| terminal-subset branches | 180 |
| terminal-subset `censored` | 177 |
| terminal-subset `no_deal` | 3 |
| terminal-subset `agreement` | 0 |
| H=4 terminal utility | 0/720 |

This run is invalid for long-horizon inference because the terminal coverage
gate failed. It is retained locally as a debugging artifact and is not
evidence for a negative result.

## Root causes and fixes

1. Continuation calls passed `opponent_audio_ids` through the backend interface,
   but the prompt builder never rendered them with `audio_ids_to_prompt()`.
   This made later turns text-history-only. The prompt now contains the audio
   special-token loopback.
2. The terminal subset could reach its maximum horizon without a terminal
   action being requested. The final transition for that subset now requires
   either `AGREED: N days` for the latest offer or `NO DEAL`. No terminal value
   is fabricated when the model fails to follow the contract.
3. The validator required `transition >= 3` for agreement, contradicting the
   protocol's early-agreement behavior. It now accepts exact agreement with the
   current latest offer at any transition.

## Small validation run

Artifact: `results/gate_e_smoke_v7.jsonl` (local, gitignored)

The 12-branch real-GPU smoke used one terminal scenario and two seeds. It got
12/12 matched openings, 93/93 parseable moves, 11 agreements, and 1 censored
branch. This smoke was collected before the final validator correction, so its
strategic-validity count is not used as a final metric. It only verifies that
the audio loopback and terminal contract can produce real terminal outcomes.

The attempted formal rerun was stopped at 27/720 branches by request and is
excluded from analysis.

## Current decision

The full test suite passes (`23 passed`). Gate E is still inconclusive because
there is not yet a short post-fix smoke with complete terminal coverage under
the final validator. Do not run Gate F, intervention, distillation, OPSD, or
RL until that small smoke passes its terminal-coverage gate.
