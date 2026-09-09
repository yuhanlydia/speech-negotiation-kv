# Formal Gate F/G — held-out terminal evaluation

Date: 2026-09-09  
Run contract: `GATE_F_RUN.md`  
Corrected selector: `results/gate_f_selector_soft_reparsed_v3.npz` (local, gitignored)

## Completed protocol

- Train teacher: scenarios 30--59, six seeds, 1080 one-step branches.
- Validation teacher: scenarios 60--69, six seeds, 360 one-step branches.
- Robustness teacher: scenarios 70--79, six seeds, 360 one-step branches.
- Final held-out terminal test: scenarios 80--99, seeds 30--35, 120 states
  per method.
- All later turns used the frozen neutral continuation policy and the Gate-E2
  H=8/H=12 terminal protocol.

Teacher collection was complete for all requested keys. A post-run audit found
that the original `parse_offer_days()` selected the first day-valued phrase,
which is wrong for responses such as “20 days is too tight; how about 120
days?” It also treated some truncated responses as utility zero. A new
proposal-cue parser reparsed the saved transcripts without regenerating them.
In train, 11 labels changed to another valid offer and one became newly
unparseable (three unparseable total). In validation, seven changed and three
became newly unparseable (five total). In robustness, three changed and two
became newly unparseable (three total). Any state without all six valid labels
was excluded.

The audit also found that the original fitter did not use the soft teacher
specified in the design. The corrected v3 fitter uses
`softmax(centered immediate utility / temperature)` and weights each state by
its raw immediate-utility spread. Ridge selection still uses validation
scenarios only. All affected terminal method arms were rerun as fresh v3 files.

## Selector and robustness checks

On corrected validation data, geometry and one-hot both reached Top-1
agreement 0.8000; geometry selected ridge 10.0. On corrected Gate-G robustness
data, both reached Top-1 agreement 0.7895. G1 therefore does not show a
geometry necessity advantage. More fundamentally, six centered style
prototypes at rank 5 span the same five-dimensional style-contrast space as a
centered six-way one-hot code. The geometry and one-hot selectors are thus
equivalent up to a change of basis in this experiment; they selected exactly
the same final styles and produced identical terminal trajectories.

G2 was rerun for all six leave-one-style-out conditions and is included in
`results/gate_g_controls_soft_reparsed_v3_summary.json`. Results are mixed and
do not establish unseen-style geometry transfer.

## Final held-out results

| Method | Mean terminal utility | Agreements | NO DEAL | Forced timeout |
|---|---:|---:|---:|---:|
| Geometry | 0.6385 | 111 | 9 | 9 |
| One-hot | 0.6385 | 111 | 9 | 9 |
| Best fixed | 0.6321 | 109 | 11 | 11 |
| Neutral | 0.6395 | 114 | 6 | 6 |
| Random | 0.6058 | 106 | 14 | 14 |

The pre-registered primary comparison is geometry minus best-fixed:

\[
\Delta U=+0.00641,\qquad CI_{95\%}=[-0.03681, 0.05079].
\]

The formal Gate-F rule requires the paired bootstrap lower bound to be greater
than zero. It is not met, so `gate_f_passes=false`.

## Interpretation and stop rule

The parser and teacher-target bugs materially changed the selector: corrected
geometry improved from 0.6270 to 0.6385 mean terminal utility. They did not,
however, change the Gate decision. Gate E2 shows that an *observed* immediate
reaction ranks long-horizon value, while Gate F asks a harder question: can
coarse opening-state features predict that reaction before observing it? The
teacher's best-style set is unstable across the six sampling seeds, whereas
the selector receives only scenario targets and therefore chooses one style
for every seed of a scenario. This unobserved stochastic variation limits the
current selector.

The corrected result is a negative method result, not a runtime or coverage
failure: every method has 120/120 paired terminal states. It also shows that
the present G1 comparison cannot identify a benefit from geometry because the
rank-5 coordinate system is expressively equivalent to centered one-hot.

Per the run contract, do not start G4 K/V steering, OPSD, GRPO, LoRA/SFT,
cross-model transfer, or cross-domain experiments from this result.
