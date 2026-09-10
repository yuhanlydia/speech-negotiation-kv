# Speech Negotiation KV

Research code for **speech-native adversarial negotiation**, internal vocal-strategy geometry, and short-horizon strategy distillation on CRAD.

## Current scientific result

The project no longer treats a single opponent-stable advantage K/V direction as the main hypothesis. The current evidence is:

- **Gate A — vocal causal channel: PASS.** With the words held fixed, changing vocal delivery changes the frozen opponent's next offer and utility proxy.
- **Gate B / B′ — fixed advantage subspace: NOT SUPPORTED.** Corrected action-side K/V did not recover a cross-scenario advantage-specific subspace above the shuffled-utility null.
- **Gate D — shared strategy geometry: STRONG PASS.** Action-side vocal styles transfer across held-out CRAD scenarios: median six-way decoding accuracy `0.8500` versus chance `0.1667`, all `126/126` balanced scenario splits exceed their own shuffled p95, and `15/15` pairwise style directions pass BH-FDR.
- **Gate E2 — short-horizon value sufficiency: STRONG CONFIRMATORY PASS.** On fresh CRAD scenarios 20--29, seeds 10--15, and six styles (`360` long-horizon branches), immediate first-response utility predicts terminal utility with mean within-state Spearman `0.6586`, bootstrap 95% CI `[0.5677, 0.7642]`, tie-aware best-style agreement `0.950`, and median regret `0.0000`.

The current paper hypothesis is therefore:

\[
\boxed{\text{stable vocal-strategy geometry} + \text{short-horizon value sufficiency}}
\]

A one-step opponent reaction may provide cheap supervision for a strategy that would otherwise require an expensive long-horizon rollout to evaluate.

## Current method phase: Gate F / Gate G

The repo now implements the next method phase. **Follow `GATE_F_RUN.md` rather than the old Gate-A/B execution order.**

### Gate F — short-horizon strategy distillation

For state `s` and vocal strategy `z`, Gate F learns a small selector from **one-step utility only**. The geometry-aware scorer is

\[
q_\theta(s,z)=\phi(s)^\top W c_z,
\]

where `c_z` is the fixed Gate-D strategy coordinate and `phi(s)` contains deterministic CRAD opening-state features. GLM-4-Voice weights remain frozen.

The training target is a tie-aware soft teacher from centered immediate utility. States with zero strategy spread are automatically down-weighted. The formal selector is compared against:

- global per-style lookup / best fixed style;
- a matched state × one-hot-style ridge head with no Gate-D geometry;
- neutral and deterministic-random opening policies;
- the geometry-aware selector.

### Gate G1 — geometry necessity

Compare the geometry-aware selector against the matched one-hot style head on the same train labels, state features, ridge grid, and held-out robustness scenarios. Parameter counts are reported explicitly.

### Gate G2 — unseen-strategy generalization

Leave one vocal strategy out of all value-training rows. The geometry model still receives its pre-existing Gate-D coordinate at test time, while the one-hot head has no learned held-out class parameter and uses the pre-registered state-mean fallback. Report held-out-style correlation, pairwise sign accuracy, all-six Top-1, and regret.

## Fixed data split

Do not change these ranges after seeing results:

| Purpose | CRAD scenarios |
|---|---:|
| Gate D / earlier mechanism pilots | 0--19 |
| Gate E2 fresh confirmation | 20--29 |
| Gate F one-step train | 30--59 |
| Gate F validation / ridge selection | 60--69 |
| Gate G robustness | 70--79 |
| **Final held-out terminal benchmark** | **80--99** |

Default Gate-F one-step teacher seeds are `20--25`. Final held-out terminal seeds are `30--35`.

## Formal Gate-F pass rule

The final benchmark runs one selected opening style on CRAD 80--99; every later turn uses the same frozen neutral continuation policy and the Gate-E2 terminal protocol (`H=8`, then at most `H=12`).

Primary metric: terminal CRAD utility, paired by `(scenario, seed)`.

Gate F passes only if

\[
\operatorname{LCB}_{95\%}\left[
U_{\text{geometry}}-U_{\text{best-fixed}}
\right] > 0.
\]

Secondary outputs include agreement/no-deal rate, rounds, forced-timeout provenance, selected-style distribution, regret, and paired deltas versus neutral/random/one-hot when available.

Until this final held-out terminal gate passes, **do not start K/V causal steering (G4), OPSD, GRPO, LoRA/SFT, cross-model transfer, or cross-domain transfer.**

### Formal Gate F result and debug audit

The first run exposed two implementation defects: the one-step parser could
mistake a referenced target for the proposed offer, and the fitter used linear
utility logits instead of the specified soft teacher. Both were fixed, the
saved teacher transcripts were conservatively reparsed, and the affected
selector and terminal arms were rerun from fresh files.

In the corrected run, all five methods have 120/120 terminal states on CRAD
80--99. Geometry has mean terminal utility `0.6385`, versus `0.6321` for
best-fixed, `0.6395` for neutral, and `0.6058` for random. The fixed
geometry-minus-best-fixed paired delta is `+0.0064`, with bootstrap 95% CI
`[-0.0368, 0.0508]`. Therefore `gate_f_passes=false` remains the honest
conclusion.

Gate G1 also found identical Top-1 accuracy for geometry and one-hot (`0.7895`
each). With six styles, the centered rank-5 Gate-D coordinates span the same
five-dimensional contrast space as centered one-hot, so the current G1 is a
reparameterization rather than a valid geometry-necessity test. G4 K/V
steering, OPSD, GRPO and fine-tuning remain blocked. Details are in
`docs/results/gate_f_terminal_report.md`.

### Gate F2 immediate-search replication

The preregistered six-probe recovery experiment is complete on CRAD 80--99
with new seeds 40--45. Both immediate-search and frozen best-fixed have 120/120
terminal states; all 720 probes were eligible and all selected openings passed
exact deterministic replay. Immediate-search reached mean terminal utility
`0.6787` versus `0.6685` for best-fixed, a paired delta of `+0.0102` with 95%
CI `[-0.0570, 0.0746]`. Therefore `gate_f2_passes=false`.

The negative is not a runtime failure. Immediate utilities were tied for
multiple styles in 102/120 states, including 50/120 states where all six styles
tied; search had a strictly better immediate offer than best-fixed in only
25/120 states. Search also had 15 forced NO-DEAL timeouts versus 11 for
best-fixed. Per the frozen stop rule, do not run F3/F4 by repeatedly changing
seeds or tie-breaking. Full details are in
`docs/results/gate_f2_terminal_report.md`.

## Exact run instructions

Use:

```bash
git pull origin main
source .venv/bin/activate
pip install -e '.[dev,glm]'
pytest -q
```

Then follow the complete command sequence in:

- `GATE_F_RUN.md`
- `configs/crad_gate_f_16gb.yaml`
- `docs/superpowers/specs/2026-09-09-short-horizon-strategy-distillation-design.md`
- `docs/superpowers/plans/2026-09-09-gate-f-g-strategy-distillation.md`

The run contract covers:

1. one-step teacher collection on scenarios 30--59 (`1080` branches);
2. validation teacher collection on 60--69 (`360` branches);
3. CPU fitting of geometry / one-hot / style-lookup selectors;
4. robustness teacher collection on 70--79 (`360` branches) and Gate G1/G2;
5. final terminal evaluation on 80--99 (`120` states per method);
6. paired-bootstrap Gate-F analysis.

Raw JSONL / K/V NPZ / audio artifacts remain local and gitignored. Commit lightweight summaries and reports only.

## Model and resource mode

Primary experiments use:

- `zai-org/glm-4-voice-9b`;
- NF4/int4;
- one model copy;
- generation microbatch `1`;
- direct discrete audio-token loopback;
- no waveform decoder / tokenizer roundtrip in the primary 16 GB path.

The method fitting and Gate-G analyses are NumPy/CPU-only after one-step teacher data and Gate-D coordinates exist.

## Key result records

- Original Gate A/B report: `results/GATE_REPORT.md`
- Corrected Gate B′: `results/debug_gate_b_prime_report.md`
- Gate D strategy geometry: `results/gate_d_strategy_geometry_report.md`
- Formal Gate E v7 (inconclusive coverage): `docs/results/gate_e_formal_v7_report.md`
- Formal Gate E2 confirmation: `docs/results/gate_e2_formal_v3_report.md`
- Gate E2 summary: `results/gate_e2_formal_v3_summary.json`
- Earlier exploratory contextual Gate F: `docs/results/gate_f_exploratory_report.md` (historical only; not the current formal method)

## Tests

```bash
pytest -q
```

The repo tests cover CRAD scoring/parsing, long-horizon terminal protocol, matched speech semantics, K/V extraction and subspace controls, Gate-D geometry, Gate-E2 metrics, and the new short-horizon selector primitives including tie-aware teachers, geometry/one-hot scoring, regret, bootstrap deltas, and unseen-strategy valuation.
