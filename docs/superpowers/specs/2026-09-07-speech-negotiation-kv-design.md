# Speech Negotiation KV — Design

## Research question

Test whether strategically useful internal memory in speech-native CRAD negotiation factorizes into a scenario-dependent component plus a low-dimensional component that is stable for a fixed opponent. The first falsifiable hypothesis is

\[
\Delta m_{s,o} \approx B_o c_s,
\]

where \(\Delta m\) is an advantage-weighted matched K/V difference, \(s\) is a CRAD state, and \(o\) is the frozen opponent.

## Non-negotiable benchmark constraints

- Dataset: CRAD only for the first paper/pilot.
- Preserve original CRAD target-day structure and report original-style Success/Outcome/Utility/Rounds wherever a full trajectory is available.
- Train split: first 80 scenarios; held-out test: final 20 scenarios, matching the existing CRAD experimental convention.
- The opponent is frozen during all Gate-1/2 experiments.
- Counterfactual branches must share scenario, state, semantic utterance and decoding seed; the intended speech realization is the only branch intervention.

## Primary 16 GB execution mode

GLM-4-Voice-9B is loaded once in NF4/int4. We do not load the waveform decoder or Whisper-VQ encoder in the primary run. The model's generated discrete audio token IDs are passed directly to the opponent as `<|begin_of_audio|><|audio_N|>...<|end_of_audio|>`. This preserves the model's speech modality while avoiding a second codec model and waveform round trip. A later 24 GB confirmatory run can decode/re-tokenize waveforms.

Generation is sequential (microbatch 1), while analysis batches CPU-side arrays. KV statistics are immediately detached, cast to fp16, and offloaded to CPU/disk. Default tapped layers are 16, 20, 24, 28, 32, 36 of the 40-layer ChatGLM backbone. No two full model copies may coexist on GPU.

## Phase 0: capability gate

Run a small deterministic CRAD subset. Require parseable day proposals and stable role adherence. If the base model cannot produce valid negotiation responses, stop and repair prompting before any subspace claim.

## Phase 1: matched vocal counterfactual sweep

At a fixed CRAD state, produce one semantic creditor utterance. Render that same utterance under a small style bank (neutral, calm-confident, assertive, empathetic, urgent, hesitant). Reject a branch if the decoded transcript changes materially. Feed only its audio tokens to the frozen debtor and record the debtor response, next offer, branch seed and resulting utility/progress signal.

This phase tests the basic phenomenon: same words + different speech realization -> different opponent policy.

## Phase 2: advantage-KV subspace

For each matched state group, let `m_k` be the concatenated pooled K/V statistics for branch `k` and `U_k` its downstream utility. Define

\[
A_k = U_k - \bar U,
\qquad
 g_s = \sum_k \frac{A_k}{\sum_j |A_j| + \epsilon}(m_k-\bar m).
\]

Normalize each nonzero \(g_s\). Stack rows into \(G_o\) and fit a rank-r right singular subspace. Raw-good-KV PCA is not the primary estimator because it is confounded by scenario semantics, numbers and round position.

## Required controls

1. Raw successful-KV PCA.
2. Shuffled utilities/advantages within matched groups.
3. Random rank-matched subspace.
4. Orthogonal-complement direction.
5. Scenario-half reproducibility: fit on disjoint scenario halves and measure principal-subspace overlap.

The research claim is not supported unless the advantage subspace is more stable/predictive than these controls.

## Phase 3: few-shot opponent calibration

If a shared strategic basis exists, represent a new opponent by a low-dimensional code `w_o` fitted from 4-16 short probes with ridge regression. This phase is only justified after Phase 2 establishes low rank and held-out stability.

## Stop gates

- Gate A: matched vocal branches must cause nontrivial opponent-action/utility variance.
- Gate B: rank 4/8/16 advantage subspaces must show reproducible scenario-held-out overlap above shuffled/random controls.
- Gate C: a direction derived from the correct opponent must outperform sign-flip/shuffled/orthogonal steering on held-out CRAD utility.
- If A or B fails, do not proceed to OPSD/RL or full long-horizon training.

## Repository deliverable

The repository must provide: CRAD downloader/scorer; 16/24 GB configs; GLM-4-Voice token-loopback adapter; matched sweep record format; fused-QKV extraction/steering primitives; subspace fitting and controls; few-shot ridge calibration; dry-run backend; tests; exact run order in README.
