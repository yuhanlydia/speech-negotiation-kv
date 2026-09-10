# ParaGeo × SpeechParaling-Bench Design

Date: 2026-09-10

## Decision

The negotiation line is retained as the discovery record, not the main benchmark. Gate A and Gate D established that matched vocal delivery changes a frozen speech agent and occupies a transferable internal K/V geometry. Gate F and Gate F2 are valid negative method results and remain reported as such.

The new paper hypothesis is:

\[
\boxed{\text{paralinguistic behavior occupies a low-dimensional, content-invariant, compositional geometry that can be directly controlled during speech generation}}
\]

The primary benchmark is SpeechParaling-Bench, whose public evaluation covers fine-grained Paralanguage Control, Dynamic Variation, and Situational Adaptation. The first pilot uses English Paralanguage Control and Dynamic Variation only.

## Method

### P1. Content-invariant paralinguistic basis

For matched lexical content `x` rendered with paralinguistic attribute `a`, collect action-side K/V representation `h(x,a)`. Remove content by within-content centering:

\[
\tilde h(x,a)=h(x,a)-\frac{1}{|A|}\sum_{a'}h(x,a').
\]

Fit a truncated SVD basis:

\[
\tilde h(x,a)\approx Bc_a,\qquad B\in\mathbb R^{d\times r},\quad r\ll |A|.
\]

The pilot catalog uses reusable controls from SpeechParaling-Bench short-single prompts for Pitch, Timbre, Pace, Volume, Rhythm, and Age.

### P2. Semantic-orthogonal basis

Compute content means and their semantic basis `S`. Remove semantic directions from `B`:

\[
B_{\rm para}=\operatorname{qr}((I-SS^\top)B).
\]

All steering uses `B_para`. This is intended to alter delivery while preserving lexical content.

### P3. Static control

For a requested attribute `a`, apply:

\[
\Delta h=\alpha B_{\rm para}c_a.
\]

The baseline receives the identical benchmark input audio and uses the official GLM-4-Voice generation path without steering.

### P4. Compositional control

For unseen multi-attribute instructions, compose single-attribute coordinates without fitting the combination:

\[
c_{a_1+\cdots+a_m}=\sum_i c_{a_i},\qquad \Delta h=\alpha B_{\rm para}c_{a_1+\cdots+a_m}.
\]

The pilot uses the public short-multi subset. The first pilot uses the benchmark prompt transcript only as an oracle router so that routing/ASR quality does not confound the geometry test. A successful pilot must later replace this oracle with an end-to-end router.

### P5. Dynamic control

For intra-utterance transitions from `a` to `b`, define a coordinate trajectory:

\[
c_t=(1-\lambda_t)c_a+\lambda_t c_b.
\]

Gradual benchmark prompts use linear interpolation; prompts containing `suddenly` use a step schedule. A scheduled fused-QKV hook advances once per model generation forward and applies the corresponding K/V direction to all selected layers.

### P6. Random-direction control

Random rank-matched directions with the same norm as the target ParaGeo direction are required to distinguish generic perturbation from paralinguistic geometry.

## Full speech pipeline

The output path must use all three official GLM-4-Voice components:

1. GLM-4-Voice-Tokenizer / Whisper-VQ for benchmark input waveform tokenization;
2. GLM-4-Voice-9B for interleaved text/audio-token generation;
3. GLM-4-Voice-Decoder (Flow + HiFT) for output waveform decoding at 22.05 kHz.

The wrapper follows the official repository imports (`speech_tokenizer.modeling_whisper.WhisperVQEncoder`, `speech_tokenizer.utils.extract_speech_token`, and `flow_inference.AudioDecoder`). Primary generation temperature is `0.8`, with `top_p=0.8`.

## Pilot benchmark

English only.

- Static single control: first 80 eligible items from `para_con/short_sin` restricted to Pitch, Timbre, Pace, Volume, Rhythm, Age.
- Compositional control: first 40 `para_con/short_multi` items for which every requested reusable control exists in the catalog.
- Dynamic variation: first 60 `dyn_var` items whose start/end controls are present in the catalog.

Methods:

- `prompt_only`
- `parageo_static`
- `parageo_composed`
- `parageo_dynamic`
- `random`

The benchmark's official pairwise judge and score calculator remain the primary metric. Generated audio follows the benchmark directory/file naming convention.

## Scale selection without test leakage

Use a deterministic 20% development slice by item index (`index % 5 == 0`) only to choose `alpha` from `[0.25, 0.5, 1.0, 1.5]`. Freeze one scale per task family before evaluating the remaining 80% pilot items. Do not tune on the held-out pilot scores.

## Semantic fidelity

Report text-channel WER between the benchmark target sentence and GLM's generated text stream. Steering is rejected if held-out mean WER worsens by more than 0.02 absolute versus prompt-only. The full benchmark judge remains the speech-quality/control metric; text-channel WER is a safety check against lexical corruption.

## Pilot pass / stop rule

Continue only if the held-out pilot satisfies semantic fidelity and at least one of:

- static or compositional score gain >= 5.0 points over prompt-only;
- Dynamic Variation score gain >= 8.0 points over prompt-only.

Random steering must not reproduce the gain.

If the pilot fails these frozen criteria, stop ParaGeo rather than adding RL, LoRA, new seeds, or post-hoc steering schedules.

## Expansion only after a pass

1. Full SpeechParaling-Bench English 1001 items.
2. Full Chinese 1001 items.
3. Situational Adaptation with a learned context-to-coordinate router.
4. Qwen-Omni second-model replication.
5. Cross-model comparison of geometry rank and compositional transfer.
6. Optional S2S-Arena / WildSpeech-Bench external generalization.

## Claims prohibited before pilot pass

Do not claim that ParaGeo improves SpeechParaling-Bench, that geometry is universally low-rank across speech LMs, or that composition generalizes to unseen controls until the corresponding experiments are complete.
