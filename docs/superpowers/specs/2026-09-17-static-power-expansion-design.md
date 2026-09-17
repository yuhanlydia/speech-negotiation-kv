# Static Attribute Power Expansion Design

## Objective

Measure whether ParaGeo directions outperform norm-matched random directions for the 18 Attitude and Cognitive State controls with enough lexical replication to avoid the current two-or-three-example instability.

## Frozen protocol

- Attributes: the nine Attitude and nine Cognitive State controls already present in SpeechParaling-Bench.
- Sample size: exactly 10 distinct English target sentences per attribute, 180 items total.
- Existing benchmark sentences may be retained; missing cells are filled with new lexical content. No target sentence may appear in calibration, development, or another expanded-test cell after normalized-text deduplication.
- Input prompts are synthesized in one fixed neutral English TTS voice. TTS identity is paired across all output variants and is not itself an experimental factor.
- ParaGeo uses the already selected static alpha 0.5 and the frozen r2 basis/layers.
- Random controls use five preregistered seeds: 15242424242, 16242424242, 17242424242, 18242424242, and 19242424242. Each seed produces a deterministic norm-matched random direction independently for every item.
- Primary comparisons are five direct, blinded ParaGeo-vs-random pairwise judges. Prompt-only comparisons are secondary fidelity diagnostics.
- Primary aggregation is equal-weight macro averaging over the 18 attributes. Confidence intervals use hierarchical bootstrap: sample attributes, then utterances within attributes.
- No seed, item, or judge output may be dropped because of score. Generation/API failures are retried only under the same frozen inputs and seeds; persistent failures remain explicit missingness/limitations.

## Components and data flow

1. A dataset builder extracts the 18 control labels, retains eligible benchmark sentences, generates deterministic new sentences from checked-in templates, rejects lexical duplicates, and writes a JSONL manifest.
2. A neutral TTS adapter produces 22.05 kHz mono WAV prompts. The adapter records engine/version/voice in the manifest and validates every WAV before generation.
3. The existing GLM waveform backend generates prompt-only, ParaGeo, and five random variants. Random seed is a first-class CLI/config field rather than an implicit global constant.
4. The official pairwise judge runs ParaGeo directly against each random variant with randomized candidate position.
5. A dedicated analyzer reports per-attribute counts, macro preference, hierarchical 95% CI, each seed's result, pooled descriptive statistics, WER, failure inventory, and a machine-readable decision.

## Validity and stopping rules

- Dataset construction is complete only if all 18 attributes have exactly 10 unique targets and all 180 WAVs pass decoding, sample-rate, duration, and clipping checks.
- The primary claim requires the hierarchical 95% CI for ParaGeo-vs-random macro preference to lie above 50% and the result to be positive for at least four of five random seeds.
- Results that fail this rule are reported as inconclusive or negative; the experiment is never rerun with alternate seeds selected after inspection.
- Synthetic prompt audio is disclosed as a distribution-shift limitation.

## Testing

- Unit tests cover attribute balancing, normalized-text deduplication, fixed random seeds, direction norm matching, manifest resumability, and hierarchical bootstrap.
- Integration dry-run verifies the 180-item × seven-generation-variant × judge matrix.
- Final verification includes complete-count audits, WAV integrity checks, JSON parsing, `pytest -q`, and `compileall`.
