# ParaGeo: results and provenance

[Back to ParaGeo](../README.md) · [Reproduction guide](PARAGEO_REPRODUCIBILITY.md)

**Result snapshot:** [`b71c6cc5a1faf51f3e95245068b4c9c9241f8c0e`](https://github.com/yuhanlydia/speech-negotiation-kv/tree/b71c6cc5a1faf51f3e95245068b4c9c9241f8c0e). Values below summarize existing measurements; documentation preparation adds no model runs.

## 1. Matched-content geometry

Source: [`results/parageo_basis_summary.json`](../results/parageo_basis_summary.json).

The probe has 80 exact requested labels from 12 benchmark families and eight author-specified sentences. The intended grid has 640 cells, of which 590 yield retained representations. A classifier predicts the requested label, not words, negotiation utility, or a listener-annotated emotional state.

| Representation | Correct / records | 80-way accuracy | Same-label cross-content cosine |
|---|---:|---:|---:|
| Content-centered full K/V | 69 / 590 | 11.6949% | 0.151710 |
| Content-projected 16-D coordinates | 56 / 590 | 9.4915% | 0.285311 |

The source's top-level `leave_one_content_out_centroid_accuracy` is the **full-space** value. Projected values are under `shuffled_geometry_null.real`; do not mix these fields.

| Projected statistic | Observed | Null mean | Null 5th–95th percentiles | Conditional p |
|---|---:|---:|---:|---:|
| Centroid accuracy (%) | 9.4915 | 1.2468 | [0.5085, 2.0339] | 1/1001 |
| Cross-content cosine | 0.28531 | 0.01703 | [0.00443, 0.03117] | 1/1001 |

The 1,000 permutations shuffle labels independently within each sentence. Their percentiles are null-distribution quantiles, not confidence intervals. Centroids are held out by content, but the projection and unlabeled sentence centering use the full calibration pool.

The raw rank-16 SVD captures **61.7621%** of residual variance, before the content-associated projection. The `rank_95_energy: 32` field is capped by the configured maximum; it does not by itself establish 95% energy at rank 32. The actual uncapped rank requires the saved singular spectrum. Eight centered sentence means also have at most seven nonzero-variance directions, even though the original content-reference setting requests eight columns.

## 2. Six-style discovery

Sources: [Gate D report](../results/gate_d_strategy_geometry_report.md) and [audio-position summary](../results/gate_d_action_audio_summary.json).

This separate study uses ten CRAD scenarios, two seeds, and six styles (120 representations). Median cross-scenario decoding is **85.0%**, with 16.7% uniform chance. All 15 unique style contrasts pass the recorded BH-FDR procedure. A style-contrast cosine measures whether the *same change* repeats across scenarios; it is not similarity between opposing styles. The 126 balanced 5/5 partitions overlap, so they are not 126 independent datasets.

The six-style study motivates the broader calibration. It is not an 80-class result or evidence of a utility-improving negotiation policy.

## 3. Generation comparisons and arm status

Sources: [`results/icassp2027/summary.json`](../results/icassp2027/summary.json) and [`results/icassp2027_static_power/summary.json`](../results/icassp2027_static_power/summary.json).

Preference is `100 * mean(item score)`, where an item averages its dimension-level wins/ties/losses as `1 / 0.5 / 0`. **50 is parity with the named comparator, not 50% task accuracy.** Confidence intervals are those recorded in the summaries; no interval is newly estimated here.

| Task | Candidate vs comparator | Unique items | Preference | Recorded 95% CI | Status |
|---|---|---:|---:|---:|---|
| Static | ParaGeo vs prompt-only | 168 | 55.36 | [49.11, 61.61] | Selected alpha 0.5 |
| Static | Random direction vs prompt-only | 168 | 55.95 | [49.40, 62.50] | Matched control |
| Composition | ParaGeo vs prompt-only | 38 | 45.39 | [36.18, 54.82] | Exploratory alpha 0.25 |
| Composition | Full-space vs prompt-only | 38 | 48.03 | [37.94, 58.33] | Exploratory comparator |
| Composition | ParaGeo vs full-space | 38 | 42.98 | [34.87, 51.10] | Direct exploratory comparison |
| Dynamic | ParaGeo vs prompt-only | 60 | 51.67 | [40.00, 63.33] | Exploratory alpha 0.25 |
| Dynamic | Random direction vs prompt-only | 60 | 46.67 | [35.00, 58.33] | Exploratory control |
| Balanced static | ParaGeo vs five random-direction controls | 180 | 51.50 | [45.39, 57.56] | Pooled direct comparison |

Composition has no admissible development-selected alpha: the recorded development text-WER degradation is +0.12 at every tested scale. Its random arm is incomplete, not a zero-valued result. Dynamic selected alpha 1.5 on development, but that held-out arm is incomplete; the completed alpha-0.25 results are explicitly exploratory. These distinctions override prospective language in old run plans.

The balanced study uses 18 labels and ten unique inputs per label. The same ParaGeo output is reused across five random-direction controls: 900 comparisons are not 900 independent inputs. The full pooled interval contains parity; it does not establish a general direction-specific control advantage.

## 4. Dynamic configuration analysis

These are the fixed 24-item ablations, not the 60-item exploratory main evaluation. All preferences below use prompt-only as comparator. Intervals are nominal, without selection correction across configurations. Delta WER uses the emitted **text channel**, in absolute WER units.

| Configuration | Preference | Recorded 95% CI | Delta text WER |
|---|---:|---:|---:|
| All layers, rank 16 | 54.17 | [37.50, 70.83] | +0.0055 |
| Without content projection | 43.75 | [27.08, 60.42] | +0.0021 |
| Rank 4 | 58.33 | [41.67, 75.00] | 0.0000 |
| Rank 8 | 54.17 | [37.50, 70.83] | +0.0108 |
| Rank 32 | 47.92 | [31.25, 64.58] | +0.0021 |
| Early layers | 60.42 | [45.83, 75.00] | +0.0160 |
| Middle layers | 68.75 | [56.25, 81.25] | -0.0060 |
| Late layers | 60.42 | [45.83, 75.00] | 0.0000 |
| Start only | 56.25 | [39.58, 70.83] | +0.0010 |
| End only | 47.92 | [31.25, 66.67] | +0.0327 |
| Midpoint only | 45.83 | [29.17, 62.50] | 0.0000 |

The middle-layer point estimate is a useful configuration observation. Masking layers also changes total update energy, so it is not an energy-matched causal localization result. All static and composition variants remain available in the source summary.

## 5. Interpreting projection and attribute responses

For a prototype `p_a`, the reconstructed direction is `B B^T p_a`. Additive composition satisfies `B(c_a + c_b) = B B^T(p_a + p_b)`. The operation being tested against full-space addition is the common projection, not the existence of vector addition.

On the same 38 exploratory composition items, mean text-channel WER is **45.67%** for ParaGeo and **50.28%** for matched full-space addition, a descriptive 4.61-percentage-point difference. Prompt-only is **38.79%**. A smaller text disturbance than full-space addition is not a control-preference gain or a waveform-transcription result.

In the balanced static study, Tired and Polite average **71** and **64** against the five controls; their lowest control-specific means are **65** and **55**. Teasing averages **65**, with four above-parity means and one tie. These are descriptive selections from all 18 labels, not independent confirmatory replications. Read them alongside the full pooled comparison and the complete source matrix.

## Scope of the release

The recoverable structure is measured in replay-conditioned representations of one quantized backbone. Requested labels, globally fitted preprocessing, model-based judging, and text-channel fidelity define the measurement protocol. The public summaries do not recover raw latent point clouds, the full null distributions, or waveform trajectories. See [artifact and measurement notes](PARAGEO_REPRODUCIBILITY.md#artifact-and-measurement-notes) before reconstructing figures or extending the study.
