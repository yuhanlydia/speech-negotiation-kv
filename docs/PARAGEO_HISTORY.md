# Research history and frozen records

[Back to ParaGeo](../README.md)

The repository began as `speech-negotiation-kv`, studying how vocal delivery affects a model opponent in debt negotiation. ParaGeo developed from its representation measurements. The repository URL, Python package name, source files, run contracts, and result history are retained for traceability.

## Read the evidence in this order

1. **Speech-channel discovery:** matched spoken openings can change an opponent's response. See [the original gate report](../results/GATE_REPORT.md).
2. **Style geometry:** the six-style [Gate D report](../results/gate_d_strategy_geometry_report.md) and [audio-position summary](../results/gate_d_action_audio_summary.json) test style identity and reproducible style changes across scenarios. These results do not use a successful-negotiation label as their target.
3. **Broader ParaGeo probe:** [80-label calibration](../results/parageo_basis_summary.json) measures requested-label structure across fixed lexical sentences.
4. **Generation comparison:** [the main summary](../results/icassp2027/summary.json) and [balanced static study](../results/icassp2027_static_power/summary.json) evaluate the resulting interventions.

## Separate questions, separate conclusions

The corrected [Gate B-prime report](../results/debug_gate_b_prime_report.md) did not establish a stable advantage-specific K/V subspace. Recoverable *style* information is not the same result as a transferable *winning strategy*.

[Gate E2](results/gate_e2_formal_v3_report.md) concerns observed first-response and terminal utilities. Its tie-aware best-set overlap is not a policy win rate. [Gate F2](results/gate_f2_terminal_report.md) did not establish a reliable terminal-utility advantage over the frozen best-fixed style. These remain historical findings; this documentation update does not relabel or rerun them.

## Reading old plans

The [original README at the experiment snapshot](https://github.com/yuhanlydia/speech-negotiation-kv/blob/b71c6cc5a1faf51f3e95245068b4c9c9241f8c0e/README.md), root-level `*_RUN.md` files, and `docs/superpowers/` preserve earlier hypotheses and execution plans. A planned experiment, preregistered threshold, or early pilot statement is not a final outcome.

Use [PARAGEO_RESULTS.md](PARAGEO_RESULTS.md) for final arm status. In particular, the formal composition development grid did **not** have zero added text WER: it reported +0.12 at every tested scale, and selected no admissible scale. The completed dynamic 0.25 arm must not be confused with the incomplete selected 1.5 arm.

No experimental source file, configuration, or result JSON was changed in preparing the public-facing documentation. The frozen result snapshot remains `b71c6cc5a1faf51f3e95245068b4c9c9241f8c0e`.
