# ICASSP 2027 ParaGeo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ParaGeo repository directly runnable for all ICASSP 2027 main experiments, fixed ablations, official SpeechParaling judging, statistical aggregation, and LaTeX table export.

**Architecture:** Keep the existing pilot path untouched for provenance. Add an ICASSP-specific experiment layer around the existing ParaGeo geometry and official GLM waveform backend. Generalized generation reads one frozen config, reconstructs rank/basis/layer/composition/dynamic variants from the saved calibration artifact, official-judge wrappers reuse SpeechParaling's own judge code, and one summarizer exports JSON plus paper-ready LaTeX tables.

**Tech Stack:** Python 3.10+, NumPy, PyYAML, PyTorch/Transformers, GLM-4-Voice official assets, SpeechParaling-Bench official judge code, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-icassp2027-parageo-design.md`

## Global Constraints

- Main backbone remains `zai-org/glm-4-voice-9b` with int4/NF4 and official Whisper-VQ + Flow/HiFT waveform path.
- Development split is `eligible_index % 5 == 0`; held-out is the complement.
- Alpha grid is exactly `{0.25,0.5,1.0,1.5}` and held-out outcomes never select hyperparameters.
- Main rank is 16; rank ablations are 4, 8, 16, 32.
- Main layers are `[16,20,24,28,32,36]`.
- Ablation judging uses the first 24 benchmark-index-sorted held-out eligible items per task.
- All official judge comparisons use prompt-only GLM as the baseline.
- Existing negative CRAD Gate F/F2 results remain unchanged.

---

### Task 1: Robust geometry diagnostics and reusable basis artifact

**Files:**
- Modify: `src/speech_negotiation_kv/parageo.py`
- Modify: `scripts/fit_parageo_basis.py`
- Modify: `tests/test_parageo.py`

**Interfaces:**
- Produces `leave_one_content_out_centroid_accuracy(...) -> float`.
- Produces `same_attribute_cross_content_cosine(...) -> dict`.
- Basis NPZ additionally stores `raw_basis_full`, `attribute_prototypes`, and `content_means` while retaining existing fields.

- [ ] Add tests with synthetic content/attribute geometry proving LOCO decoding is above chance and same-attribute cosine is high.
- [ ] Run `pytest tests/test_parageo.py -q` and verify new tests fail before implementation.
- [ ] Implement LOCO decoding and cross-content cosine diagnostics.
- [ ] Change basis fitting to estimate up to rank 32, retain rank-16 main fields, and save ambient attribute prototypes needed to reconstruct rank/raw-basis ablations.
- [ ] Run `pytest tests/test_parageo.py -q` and `python -m py_compile src/speech_negotiation_kv/parageo.py scripts/fit_parageo_basis.py`.

### Task 2: Frozen ICASSP config and variant reconstruction

**Files:**
- Create: `configs/icassp2027_parageo.yaml`
- Create: `src/speech_negotiation_kv/icassp_variants.py`
- Create: `tests/test_icassp_variants.py`

**Interfaces:**
- Produces `load_geometry_variant(npz_path, variant, rank, layer_set) -> GeometryVariant`.
- Produces `compose_variant_coordinate(coords, names, mode) -> ndarray`.
- Produces `dynamic_variant_schedule(start, end, variant, steps, transition_at) -> ndarray`.

- [ ] Write tests for orthogonal vs raw basis, rank slicing, layer chunk selection, composition sum/mean/normalized modes, and dynamic scheduled/start/end/midpoint modes.
- [ ] Run focused tests and verify failure.
- [ ] Implement the variant utilities without changing historical pilot behavior.
- [ ] Add frozen task paths, layer sets, ablation count, alpha grid, bootstrap repeats, and output roots to the ICASSP config.
- [ ] Run focused tests and compile the new module.

### Task 3: Generalized ICASSP waveform generation

**Files:**
- Create: `scripts/run_icassp_generation.py`
- Modify: `src/speech_negotiation_kv/speechparaling.py`
- Create: `tests/test_icassp_generation.py`

**Interfaces:**
- CLI accepts `--task static|composed|dynamic`, `--split dev|heldout|ablation`, `--variant`, `--alpha`, and writes WAVs plus `manifest.jsonl`.
- Ablation split is deterministic first-24 held-out eligible items after benchmark-index sorting.

- [ ] Add tests for deterministic dev/held-out/ablation selection and variant validation without loading GLM weights.
- [ ] Verify tests fail.
- [ ] Implement generalized generation by reusing `GLMOfficialWaveformBackend` and `ScheduledFusedQKVSteerer`.
- [ ] Ensure prompt-only, random, orthogonal/raw rank variants, layer variants, composition variants, and dynamic variants share identical generation seeds and benchmark items.
- [ ] Run focused tests and compile the runner.

### Task 4: Official SpeechParaling judge wrapper

**Files:**
- Create: `scripts/run_official_speechparaling_judge.py`
- Create: `tests/test_official_judge_wrapper.py`

**Interfaces:**
- Dynamically loads the upstream English judge module for `static`, `composed`, or `dynamic`.
- Overrides only runtime path globals; it never edits the external SpeechParaling repository.
- Writes official judge metadata to the requested output directory.

- [ ] Test task-to-upstream-module mapping and exact candidate/baseline filename validation with temporary files.
- [ ] Verify tests fail.
- [ ] Implement dynamic import, temporary working-directory switching, path overrides, and call to upstream `evaluate(candidate_name, baseline_name)`.
- [ ] Run focused tests and compile the wrapper.

### Task 5: Pairwise statistics, fidelity, and LaTeX export

**Files:**
- Create: `src/speech_negotiation_kv/icassp_eval.py`
- Create: `scripts/summarize_icassp2027.py`
- Create: `tests/test_icassp_eval.py`

**Interfaces:**
- `read_official_metadata(path) -> list[dict]`.
- `pairwise_preference(rows) -> dict` maps win/tie/loss to 1/0.5/0.
- `bootstrap_preference(rows, repeats, seed) -> dict` returns mean score and 95% CI.
- Summarizer writes `results/icassp2027/summary.json` and `paper/generated/{main_results,geometry_table,ablation_table,result_macros}.tex`.

- [ ] Add synthetic metadata tests for single- and multi-dimension judge formats, bootstrap reproducibility, and LaTeX escaping.
- [ ] Verify tests fail.
- [ ] Implement aggregation and paired preference statistics.
- [ ] Merge text-channel fidelity JSON with judge results and enforce WER/random controls.
- [ ] Implement deterministic LaTeX table and macro rendering.
- [ ] Run focused tests and compile.

### Task 6: One-command experiment orchestration

**Files:**
- Create: `scripts/run_icassp2027_all.py`
- Create: `ICASSP2027_RUN.md`
- Modify: `README.md`
- Create: `tests/test_icassp_orchestrator.py`

**Interfaces:**
- `python scripts/run_icassp2027_all.py --stage geometry|dev|main|ablations|judge|summarize|all --dry-run`.
- Dry-run prints exact commands without model/API calls.
- Real mode executes subprocess stages and fails fast on missing required artifacts.

- [ ] Test dry-run command graph for all stages and verify it contains the frozen main/ablation variants.
- [ ] Verify tests fail.
- [ ] Implement staged orchestration with resume-by-existing-artifact behavior where safe.
- [ ] Document environment variables, expected external repos, judge API configuration, artifact paths, and the exact one-command agent workflow.
- [ ] Update README so `ICASSP2027_RUN.md` is the current execution path while historical ParaGeo/CRAD records remain linked.
- [ ] Run focused tests, full `pytest -q`, and compile all new scripts.

### Task 7: ICASSP 2027 paper package

**Files:**
- Create: `paper/ICASSP2027_ParaGeo_draft.tex`
- Create: `paper/icassp2027_parageo_refs.bib`
- Create: `paper/README.md`

**Interfaces:**
- Draft uses the uploaded ICASSP 2027 `spconf` template conventions.
- Contains multiple title options, three substantially different abstract options, three substantially different introduction options, one recommended Experimental Setup, and optional `\input{generated/...}` hooks.

- [ ] Write 6--8 title options emphasizing geometry/composition/dynamics rather than generic activation steering.
- [ ] Write three 100--150-word abstracts: geometry-first, benchmark-first, and training-free-control-first.
- [ ] Write three introductions with distinct rhetorical structures but consistent factual claims and citations.
- [ ] Write Experimental Setup covering backbone, official waveform path, calibration, benchmark/tasks, frozen split/alpha, baselines, judge, fidelity, statistics, and ablations.
- [ ] Add BibTeX entries for GLM-4-Voice, SpeechParaling-Bench, EmoShift, EmoSteer-TTS, and CoCoEmo.
- [ ] Validate LaTeX structure against the supplied template and compile if a LaTeX engine is available.

### Task 8: Final verification

**Files:** all files above.

- [ ] Run `pytest -q` and require zero failures.
- [ ] Run `python -m compileall -q src scripts`.
- [ ] Run the orchestrator with `--stage all --dry-run` and inspect command coverage.
- [ ] Confirm no raw WAV/NPZ/model weights were committed.
- [ ] Compare the final GitHub head against the pre-ICASSP head and report exact files changed.