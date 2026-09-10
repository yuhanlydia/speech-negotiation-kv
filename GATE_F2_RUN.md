# Gate F2 Immediate-Search Run Contract

Gate F2 is a new-seed replication of short-horizon test-time strategy search.
It does not replace the negative pre-response selector result from Gate F.

## Frozen protocol

- CRAD scenarios: 80--99
- Seeds: 40--45
- Styles: the six styles in `configs/crad_gate_f2_16gb.yaml`
- Search: probe every style for one opponent response, choose maximum immediate
  creditor utility with fixed config-order tie-breaking, and continue only that
  selected style to terminal
- Baseline: corrected Gate-F artifact's frozen best-fixed style
- PASS: paired bootstrap 95% CI lower bound of
  `terminal utility(immediate_search) - terminal utility(best_fixed) > 0`

## Verification

```bash
source .venv/bin/activate
pytest -q
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Immediate-search method

Start with a new output file:

```bash
python scripts/run_gate_f_terminal_eval.py \
  --config configs/crad_gate_f2_16gb.yaml \
  --selector results/gate_f_selector_soft_reparsed_v3.npz \
  --method immediate_search \
  --fresh \
  --output results/gate_f2_immediate_search.jsonl
```

If interrupted, rerun the same command without `--fresh`. Completed
`(scenario, seed)` keys are skipped.

## Frozen best-fixed baseline

```bash
python scripts/run_gate_f_terminal_eval.py \
  --config configs/crad_gate_f2_16gb.yaml \
  --selector results/gate_f_selector_soft_reparsed_v3.npz \
  --method best_fixed \
  --fresh \
  --output results/gate_f2_best_fixed.jsonl
```

Resume without `--fresh` after interruption.

## Analysis

```bash
python scripts/analyze_gate_f2_terminal.py \
  --config configs/crad_gate_f2_16gb.yaml \
  --records immediate_search=results/gate_f2_immediate_search.jsonl \
  --records best_fixed=results/gate_f2_best_fixed.jsonl \
  --bootstrap-repeats 10000 \
  --output results/gate_f2_terminal_summary.json
```

Do not change the seeds, tie-breaking, baseline, bootstrap threshold, or
terminal protocol after inspecting results. F3 probe-budget experiments are
authorized only after a valid Gate-F2 PASS.
