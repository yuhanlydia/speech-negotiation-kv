# Gate F/G Run Contract

This phase starts only because formal Gate E2 passed the short-horizon-value sufficiency gate. The frozen method hypothesis is now: **one-step opponent reaction is cheap supervision for selecting a long-horizon vocal strategy**.

Raw JSONL/NPZ/audio artifacts stay local and gitignored. Commit only summaries, reports, and small model metadata after each gate.

## 0. Setup

```bash
git pull origin main
source .venv/bin/activate
pip install -e '.[dev,glm]'
pytest -q
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1. Collect one-step training teacher: scenarios 30--59

Dry-run first:

```bash
python scripts/run_gate_f_teacher.py \
  --config configs/crad_gate_f_16gb.yaml \
  --scenario-start 30 --scenario-end 60 \
  --seeds 20 21 22 23 24 25 \
  --dry-run --fresh \
  --output results/gate_f_train_teacher_dryrun.jsonl
```

Real run (`30 scenarios x 6 seeds x 6 styles = 1080` one-step branches):

```bash
python scripts/run_gate_f_teacher.py \
  --config configs/crad_gate_f_16gb.yaml \
  --scenario-start 30 --scenario-end 60 \
  --seeds 20 21 22 23 24 25 \
  --fresh \
  --output results/gate_f_train_teacher.jsonl
```

If interrupted, rerun the same command without `--fresh`; completed `(scenario, seed, style)` keys are skipped.

## 2. Collect validation teacher: scenarios 60--69

```bash
python scripts/run_gate_f_teacher.py \
  --config configs/crad_gate_f_16gb.yaml \
  --scenario-start 60 --scenario-end 70 \
  --seeds 20 21 22 23 24 25 \
  --fresh \
  --output results/gate_f_validation_teacher.jsonl
```

Expected: `10 x 6 x 6 = 360` one-step branches.

## 3. Fit formal selector on CPU

Requires the existing Gate-D artifacts `results/gateA_glm_sweep.jsonl` and `results/gateA_action_audio_kv.npz` locally.

```bash
python scripts/fit_gate_f_selector.py \
  --config configs/crad_gate_f_16gb.yaml \
  --train-records results/gate_f_train_teacher.jsonl \
  --validation-records results/gate_f_validation_teacher.jsonl \
  --gate-d-records results/gateA_glm_sweep.jsonl \
  --gate-d-kv results/gateA_action_audio_kv.npz \
  --model-output results/gate_f_selector.npz \
  --summary-output results/gate_f_selector_summary.json
```

The script freezes ridge hyperparameters using validation scenarios 60--69 only. It reports matched `style_lookup`, `onehot`, and `geometry` selectors. Do not inspect CRAD 80--99 while selecting hyperparameters.

## 4. Collect robustness teacher and run Gate G1/G2

```bash
python scripts/run_gate_f_teacher.py \
  --config configs/crad_gate_f_16gb.yaml \
  --scenario-start 70 --scenario-end 80 \
  --seeds 20 21 22 23 24 25 \
  --fresh \
  --output results/gate_g_robustness_teacher.jsonl

python scripts/run_gate_g_controls.py \
  --config configs/crad_gate_f_16gb.yaml \
  --train-records results/gate_f_train_teacher.jsonl \
  --robustness-records results/gate_g_robustness_teacher.jsonl \
  --selector results/gate_f_selector.npz \
  --output results/gate_g_controls_summary.json
```

G1 compares geometry vs matched one-hot strategy heads. G2 removes one strategy from all value-training rows, keeps its Gate-D coordinate, and evaluates zero-shot value transfer on scenarios 70--79.

## 5. Final held-out terminal evaluation: CRAD 80--99

Use new seeds 30--35. Every method chooses one opening style; all later turns use the same neutral continuation policy and the Gate-E2 H=8/H=12 terminal contract.

```bash
python scripts/run_gate_f_terminal_eval.py \
  --config configs/crad_gate_f_16gb.yaml --selector results/gate_f_selector.npz \
  --method geometry --fresh --output results/gate_f_test_geometry.jsonl

python scripts/run_gate_f_terminal_eval.py \
  --config configs/crad_gate_f_16gb.yaml --selector results/gate_f_selector.npz \
  --method onehot --fresh --output results/gate_f_test_onehot.jsonl

python scripts/run_gate_f_terminal_eval.py \
  --config configs/crad_gate_f_16gb.yaml --selector results/gate_f_selector.npz \
  --method best_fixed --fresh --output results/gate_f_test_best_fixed.jsonl

python scripts/run_gate_f_terminal_eval.py \
  --config configs/crad_gate_f_16gb.yaml --selector results/gate_f_selector.npz \
  --method neutral --fresh --output results/gate_f_test_neutral.jsonl

python scripts/run_gate_f_terminal_eval.py \
  --config configs/crad_gate_f_16gb.yaml --selector results/gate_f_selector.npz \
  --method random --fresh --output results/gate_f_test_random.jsonl
```

Each file should contain `20 scenarios x 6 seeds = 120` terminal branches.

## 6. Analyze formal Gate F

```bash
python scripts/analyze_gate_f_terminal.py \
  --config configs/crad_gate_f_16gb.yaml \
  --records geometry=results/gate_f_test_geometry.jsonl \
  --records onehot=results/gate_f_test_onehot.jsonl \
  --records best_fixed=results/gate_f_test_best_fixed.jsonl \
  --records neutral=results/gate_f_test_neutral.jsonl \
  --records random=results/gate_f_test_random.jsonl \
  --output results/gate_f_terminal_summary.json
```

Formal Gate-F pass rule:

```text
paired bootstrap 95% CI lower bound of
terminal utility(geometry) - terminal utility(best_fixed) > 0
```

Do not start G4 K/V steering, OPSD, GRPO, LoRA/SFT, cross-model, or cross-domain experiments before this final held-out terminal gate passes.
