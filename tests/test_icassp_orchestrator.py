from pathlib import Path
import importlib.util


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "run_icassp2027_all.py"
    spec = importlib.util.spec_from_file_location("icassp_orchestrator", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return module


def config():
    return {"benchmark": {"root": "/bench"},
            "parageo": {"calibration_config": "configs/pilot.yaml", "catalog": "results/catalog.json",
                         "calibration_records": "results/cal.jsonl", "calibration_kv": "results/cal.npz",
                         "basis": "results/basis.npz", "basis_summary": "results/basis.json"},
            "experiment": {"alpha_grid": [.25, .5, 1.0, 1.5]},
            "outputs": {"root": "results/icassp2027"}}


def test_dev_matrix_covers_prompt_four_alphas_judges_and_selector():
    module = load_module(); commands = module.command_matrix(config(), config_path="configs/icassp.yaml", stage="dev")
    text = "\n".join(module.shell_join(command) for command in commands)
    assert text.count("--variant prompt_only") == 3
    assert text.count("run_official_speechparaling_judge.py") == 12
    assert "select_icassp_alpha.py" in text
    for alpha in ("0.25", "0.5", "1.0", "1.5"): assert f"--alpha {alpha}" in text


def test_ablation_matrix_contains_frozen_mechanism_controls():
    module = load_module(); commands = module.command_matrix(config(), config_path="configs/icassp.yaml", stage="ablations",
        selected_alphas={"static": 1.0, "composed": .5, "dynamic": .25})
    text = "\n".join(module.shell_join(command) for command in commands)
    for variant in ("raw_basis", "rank4", "rank8", "rank32", "layers_early", "layers_middle", "layers_late"):
        assert f"--variant {variant}" in text
    assert "--variant comp_mean" in text and "--variant comp_normalized" in text
    assert "--variant dyn_start" in text and "--variant dyn_end" in text and "--variant dyn_midpoint" in text
    assert "--split ablation" in text


def test_dry_matrix_uses_placeholders_before_alpha_selection():
    module = load_module(); commands = module.command_matrix(config(), config_path="configs/icassp.yaml", stage="main")
    text = "\n".join(module.shell_join(command) for command in commands)
    assert "<SELECTED_ALPHA:static>" in text and "--variant random" in text
