import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "run_static_power_experiment.py"
    spec = importlib.util.spec_from_file_location("run_static_power_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def config():
    return {
        "experiment": {
            "alpha": 0.5,
            "random_seeds": [11, 22, 33, 44, 55],
        },
        "dataset": {
            "manifest": "results/power/dataset.jsonl",
            "input_audio_dir": "results/power/input_audio",
            "tts_voice": "voice",
            "sample_rate": 22050,
        },
        "parageo": {"base_config": "configs/base.yaml"},
        "outputs": {"root": "results/power"},
    }


def test_generation_matrix_has_prompt_main_and_five_fixed_random_seeds():
    module = load_module()
    commands = module.command_matrix(config(), stage="generate")
    text = "\n".join(module.shell_join(command) for command in commands)

    assert len(commands) == 7
    assert text.count("--variant prompt_only") == 1
    assert text.count("--variant main") == 1
    assert text.count("--variant random") == 5
    for seed in (11, 22, 33, 44, 55):
        assert f"--random-seed {seed}" in text
        assert f"random_seed_{seed}" in text


def test_judge_matrix_is_five_direct_main_vs_random_comparisons():
    module = load_module()
    commands = module.command_matrix(config(), stage="judge")
    text = "\n".join(module.shell_join(command) for command in commands)

    assert len(commands) == 5
    assert text.count("--candidate-name main") == 5
    assert text.count("--baseline-name random_seed_") == 5
    assert "prompt_only" not in text
    assert text.count("--prompt-jsonl results/power/dataset.jsonl") == 5
