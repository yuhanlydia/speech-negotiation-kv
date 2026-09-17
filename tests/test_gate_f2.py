import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_selector_fixture(tmp_path):
    selector = tmp_path / "selector.npz"
    styles = np.asarray([
        "neutral",
        "calm and confident",
        "firm and assertive",
        "empathetic and warm",
        "urgent but controlled",
        "hesitant and uncertain",
    ])
    np.savez_compressed(
        selector,
        styles=styles,
        style_coordinates=np.zeros((len(styles), 2), dtype=np.float32),
        geometry_weights=np.zeros(1, dtype=np.float32),
        geometry_state_mean=np.zeros(1, dtype=np.float32),
        geometry_state_scale=np.ones(1, dtype=np.float32),
        onehot_weights=np.zeros(1, dtype=np.float32),
        onehot_state_mean=np.zeros(1, dtype=np.float32),
        onehot_state_scale=np.ones(1, dtype=np.float32),
        best_fixed_style=np.asarray(styles[0]),
    )
    return selector


def write_crad_fixture(tmp_path):
    data = tmp_path / "crad.csv"
    pd.DataFrame(
        [
            {
                "Creditor Name": f"Creditor {index}",
                "Debtor Name": f"Debtor {index}",
                "Creditor Target Days": 30,
                "Debtor Target Days": 120,
            }
            for index in range(100)
        ]
    ).to_csv(data, index=False)
    return data


def write_gate_f2_fixture(tmp_path, *, drop_last=False, replay_verified=True):
    config = tmp_path / "gate_f2.yaml"
    config.write_text(
        "experiment:\n"
        "  final_start: 80\n"
        "  final_end: 82\n"
        "  final_seeds: [40, 41]\n"
    )
    search_path = tmp_path / "search.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    search_rows = []
    baseline_rows = []
    for scenario_id in (80, 81):
        for seed in (40, 41):
            probes = [
                {
                    "style": f"style-{index}",
                    "eligible": True,
                    "matched_semantics": True,
                    "rendered_audio_token_ids": [index + 1],
                    "response_audio_token_ids": [index + 11],
                    "immediate_utility": index / 5,
                }
                for index in range(6)
            ]
            search_rows.append({
                "scenario_id": scenario_id,
                "seed": seed,
                "outcome": "agreement",
                "utility": 0.8,
                "rounds": 2,
                "selected_style": "style-5",
                "opening_probes": probes,
                "opening_probe_fallback_used": False,
                "opening_probe_replay_verified": replay_verified,
            })
            baseline_rows.append({
                "scenario_id": scenario_id,
                "seed": seed,
                "outcome": "agreement",
                "utility": 0.5,
                "rounds": 2,
                "selected_style": "style-3",
            })
    if drop_last:
        search_rows.pop()
    search_path.write_text("".join(json.dumps(row) + "\n" for row in search_rows))
    baseline_path.write_text("".join(json.dumps(row) + "\n" for row in baseline_rows))
    return config, search_path, baseline_path


def run_gate_f2_analyzer(tmp_path, config, search_path, baseline_path):
    output = tmp_path / "summary.json"
    command = [
        sys.executable,
        "scripts/analyze_gate_f2_terminal.py",
        "--config", str(config),
        "--records", f"immediate_search={search_path}",
        "--records", f"best_fixed={baseline_path}",
        "--bootstrap-repeats", "1000",
        "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text())


def test_gate_f2_immediate_search_cli_dry_run_records_probes_and_verified_replay(tmp_path):
    output = tmp_path / "gate_f2_dryrun.jsonl"
    selector = write_selector_fixture(tmp_path)
    data = write_crad_fixture(tmp_path)
    command = [
        sys.executable,
        "scripts/run_gate_f_terminal_eval.py",
        "--config", "configs/crad_gate_f_16gb.yaml",
        "--data", str(data),
        "--selector", str(selector),
        "--method", "immediate_search",
        "--scenario-start", "80",
        "--scenario-end", "82",
        "--seeds", "40",
        "--dry-run",
        "--fresh",
        "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["method"] == "immediate_search" for row in rows)
    assert all(len(row["opening_probes"]) == 6 for row in rows)
    assert all(row["opening_probe_replay_verified"] is True for row in rows)
    assert all(row["opening_probe_fallback_used"] is False for row in rows)


def test_gate_f2_analyzer_passes_only_complete_positive_paired_result(tmp_path):
    fixture = write_gate_f2_fixture(tmp_path)
    summary = run_gate_f2_analyzer(tmp_path, *fixture)
    assert summary["data_gate_passes"] is True
    assert summary["primary_delta"]["lower_95"] > 0
    assert summary["gate_f2_passes"] is True


def test_gate_f2_analyzer_rejects_incomplete_coverage(tmp_path):
    fixture = write_gate_f2_fixture(tmp_path, drop_last=True)
    summary = run_gate_f2_analyzer(tmp_path, *fixture)
    assert summary["data_gate_passes"] is False
    assert summary["gate_f2_passes"] is False


def test_gate_f2_analyzer_rejects_unverified_probe_replay(tmp_path):
    fixture = write_gate_f2_fixture(tmp_path, replay_verified=False)
    summary = run_gate_f2_analyzer(tmp_path, *fixture)
    assert summary["probe_quality"]["verified_replay_rate"] == 0.0
    assert summary["data_gate_passes"] is False
    assert summary["gate_f2_passes"] is False
