from __future__ import annotations

from typing import Mapping


def select_scale(scores: Mapping[float, float]) -> float:
    if not scores:
        raise ValueError("scores must be non-empty")
    clean = {float(scale): float(score) for scale, score in scores.items()}
    best_score = max(clean.values())
    return min(scale for scale, score in clean.items() if score == best_score)


def pilot_decision(*, baseline: Mapping[str, float], ours: Mapping[str, float],
                   random: Mapping[str, float], wer_degradation: Mapping[str, float],
                   static_min_gain: float = 5.0, dynamic_min_gain: float = 8.0,
                   max_wer_degradation: float = 0.02) -> dict:
    tasks = sorted(set(baseline) & set(ours) & set(random) & set(wer_degradation))
    if not tasks:
        raise ValueError("no task has baseline, ours, random, and fidelity metrics")
    rows = {}
    passing = []
    for task in tasks:
        threshold = float(dynamic_min_gain if task == "dynamic" else static_min_gain)
        gain = float(ours[task]) - float(baseline[task])
        random_gain = float(random[task]) - float(baseline[task])
        fidelity = float(wer_degradation[task]) <= float(max_wer_degradation)
        random_control = random_gain < threshold
        task_pass = bool(gain >= threshold and fidelity and random_control)
        rows[task] = {
            "baseline": float(baseline[task]),
            "ours": float(ours[task]),
            "random": float(random[task]),
            "gain": gain,
            "random_gain": random_gain,
            "required_gain": threshold,
            "wer_degradation": float(wer_degradation[task]),
            "max_wer_degradation": float(max_wer_degradation),
            "fidelity_passes": fidelity,
            "random_control_passes": random_control,
            "task_passes": task_pass,
        }
        if task_pass:
            passing.append(task)
    return {
        "tasks": rows,
        "passing_tasks": passing,
        "parageo_pilot_passes": bool(passing),
        "decision": "continue_to_full_benchmark" if passing else "stop_parageo",
    }
