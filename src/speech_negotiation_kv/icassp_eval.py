from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


def read_official_metadata(path: str | Path) -> list[dict]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    rows = []
    for file in sorted(root.rglob("*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        if payload.get("status") == "Success":
            rows.append(payload)
    return rows


def _outcome_score(winner_position: int, candidate_position: int) -> float | None:
    winner = int(winner_position)
    candidate = int(candidate_position)
    if winner < 0:
        return None
    if winner == 0:
        return 0.5
    return 1.0 if winner == candidate else 0.0


def sample_preference_scores(row: Mapping) -> list[float]:
    if row.get("status") not in {None, "Success"}:
        return []
    candidate = row.get("candidate_position")
    if candidate is None:
        return []
    if "winner_position" in row:
        value = _outcome_score(row["winner_position"], candidate)
        return [] if value is None else [value]
    keys = [key for key in row if re.fullmatch(r"winner_position_\d+", str(key))]
    keys.sort(key=lambda key: int(str(key).rsplit("_", 1)[-1]))
    values = [_outcome_score(row[key], candidate) for key in keys]
    return [float(value) for value in values if value is not None]


def pairwise_preference(rows: Iterable[Mapping]) -> dict:
    sample_scores = []
    decisions = []
    wins = ties = losses = 0
    for row in rows:
        values = sample_preference_scores(row)
        if not values:
            continue
        sample_scores.append(float(np.mean(values)))
        decisions.extend(values)
        wins += sum(value == 1.0 for value in values)
        ties += sum(value == 0.5 for value in values)
        losses += sum(value == 0.0 for value in values)
    if not sample_scores:
        return {
            "n_samples": 0, "n_decisions": 0, "preference_score": None,
            "gain_vs_tie": None, "wins": 0, "ties": 0, "losses": 0,
            "sample_scores": [],
        }
    score = 100.0 * float(np.mean(sample_scores))
    return {
        "n_samples": len(sample_scores), "n_decisions": len(decisions),
        "preference_score": score, "gain_vs_tie": score - 50.0,
        "wins": int(wins), "ties": int(ties), "losses": int(losses),
        "sample_scores": sample_scores,
    }


def bootstrap_preference(rows: Iterable[Mapping], *, repeats: int = 10000,
                         seed: int = 14242424242) -> dict:
    result = pairwise_preference(rows)
    scores = np.asarray(result["sample_scores"], dtype=np.float64)
    if not len(scores):
        return {**result, "lower_95": None, "upper_95": None,
                "gain_lower_95": None, "gain_upper_95": None, "repeats": int(repeats)}
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        sampled = scores[rng.integers(0, len(scores), size=len(scores))]
        boot[index] = 100.0 * sampled.mean()
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {**result, "lower_95": float(lower), "upper_95": float(upper),
            "gain_lower_95": float(lower - 50.0), "gain_upper_95": float(upper - 50.0),
            "repeats": int(repeats)}


def preference_by_dimension(rows: Iterable[Mapping]) -> dict[str, dict]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        dimensions = [str(value) for value in row.get("dimensions", [])]
        candidate = row.get("candidate_position")
        if candidate is None:
            continue
        if "winner_position" in row:
            values = sample_preference_scores(row)
            if values:
                name = dimensions[0] if dimensions else "Overall"
                buckets.setdefault(name, []).append(values[0])
            continue
        for index, dimension in enumerate(dimensions, start=1):
            key = f"winner_position_{index}"
            if key not in row:
                continue
            value = _outcome_score(row[key], candidate)
            if value is not None:
                buckets.setdefault(dimension, []).append(value)
    return {dimension: {"n": len(values), "preference_score": 100.0 * float(np.mean(values))}
            for dimension, values in sorted(buckets.items()) if values}


def read_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def paired_wer_degradation(candidate_manifest: str | Path, baseline_manifest: str | Path) -> dict:
    candidate = {str(row["item_id"]): row for row in read_manifest(candidate_manifest)}
    baseline = {str(row["item_id"]): row for row in read_manifest(baseline_manifest)}
    keys = sorted(set(candidate) & set(baseline))
    if not keys:
        return {"n": 0, "candidate_mean_wer": None, "baseline_mean_wer": None, "mean_degradation": None}
    cand = np.asarray([float(candidate[key]["text_channel_wer"]) for key in keys])
    base = np.asarray([float(baseline[key]["text_channel_wer"]) for key in keys])
    return {"n": len(keys), "candidate_mean_wer": float(cand.mean()),
            "baseline_mean_wer": float(base.mean()), "mean_degradation": float((cand - base).mean())}


def choose_alpha(candidates: Sequence[Mapping], *, max_wer_degradation: float = 0.02) -> dict:
    valid = [row for row in candidates if row.get("preference_score") is not None
             and row.get("wer_degradation") is not None
             and float(row["wer_degradation"]) <= float(max_wer_degradation)]
    if not valid:
        raise ValueError("no alpha satisfies the frozen WER gate")
    valid.sort(key=lambda row: (-float(row["preference_score"]), float(row["alpha"])))
    return dict(valid[0])


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in text)


def format_ci(result: Mapping, *, digits: int = 1) -> str:
    lower, upper = result.get("lower_95"), result.get("upper_95")
    if lower is None or upper is None:
        return "--"
    return f"[{float(lower):.{digits}f}, {float(upper):.{digits}f}]"


def _fmt(value: object, digits: int = 1) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def render_main_results_table(summary: Mapping) -> str:
    labels = {"static": "Static", "composed": "Composition", "dynamic": "Dynamic"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Held-out SpeechParaling-Bench results. Preference is the percentage of pairwise decisions favoring ParaGeo over prompt-only, counting ties as 0.5.}",
             r"\label{tab:main}", r"\begin{tabular}{lcccc}", r"\hline",
             r"Task & ParaGeo pref. & 95\% CI & Random pref. & $\Delta$WER \\", r"\hline"]
    for task in ("static", "composed", "dynamic"):
        row = summary.get("tasks", {}).get(task, {})
        main, random, fidelity = row.get("main", {}), row.get("random", {}), row.get("fidelity", {})
        lines.append(f"{labels[task]} & {_fmt(main.get('preference_score'))} & {format_ci(main)} & "
                     f"{_fmt(random.get('preference_score'))} & {_fmt(fidelity.get('mean_degradation'), 3)} \\\\ ")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def render_geometry_table(summary: Mapping) -> str:
    geo = summary.get("geometry", {})
    cosine = geo.get("same_attribute_cross_content_cosine", {})
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Calibration-only geometry diagnostics.}",
             r"\label{tab:geometry}", r"\begin{tabular}{ccccc}", r"\hline",
             r"Attr. & LOCO acc. & Chance & Cross-content cos. & EV@16 \\", r"\hline",
             f"{int(geo.get('attributes', 0))} & {_fmt(100 * geo.get('leave_one_content_out_centroid_accuracy', 0.0))} & "
             f"{_fmt(100 * geo.get('chance', 0.0))} & {_fmt(cosine.get('mean'), 3)} & "
             f"{_fmt(100 * geo.get('explained_variance_at_main_rank', 0.0))} \\\\ ",
             r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def render_ablation_table(summary: Mapping) -> str:
    ablations = summary.get("ablations", {})
    order = [("main", "ParaGeo"), ("raw_basis", "w/o semantic orthog."), ("rank4", "rank 4"),
             ("rank8", "rank 8"), ("rank32", "rank 32"), ("layers_early", "early layers"),
             ("layers_middle", "middle layers"), ("layers_late", "late layers"),
             ("comp_mean", "composition: mean"), ("comp_normalized", "composition: normalized"),
             ("dyn_start", "dynamic: start only"), ("dyn_end", "dynamic: end only"),
             ("dyn_midpoint", "dynamic: midpoint")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Fixed-subset ablations (pairwise preference over prompt-only; higher is better).}",
             r"\label{tab:ablation}", r"\begin{tabular}{lccc}", r"\hline",
             r"Variant & Static & Composition & Dynamic \\", r"\hline"]
    for key, label in order:
        vals, present = [], False
        for task in ("static", "composed", "dynamic"):
            value = ablations.get(task, {}).get(key, {}).get("preference_score")
            vals.append(_fmt(value)); present = present or value is not None
        if present:
            lines.append(f"{latex_escape(label)} & {vals[0]} & {vals[1]} & {vals[2]} \\\\ ")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def render_result_macros(summary: Mapping) -> str:
    tasks, geometry = summary.get("tasks", {}), summary.get("geometry", {})
    cosine = geometry.get("same_attribute_cross_content_cosine", {})
    macros = {"StaticPref": tasks.get("static", {}).get("main", {}).get("preference_score"),
              "StaticGain": tasks.get("static", {}).get("main", {}).get("gain_vs_tie"),
              "CompPref": tasks.get("composed", {}).get("main", {}).get("preference_score"),
              "CompGain": tasks.get("composed", {}).get("main", {}).get("gain_vs_tie"),
              "DynPref": tasks.get("dynamic", {}).get("main", {}).get("preference_score"),
              "DynGain": tasks.get("dynamic", {}).get("main", {}).get("gain_vs_tie"),
              "GeometryLOCO": 100 * geometry.get("leave_one_content_out_centroid_accuracy", 0.0),
              "GeometryChance": 100 * geometry.get("chance", 0.0),
              "GeometryCosine": cosine.get("mean")}
    lines = ["% Auto-generated; do not edit by hand."]
    for name, value in macros.items():
        text = "TBD" if value is None else (f"{float(value):.3f}" if name == "GeometryCosine" else f"{float(value):.1f}")
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    lines.append(f"\\newcommand{{\\ParaGeoDecision}}{{{latex_escape(summary.get('decision', 'TBD'))}}}")
    return "\n".join(lines) + "\n"
