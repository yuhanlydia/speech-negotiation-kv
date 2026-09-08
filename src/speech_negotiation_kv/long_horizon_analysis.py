from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Any

import numpy as np

from .subspace import spearman_rank_correlation


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def style_ranking_correlations(rows: Iterable[Mapping[str, Any]], *, expected_styles: set[str],
                               outcome_key: str = "utility") -> dict:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["state_id"])].append(row)
    per_state = []
    complete_states = 0
    for state_id in sorted(groups):
        by_style = {str(row["style"]): row for row in groups[state_id]}
        if set(by_style) != set(expected_styles):
            continue
        if any(by_style[style].get(outcome_key) is None for style in expected_styles):
            continue
        complete_states += 1
        ordered = sorted(expected_styles)
        immediate = np.asarray([float(by_style[style]["immediate_offer_utility"]) for style in ordered])
        outcome = np.asarray([float(by_style[style][outcome_key]) for style in ordered])
        try:
            rho = spearman_rank_correlation(immediate, outcome)
        except ValueError:
            rho = float("nan")
        immediate_best = {ordered[index] for index in np.flatnonzero(immediate == immediate.max())}
        outcome_best = {ordered[index] for index in np.flatnonzero(outcome == outcome.max())}
        per_state.append({
            "state_id": state_id,
            "spearman": float(rho) if np.isfinite(rho) else None,
            "best_style_overlap": bool(immediate_best & outcome_best),
            "immediate_best_styles": sorted(immediate_best),
            "outcome_best_styles": sorted(outcome_best),
        })
    correlations = [row["spearman"] for row in per_state if row["spearman"] is not None]
    return {
        "n_complete_states": int(complete_states),
        "n_informative_states": len(correlations),
        "spearman": _distribution(correlations),
        "best_style_agreement_rate": (
            float(np.mean([row["best_style_overlap"] for row in per_state])) if per_state else None
        ),
        "per_state": per_state,
    }
