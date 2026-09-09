from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .subspace import spearman_rank_correlation


def _distribution(values: list[float]) -> dict[str, Any]:
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


def terminal_state_metrics(
    rows: Iterable[Mapping[str, Any]], *, expected_styles: set[str]
) -> dict[str, Any]:
    """Compare immediate and terminal style rankings within complete states.

    A state is complete only when every expected style has a policy-valid,
    terminal agreement/no-deal row with finite immediate and terminal utility.
    Best-style agreement is tie-aware.  If immediate best is tied, regret is
    the terminal oracle utility minus the mean terminal utility of those tied
    immediate-best styles.
    """
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("policy_valid"):
            continue
        if row.get("outcome") not in {"agreement", "no_deal"}:
            continue
        groups[str(row["state_id"])].append(row)

    per_state: list[dict[str, Any]] = []
    for state_id in sorted(groups):
        by_style = {str(row["style"]): row for row in groups[state_id]}
        if set(by_style) != set(expected_styles):
            continue
        if any(
            row.get("immediate_offer_utility") is None or row.get("utility") is None
            for row in by_style.values()
        ):
            continue
        ordered = sorted(expected_styles)
        immediate = np.asarray(
            [float(by_style[style]["immediate_offer_utility"]) for style in ordered],
            dtype=np.float64,
        )
        terminal = np.asarray(
            [float(by_style[style]["utility"]) for style in ordered], dtype=np.float64
        )
        if not np.all(np.isfinite(immediate)) or not np.all(np.isfinite(terminal)):
            continue
        immediate_best = set(np.asarray(ordered)[np.flatnonzero(immediate == immediate.max())])
        terminal_best = set(np.asarray(ordered)[np.flatnonzero(terminal == terminal.max())])
        try:
            spearman = float(spearman_rank_correlation(immediate, terminal))
        except ValueError:
            spearman = float("nan")
        immediate_best_indices = [index for index, style in enumerate(ordered) if style in immediate_best]
        regret = float(terminal.max() - terminal[immediate_best_indices].mean())
        per_state.append({
            "state_id": state_id,
            "spearman": spearman if np.isfinite(spearman) else None,
            "best_style_agreement": bool(immediate_best & terminal_best),
            "immediate_best_styles": sorted(immediate_best),
            "terminal_best_styles": sorted(terminal_best),
            "regret": regret,
        })

    correlations = [
        float(row["spearman"])
        for row in per_state
        if row["spearman"] is not None
    ]
    regrets = [float(row["regret"]) for row in per_state]
    return {
        "n_complete_states": len(per_state),
        "n_informative_states": len(correlations),
        "spearman": _distribution(correlations),
        "top1_agreement_rate": (
            float(np.mean([row["best_style_agreement"] for row in per_state]))
            if per_state else None
        ),
        "regret": _distribution(regrets),
        "per_state": per_state,
    }
