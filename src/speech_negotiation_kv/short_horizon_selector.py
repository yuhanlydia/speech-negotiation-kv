from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def teacher_distribution(utilities: np.ndarray, *, temperature: float = 0.1) -> np.ndarray:
    values = np.asarray(utilities, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("utilities must be a non-empty 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("utilities must be finite")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    advantage = values - values.mean()
    logits = advantage / float(temperature)
    logits -= logits.max()
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def state_spread_weights(utility_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(utility_matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("utility_matrix must be 2D with states x styles")
    return values.max(axis=1) - values.min(axis=1)


def opening_state_features(creditor_target: float, debtor_target: float) -> np.ndarray:
    creditor = float(creditor_target)
    debtor = float(debtor_target)
    if creditor <= 0 or debtor <= 0 or debtor <= creditor:
        raise ValueError("require 0 < creditor_target < debtor_target")
    scale = max(debtor, 365.0)
    gap = debtor - creditor
    midpoint = 0.5 * (creditor + debtor)
    return np.asarray([
        creditor / scale,
        debtor / scale,
        gap / scale,
        midpoint / scale,
        creditor / debtor,
    ], dtype=np.float64)


def prototype_style_coordinates(
    features: np.ndarray,
    style_labels: Sequence[str],
    styles: Sequence[str],
    *,
    rank: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    X = np.asarray(features, dtype=np.float64)
    labels = np.asarray(style_labels, dtype=object)
    ordered = [str(style) for style in styles]
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError("features and style_labels must have aligned rows")
    prototypes = []
    for style in ordered:
        mask = labels == style
        if not np.any(mask):
            raise ValueError(f"missing style {style!r}")
        prototypes.append(X[mask].mean(axis=0))
    matrix = np.asarray(prototypes, dtype=np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    max_rank = min(len(ordered) - 1, vt.shape[0], vt.shape[1])
    used_rank = max_rank if rank is None else min(max(1, int(rank)), max_rank)
    coordinates = matrix @ vt[:used_rank].T
    return (
        {style: coordinates[index].copy() for index, style in enumerate(ordered)},
        {
            "rank": int(used_rank),
            "styles": ordered,
            "singular_values": singular_values[:used_rank].tolist(),
            "ambient_dim": int(X.shape[1]),
        },
    )


def _center_targets_by_state(utilities: np.ndarray, state_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(utilities, dtype=np.float64)
    ids = np.asarray(state_ids, dtype=object)
    if y.ndim != 1 or len(y) != len(ids):
        raise ValueError("utilities and state_ids must have aligned rows")
    centered = np.empty_like(y)
    row_weights = np.empty_like(y)
    for state in dict.fromkeys(ids.tolist()):
        mask = ids == state
        values = y[mask]
        centered[mask] = values - values.mean()
        row_weights[mask] = values.max() - values.min()
    return centered, row_weights


def _fit_state_standardizer(state_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(state_features, dtype=np.float64)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _interaction_design(
    state_features: np.ndarray,
    style_vectors: np.ndarray,
    *,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
) -> np.ndarray:
    phi = np.asarray(state_features, dtype=np.float64)
    c = np.asarray(style_vectors, dtype=np.float64)
    if phi.ndim != 2 or c.ndim != 2 or len(phi) != len(c):
        raise ValueError("state_features and style_vectors must be aligned 2D arrays")
    normalized = (phi - state_mean) / state_scale
    augmented = np.column_stack([np.ones(len(normalized)), normalized])
    return np.einsum("ij,ik->ijk", augmented, c).reshape(len(phi), -1)


def _weighted_ridge(X: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    row_weights = np.asarray(weights, dtype=np.float64)
    if np.any(row_weights < 0):
        raise ValueError("weights must be non-negative")
    if not np.any(row_weights > 0):
        row_weights = np.ones_like(row_weights)
    sqrt_w = np.sqrt(row_weights)[:, None]
    Xw = X * sqrt_w
    yw = y * sqrt_w[:, 0]
    penalty = np.eye(X.shape[1], dtype=np.float64) * float(ridge)
    return np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw)


def fit_geometry_selector(
    state_features: np.ndarray,
    style_coordinates: np.ndarray,
    utilities: np.ndarray,
    state_ids: Sequence[str],
    *,
    ridge: float = 1.0,
) -> dict[str, Any]:
    phi = np.asarray(state_features, dtype=np.float64)
    coords = np.asarray(style_coordinates, dtype=np.float64)
    y, weights = _center_targets_by_state(utilities, state_ids)
    mean, scale = _fit_state_standardizer(phi)
    design = _interaction_design(phi, coords, state_mean=mean, state_scale=scale)
    coefficients = _weighted_ridge(design, y, weights, ridge)
    return {
        "kind": "geometry",
        "weights": coefficients,
        "state_mean": mean,
        "state_scale": scale,
        "state_dim": int(phi.shape[1]),
        "style_dim": int(coords.shape[1]),
        "ridge": float(ridge),
        "parameter_count": int(coefficients.size),
    }


def score_geometry_selector(
    model: Mapping[str, Any], state_features: np.ndarray, style_coordinates: np.ndarray
) -> np.ndarray:
    design = _interaction_design(
        state_features,
        style_coordinates,
        state_mean=np.asarray(model["state_mean"], dtype=np.float64),
        state_scale=np.asarray(model["state_scale"], dtype=np.float64),
    )
    return design @ np.asarray(model["weights"], dtype=np.float64)


def _onehot(style_labels: Sequence[str], styles: Sequence[str]) -> np.ndarray:
    ordered = [str(style) for style in styles]
    index = {style: i for i, style in enumerate(ordered)}
    matrix = np.zeros((len(style_labels), len(ordered)), dtype=np.float64)
    for row, label in enumerate(style_labels):
        label = str(label)
        if label not in index:
            raise ValueError(f"unknown style {label!r}")
        matrix[row, index[label]] = 1.0
    return matrix


def fit_onehot_selector(
    state_features: np.ndarray,
    style_labels: Sequence[str],
    utilities: np.ndarray,
    state_ids: Sequence[str],
    *,
    styles: Sequence[str],
    ridge: float = 1.0,
) -> dict[str, Any]:
    ordered = [str(style) for style in styles]
    model = fit_geometry_selector(
        state_features,
        _onehot(style_labels, ordered),
        utilities,
        state_ids,
        ridge=ridge,
    )
    model["kind"] = "onehot"
    model["styles"] = ordered
    return model


def score_onehot_selector(
    model: Mapping[str, Any], state_features: np.ndarray, style_labels: Sequence[str]
) -> np.ndarray:
    return score_geometry_selector(
        model, state_features, _onehot(style_labels, model["styles"])
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def evaluate_selector_states(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_styles: set[str],
    prediction_key: str = "prediction",
    utility_key: str = "utility",
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["state_id"])].append(row)
    correlations: list[float] = []
    regrets: list[float] = []
    selected_utilities: list[float] = []
    top1: list[float] = []
    complete = 0
    per_state = []
    for state_id, state_rows in sorted(groups.items()):
        by_style = {str(row["style"]): row for row in state_rows}
        if set(by_style) != set(expected_styles):
            continue
        ordered = sorted(expected_styles)
        if any(by_style[s].get(prediction_key) is None or by_style[s].get(utility_key) is None for s in ordered):
            continue
        prediction = np.asarray([float(by_style[s][prediction_key]) for s in ordered], dtype=np.float64)
        truth = np.asarray([float(by_style[s][utility_key]) for s in ordered], dtype=np.float64)
        if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
            continue
        complete += 1
        rho = _spearman(prediction, truth)
        if np.isfinite(rho):
            correlations.append(rho)
        predicted_best = np.flatnonzero(np.isclose(prediction, prediction.max(), rtol=1e-10, atol=1e-12))
        truth_best = np.flatnonzero(np.isclose(truth, truth.max(), rtol=1e-10, atol=1e-12))
        chosen_utility = float(truth[predicted_best].mean())
        regret = float(truth.max() - chosen_utility)
        selected_utilities.append(chosen_utility)
        regrets.append(regret)
        agreement = bool(set(predicted_best.tolist()) & set(truth_best.tolist()))
        top1.append(float(agreement))
        per_state.append({
            "state_id": state_id,
            "spearman": rho if np.isfinite(rho) else None,
            "top1_agreement": agreement,
            "regret": regret,
            "selected_utility": chosen_utility,
        })
    return {
        "n_complete_states": int(complete),
        "n_informative_states": len(correlations),
        "spearman": _distribution(correlations),
        "top1_agreement_rate": float(np.mean(top1)) if top1 else None,
        "regret": _distribution(regrets),
        "selected_utility": _distribution(selected_utilities),
        "per_state": per_state,
    }


def paired_bootstrap_mean_delta(
    method: Sequence[float], baseline: Sequence[float], *, repeats: int = 10000, seed: int = 4242424242
) -> dict[str, Any]:
    a = np.asarray(method, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("method and baseline must be aligned non-empty 1D arrays")
    delta = a - b
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(delta), size=(int(repeats), len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "n": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "repeats": int(repeats),
    }


def pairwise_unseen_style_accuracy(
    rows: Iterable[Mapping[str, Any]], *, held_out_style: str, seen_styles: set[str],
    prediction_key: str = "prediction", utility_key: str = "utility",
) -> float | None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["state_id"])].append(row)
    correct = []
    for state_rows in groups.values():
        by_style = {str(row["style"]): row for row in state_rows}
        if held_out_style not in by_style:
            continue
        unseen = by_style[held_out_style]
        for style in seen_styles:
            if style not in by_style:
                continue
            truth_delta = float(unseen[utility_key]) - float(by_style[style][utility_key])
            pred_delta = float(unseen[prediction_key]) - float(by_style[style][prediction_key])
            if abs(truth_delta) < 1e-12:
                continue
            correct.append(float(np.sign(truth_delta) == np.sign(pred_delta)))
    return float(np.mean(correct)) if correct else None
