from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Iterable, Mapping

import numpy as np

from .subspace import spearman_rank_correlation


def leave_one_scenario_out(scenarios: Iterable[int]) -> list[tuple[set[int], set[int]]]:
    values = sorted({int(value) for value in scenarios})
    return [(set(values) - {value}, {value}) for value in values]


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


def evaluate_state_predictions(rows: Iterable[Mapping], *, expected_styles: set[str],
                               prediction_key: str = "prediction", utility_key: str = "utility") -> dict:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[str(row["state_id"])].append(row)
    correlations: list[float] = []
    top1: list[float] = []
    complete = 0
    for state_rows in groups.values():
        by_style = {str(row["style"]): row for row in state_rows}
        if set(by_style) != set(expected_styles):
            continue
        if any(row.get(prediction_key) is None or row.get(utility_key) is None
               for row in by_style.values()):
            continue
        ordered = sorted(expected_styles)
        truth = np.asarray([float(by_style[style][utility_key]) for style in ordered])
        prediction = np.asarray([float(by_style[style][prediction_key]) for style in ordered])
        complete += 1
        rho = float(spearman_rank_correlation(prediction, truth))
        if np.isfinite(rho):
            correlations.append(rho)
        top1.append(float(np.argmax(prediction) in np.flatnonzero(truth == truth.max())))
    return {
        "n_complete_states": int(complete),
        "spearman": _distribution(correlations),
        "top1_accuracy": float(np.mean(top1)) if top1 else None,
        "per_state_count": len(correlations),
    }


def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                      *, ridge: float = 1.0) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (X_train - mean) / scale
    test = (X_test - mean) / scale
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    penalty = np.eye(train.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + float(ridge) * penalty, train.T @ y_train)
    return test @ weights


def interaction_features(state_features: np.ndarray, style_coordinates: np.ndarray) -> np.ndarray:
    state_features = np.asarray(state_features, dtype=np.float64)
    style_coordinates = np.asarray(style_coordinates, dtype=np.float64)
    if len(state_features) != len(style_coordinates):
        raise ValueError("state and style rows must align")
    return np.einsum("ij,ik->ijk", state_features, style_coordinates).reshape(len(state_features), -1)
