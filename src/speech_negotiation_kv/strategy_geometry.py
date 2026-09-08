from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


def _as_aligned(features: np.ndarray, *arrays: Iterable) -> tuple[np.ndarray, ...]:
    features = np.asarray(features, dtype=np.float64)
    aligned = (features, *(np.asarray(list(values)) for values in arrays))
    if features.ndim != 2 or any(len(values) != len(features) for values in aligned[1:]):
        raise ValueError("features must be 2D and all metadata arrays must align with its rows")
    return aligned


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 1e-12)


def center_within_states(features: np.ndarray, state_ids: Iterable[str]) -> np.ndarray:
    features, state_ids = _as_aligned(features, state_ids)
    centered = np.empty_like(features)
    for state_id in np.unique(state_ids):
        mask = state_ids == state_id
        centered[mask] = features[mask] - features[mask].mean(axis=0, keepdims=True)
    return centered


def cosine_centroid_accuracy(features: np.ndarray, styles: Iterable[str], scenario_ids: Iterable[int],
                             train_scenarios: set[int]) -> tuple[float, np.ndarray]:
    features, styles, scenario_ids = _as_aligned(features, styles, scenario_ids)
    train_mask = np.isin(scenario_ids, list(train_scenarios))
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError("train_scenarios must leave non-empty train and test rows")
    classes = np.unique(styles)
    normalized = _normalize_rows(features)
    centroids = []
    for style in classes:
        mask = train_mask & (styles == style)
        if not mask.any():
            raise ValueError(f"training split has no rows for style {style!r}")
        centroid = normalized[mask].mean(axis=0, keepdims=True)
        centroids.append(_normalize_rows(centroid)[0])
    scores = normalized[test_mask] @ np.stack(centroids).T
    predictions = classes[np.argmax(scores, axis=1)]
    return float(np.mean(predictions == styles[test_mask])), predictions


def shuffle_labels_within_states(labels: Iterable[str], state_ids: Iterable[str],
                                 rng: np.random.Generator) -> np.ndarray:
    labels, state_ids = np.asarray(list(labels)), np.asarray(list(state_ids))
    if labels.shape != state_ids.shape:
        raise ValueError("labels and state_ids must align")
    shuffled = labels.copy()
    for state_id in np.unique(state_ids):
        positions = np.flatnonzero(state_ids == state_id)
        shuffled[positions] = rng.permutation(shuffled[positions])
    return shuffled


def pairwise_style_directions(features: np.ndarray, styles: Iterable[str], scenario_ids: Iterable[int],
                              state_ids: Iterable[str]) -> dict[tuple[str, str], dict]:
    features, styles, scenario_ids, state_ids = _as_aligned(features, styles, scenario_ids, state_ids)
    output = {}
    unique_styles = sorted(str(style) for style in np.unique(styles))
    for first, second in combinations(unique_styles, 2):
        directions, direction_scenarios = [], []
        for state_id in np.unique(state_ids):
            state_mask = state_ids == state_id
            first_rows = np.flatnonzero(state_mask & (styles == first))
            second_rows = np.flatnonzero(state_mask & (styles == second))
            if len(first_rows) != 1 or len(second_rows) != 1:
                continue
            direction = features[first_rows[0]] - features[second_rows[0]]
            norm = np.linalg.norm(direction)
            if norm <= 1e-12:
                continue
            directions.append(direction / norm)
            direction_scenarios.append(int(scenario_ids[first_rows[0]]))
        cross, same = [], []
        for i, j in combinations(range(len(directions)), 2):
            cosine = float(np.dot(directions[i], directions[j]))
            if direction_scenarios[i] == direction_scenarios[j]:
                same.append(cosine)
            else:
                cross.append(cosine)
        output[(first, second)] = {
            "n_directions": len(directions),
            "cross_scenario_cosines": cross,
            "same_scenario_cosines": same,
            "cross_scenario_mean_cosine": float(np.mean(cross)) if cross else float("nan"),
            "cross_scenario_median_cosine": float(np.median(cross)) if cross else float("nan"),
            "same_scenario_mean_cosine": float(np.mean(same)) if same else float("nan"),
            "same_scenario_median_cosine": float(np.median(same)) if same else float("nan"),
        }
    return output


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(p_values), dtype=np.float64)
    if p_values.ndim != 1 or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    if not len(p_values):
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted
