from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParaGeoFit:
    basis: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    rank: int
    centered_features: np.ndarray


def _as_2d(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("features must be a 2D array with at least two rows")
    if not np.all(np.isfinite(array)):
        raise ValueError("features must be finite")
    return array


def center_within_content(features: np.ndarray, content_ids: Sequence[str]) -> np.ndarray:
    X = _as_2d(features)
    ids = np.asarray(content_ids, dtype=object)
    if len(ids) != len(X):
        raise ValueError("content_ids must align with features")
    centered = np.empty_like(X)
    for content in np.unique(ids):
        mask = ids == content
        centered[mask] = X[mask] - X[mask].mean(axis=0, keepdims=True)
    return centered


def choose_rank(singular_values: np.ndarray, *, max_rank: int | None = None,
                energy: float = 0.95) -> int:
    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if not len(values) or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("singular_values must be finite non-negative values")
    if not 0 < energy <= 1:
        raise ValueError("energy must be in (0, 1]")
    power = values ** 2
    if power.sum() <= 0:
        return 1
    rank = int(np.searchsorted(np.cumsum(power) / power.sum(), energy) + 1)
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return max(1, min(rank, len(values)))


def fit_content_invariant_basis(features: np.ndarray, content_ids: Sequence[str], *,
                                rank: int | None = None, max_rank: int = 32,
                                energy: float = 0.95) -> ParaGeoFit:
    centered = center_within_content(features, content_ids)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    if rank is None:
        rank = choose_rank(s, max_rank=min(max_rank, vt.shape[0]), energy=energy)
    rank = int(rank)
    if rank < 1 or rank > vt.shape[0]:
        raise ValueError(f"rank must be in [1, {vt.shape[0]}]")
    power = s ** 2
    ratio = power / power.sum() if power.sum() > 0 else np.zeros_like(power)
    return ParaGeoFit(
        basis=vt[:rank].T.copy(),
        singular_values=s.copy(),
        explained_variance_ratio=ratio.copy(),
        rank=rank,
        centered_features=centered,
    )


def attribute_prototypes(centered_features: np.ndarray, attributes: Sequence[str]) -> dict[str, np.ndarray]:
    X = _as_2d(centered_features)
    attrs = np.asarray(attributes, dtype=object)
    if len(attrs) != len(X):
        raise ValueError("attributes must align with features")
    result: dict[str, np.ndarray] = {}
    for attr in sorted(str(value) for value in np.unique(attrs)):
        mask = attrs == attr
        result[attr] = X[mask].mean(axis=0)
    return result


def coordinates_from_prototypes(prototypes: Mapping[str, np.ndarray], basis: np.ndarray) -> dict[str, np.ndarray]:
    B = np.asarray(basis, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("basis must be 2D")
    return {
        str(name): np.asarray(vector, dtype=np.float64).reshape(-1) @ B
        for name, vector in prototypes.items()
    }


def direction_from_coordinate(basis: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    B = np.asarray(basis, dtype=np.float64)
    c = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    if B.ndim != 2 or B.shape[1] != len(c):
        raise ValueError("coordinate dimension must match basis rank")
    return B @ c


def compose_coordinates(coordinates: Mapping[str, np.ndarray], names: Sequence[str], *,
                        weights: Sequence[float] | None = None, normalize: bool = False) -> np.ndarray:
    if not names:
        raise ValueError("names must be non-empty")
    missing = [name for name in names if name not in coordinates]
    if missing:
        raise KeyError(f"missing coordinates: {missing}")
    if weights is None:
        weights = [1.0] * len(names)
    if len(weights) != len(names):
        raise ValueError("weights must align with names")
    vectors = [float(weight) * np.asarray(coordinates[name], dtype=np.float64) for name, weight in zip(names, weights)]
    result = np.sum(np.stack(vectors), axis=0)
    if normalize:
        norm = np.linalg.norm(result)
        if norm > 1e-12:
            result = result / norm
    return result


def _transition_fraction(steps: int, *, mode: str, transition_at: float) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0 < transition_at < 1:
        raise ValueError("transition_at must be in (0, 1)")
    t = np.linspace(0.0, 1.0, int(steps))
    if mode == "linear":
        return t
    if mode == "step":
        return (t >= transition_at).astype(np.float64)
    if mode == "sigmoid":
        sharpness = 12.0
        raw = 1.0 / (1.0 + np.exp(-sharpness * (t - transition_at)))
        return (raw - raw[0]) / max(raw[-1] - raw[0], 1e-12)
    raise ValueError(f"unknown transition mode: {mode}")


def dynamic_coordinate_schedule(start: np.ndarray, end: np.ndarray, *, steps: int,
                                mode: str = "linear", transition_at: float = 0.5) -> np.ndarray:
    a = np.asarray(start, dtype=np.float64).reshape(-1)
    b = np.asarray(end, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("start and end coordinates must have the same shape")
    fraction = _transition_fraction(steps, mode=mode, transition_at=transition_at)
    return (1.0 - fraction[:, None]) * a[None, :] + fraction[:, None] * b[None, :]


def semantic_subspace(content_means: np.ndarray, *, rank: int = 8) -> np.ndarray:
    X = _as_2d(content_means)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    rank = max(1, min(int(rank), vt.shape[0]))
    return vt[:rank].T.copy()


def orthogonalize_against_semantics(paraling_basis: np.ndarray, semantic_basis: np.ndarray) -> np.ndarray:
    B = np.asarray(paraling_basis, dtype=np.float64)
    S = np.asarray(semantic_basis, dtype=np.float64)
    if B.ndim != 2 or S.ndim != 2 or B.shape[0] != S.shape[0]:
        raise ValueError("paralinguistic and semantic bases must share ambient dimension")
    q_sem, _ = np.linalg.qr(S)
    residual = B - q_sem @ (q_sem.T @ B)
    q, r = np.linalg.qr(residual)
    keep = np.abs(np.diag(r)) > 1e-9
    if not np.any(keep):
        raise ValueError("paralinguistic basis lies entirely in semantic subspace")
    return q[:, keep]


def cross_content_centroid_accuracy(features: np.ndarray, content_ids: Sequence[str], attributes: Sequence[str],
                                    train_contents: Iterable[str]) -> float:
    X = center_within_content(features, content_ids)
    ids = np.asarray(content_ids, dtype=object)
    attrs = np.asarray(attributes, dtype=object)
    train_set = {str(value) for value in train_contents}
    train_mask = np.asarray([str(value) in train_set for value in ids])
    test_mask = ~train_mask
    classes = sorted(str(value) for value in np.unique(attrs))
    if not train_mask.any() or not test_mask.any():
        raise ValueError("train_contents must leave both train and test rows")
    centroids = {}
    for attr in classes:
        mask = train_mask & (attrs == attr)
        if not mask.any():
            raise ValueError(f"attribute {attr!r} missing from training contents")
        vec = X[mask].mean(axis=0)
        centroids[attr] = vec / max(np.linalg.norm(vec), 1e-12)
    correct = 0
    total = 0
    for row, truth in zip(X[test_mask], attrs[test_mask]):
        normalized = row / max(np.linalg.norm(row), 1e-12)
        prediction = max(classes, key=lambda attr: float(normalized @ centroids[attr]))
        correct += int(prediction == str(truth))
        total += 1
    return float(correct / total)
