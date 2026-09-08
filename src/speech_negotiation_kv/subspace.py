from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Any

import numpy as np


@dataclass(frozen=True)
class SubspaceResult:
    basis: np.ndarray
    singular_values: np.ndarray
    explained_variance: float


def balanced_scenario_splits(scenario_ids: Iterable[int], *, half_size: int | None = None) -> list[tuple[set[int], set[int]]]:
    """Enumerate unique complementary balanced splits, avoiding duplicate mirrors."""
    ids = sorted({int(s) for s in scenario_ids})
    if half_size is None:
        if len(ids) % 2:
            raise ValueError("scenario count must be even when half_size is omitted")
        half_size = len(ids) // 2
    half_size = int(half_size)
    if half_size <= 0 or 2 * half_size != len(ids):
        raise ValueError("half_size must partition all scenario IDs into two equal halves")
    anchor = ids[0]
    universe = set(ids)
    return [
        (set(left), universe - set(left))
        for left in combinations(ids, half_size)
        if anchor in left
    ]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_rank_correlation(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = np.asarray(list(a), dtype=np.float64), np.asarray(list(b), dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("Spearman correlation needs paired arrays with at least two values")
    ra, rb = _average_ranks(a), _average_ranks(b)
    if np.allclose(ra, ra[0]) or np.allclose(rb, rb[0]):
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def advantage_memory_directions(records: Iterable[Mapping[str, Any]]) -> tuple[list[str], np.ndarray]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        groups.setdefault(str(rec["state_id"]), []).append(rec)
    states: list[str] = []
    directions: list[np.ndarray] = []
    for state_id in sorted(groups):
        rows = groups[state_id]
        memories = np.stack([np.asarray(r["memory"], dtype=np.float64) for r in rows])
        utilities = np.asarray([float(r["utility"]) for r in rows], dtype=np.float64)
        centered_m = memories - memories.mean(axis=0, keepdims=True)
        advantages = utilities - utilities.mean()
        denom = np.abs(advantages).sum()
        if denom <= 1e-12:
            continue
        weights = advantages / denom
        direction = (weights[:, None] * centered_m).sum(axis=0)
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        states.append(state_id)
        directions.append(direction / norm)
    if not directions:
        return [], np.empty((0, 0), dtype=np.float64)
    return states, np.stack(directions)


def fit_low_rank_subspace(G: np.ndarray, rank: int) -> SubspaceResult:
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] < 1:
        raise ValueError("G must be a non-empty 2D matrix with rows as state directions")
    rank = int(max(1, min(rank, min(G.shape))))
    _, s, vt = np.linalg.svd(G, full_matrices=False)
    basis = vt[:rank].T
    total = float(np.sum(s**2))
    kept = float(np.sum(s[:rank]**2))
    explained = kept / total if total > 0 else 0.0
    return SubspaceResult(basis=basis, singular_values=s[:rank], explained_variance=explained)


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must be 2D matrices in the same ambient dimension")
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    r = min(qa.shape[1], qb.shape[1])
    return float(np.linalg.norm(qa.T @ qb, ord="fro") ** 2 / r)
