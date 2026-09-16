from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .parageo import leave_one_content_out_centroid_accuracy, same_attribute_cross_content_cosine


def _summary(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _permute_within_content(attributes: Sequence[str], content_ids: Sequence[str],
                            rng: np.random.Generator) -> np.ndarray:
    attrs = np.asarray(attributes, dtype=object)
    contents = np.asarray(content_ids, dtype=object)
    if len(attrs) != len(contents):
        raise ValueError("attributes and content_ids must align")
    shuffled = attrs.copy()
    for content in np.unique(contents):
        mask = contents == content
        shuffled[mask] = rng.permutation(shuffled[mask])
    return shuffled


def shuffled_geometry_null(features: np.ndarray, content_ids: Sequence[str],
                           attributes: Sequence[str], *, repeats: int = 1000,
                           seed: int = 15242424242) -> dict:
    """Permutation null that preserves every lexical-content block.

    Attribute labels are independently permuted within each lexical content. This
    retains per-content marginals and activation structure while destroying the
    cross-content identity of each paralinguistic attribute.
    """
    if int(repeats) < 1:
        raise ValueError("repeats must be positive")
    real_loco = leave_one_content_out_centroid_accuracy(features, content_ids, attributes)
    real_cosine = same_attribute_cross_content_cosine(features, content_ids, attributes).get("mean")
    if real_cosine is None:
        raise ValueError("cross-content cosine is undefined for the supplied features")
    rng = np.random.default_rng(int(seed))
    loco_null = np.empty(int(repeats), dtype=np.float64)
    cosine_null = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        shuffled = _permute_within_content(attributes, content_ids, rng)
        loco_null[index] = leave_one_content_out_centroid_accuracy(features, content_ids, shuffled)
        value = same_attribute_cross_content_cosine(features, content_ids, shuffled).get("mean")
        cosine_null[index] = np.nan if value is None else float(value)
    valid_cosine = cosine_null[np.isfinite(cosine_null)]
    loco_p = (1.0 + float(np.sum(loco_null >= float(real_loco)))) / (len(loco_null) + 1.0)
    cosine_p = (1.0 + float(np.sum(valid_cosine >= float(real_cosine)))) / (len(valid_cosine) + 1.0)
    return {
        "protocol": "independent attribute-label permutation within each lexical content",
        "repeats": int(repeats),
        "seed": int(seed),
        "real": {
            "loco_accuracy": float(real_loco),
            "cross_content_cosine": float(real_cosine),
        },
        "null": {
            "loco_accuracy": _summary(loco_null),
            "cross_content_cosine": _summary(valid_cosine),
        },
        "p_value": {
            "loco_accuracy": float(loco_p),
            "cross_content_cosine": float(cosine_p),
        },
    }


def full_space_composition_direction(prototypes: Mapping[str, np.ndarray],
                                     names: Sequence[str], *,
                                     reference_direction: np.ndarray | None = None) -> np.ndarray:
    """Compose raw full-dimensional attribute prototypes.

    When ``reference_direction`` is supplied, the raw vector is norm-matched to
    the corresponding ParaGeo direction. This makes the baseline a direction-
    quality comparison rather than an intervention-magnitude comparison.
    """
    if not names:
        raise ValueError("names must be non-empty")
    missing = [name for name in names if name not in prototypes]
    if missing:
        raise KeyError(f"missing full-space prototypes: {missing}")
    vectors = [np.asarray(prototypes[name], dtype=np.float64).reshape(-1) for name in names]
    widths = {len(vector) for vector in vectors}
    if len(widths) != 1:
        raise ValueError("all full-space prototypes must have the same width")
    direction = np.sum(np.stack(vectors), axis=0)
    if reference_direction is None:
        return direction
    reference = np.asarray(reference_direction, dtype=np.float64).reshape(-1)
    target_norm = float(np.linalg.norm(reference))
    source_norm = float(np.linalg.norm(direction))
    if target_norm <= 1e-12 or source_norm <= 1e-12:
        raise ValueError("cannot norm-match a zero-norm composition direction")
    return direction * (target_norm / source_norm)
