from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np


def hierarchical_macro_preference(records: Iterable[Mapping], *, repeats: int = 10000,
                                  seed: int = 2027091701) -> dict:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in records:
        buckets[str(row["attribute_key"])].append(float(row["score"]))
    if not buckets:
        return {
            "n_attributes": 0, "n_items": 0, "macro_preference": None,
            "lower_95": None, "upper_95": None, "repeats": int(repeats),
            "by_attribute": {},
        }
    attributes = sorted(buckets)
    by_attribute = {
        attribute: {"n": len(buckets[attribute]),
                    "preference_score": 100.0 * float(np.mean(buckets[attribute]))}
        for attribute in attributes
    }
    point = float(np.mean([by_attribute[attribute]["preference_score"] for attribute in attributes]))
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        sampled_attributes = rng.choice(attributes, size=len(attributes), replace=True)
        attribute_means = []
        for attribute in sampled_attributes:
            values = np.asarray(buckets[str(attribute)], dtype=np.float64)
            sample = rng.choice(values, size=len(values), replace=True)
            attribute_means.append(float(sample.mean()))
        boot[index] = 100.0 * float(np.mean(attribute_means))
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "n_attributes": len(attributes),
        "n_items": sum(len(values) for values in buckets.values()),
        "macro_preference": point,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "repeats": int(repeats),
        "by_attribute": by_attribute,
    }


def pool_seed_records(records_by_seed: Mapping[int, Iterable[Mapping]]) -> list[dict]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for records in records_by_seed.values():
        for row in records:
            key = (str(row["attribute_key"]), str(row["item_id"]))
            buckets[key].append(float(row["score"]))
    return [
        {"attribute_key": attribute, "item_id": item_id,
         "score": float(np.mean(values)), "seed_count": len(values)}
        for (attribute, item_id), values in sorted(buckets.items())
    ]
