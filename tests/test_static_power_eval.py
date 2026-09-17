import numpy as np

from speech_negotiation_kv.static_power_eval import (
    hierarchical_macro_preference,
    pool_seed_records,
)


def test_macro_preference_weights_attributes_equally():
    records = [
        *({"attribute_key": "A", "item_id": f"a{i}", "score": 1.0} for i in range(10)),
        {"attribute_key": "B", "item_id": "b0", "score": 0.0},
    ]

    result = hierarchical_macro_preference(records, repeats=2000, seed=7)

    assert result["n_attributes"] == 2
    assert result["n_items"] == 11
    assert result["macro_preference"] == 50.0


def test_hierarchical_bootstrap_is_reproducible_and_bounded():
    records = [
        {"attribute_key": attr, "item_id": f"{attr}{i}", "score": score}
        for attr, values in {"A": [1, 1, 0], "B": [0.5, 0.5, 1]}.items()
        for i, score in enumerate(values)
    ]

    first = hierarchical_macro_preference(records, repeats=500, seed=9)
    second = hierarchical_macro_preference(records, repeats=500, seed=9)

    assert first == second
    assert 0 <= first["lower_95"] <= first["macro_preference"] <= first["upper_95"] <= 100


def test_pool_seed_records_averages_each_item_before_macro_analysis():
    records_by_seed = {
        1: [{"attribute_key": "A", "item_id": "x", "score": 1.0}],
        2: [{"attribute_key": "A", "item_id": "x", "score": 0.0}],
    }

    pooled = pool_seed_records(records_by_seed)

    assert pooled == [{"attribute_key": "A", "item_id": "x", "score": 0.5, "seed_count": 2}]
