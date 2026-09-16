import numpy as np
import pytest

from speech_negotiation_kv.icassp_reviewer_controls import (
    full_space_composition_direction,
    shuffled_geometry_null,
)


def _synthetic_signal():
    attributes = ["a", "b", "c", "d"]
    attribute_vectors = {name: np.eye(4)[index] for index, name in enumerate(attributes)}
    rows, contents, labels = [], [], []
    for content_index in range(8):
        semantic = np.array([10 + content_index, -3 * content_index, 2 * content_index, 7 - content_index], dtype=float)
        for attribute in attributes:
            rows.append(semantic + 5.0 * attribute_vectors[attribute])
            contents.append(f"c{content_index}")
            labels.append(attribute)
    return np.stack(rows), contents, labels


def test_shuffled_geometry_null_breaks_cross_content_attribute_alignment():
    features, contents, attributes = _synthetic_signal()
    result = shuffled_geometry_null(features, contents, attributes, repeats=199, seed=7)
    assert result["real"]["loco_accuracy"] == 1.0
    assert result["real"]["cross_content_cosine"] > 0.99
    assert result["null"]["loco_accuracy"]["mean"] < 0.5
    assert result["null"]["cross_content_cosine"]["mean"] < 0.5
    assert result["p_value"]["loco_accuracy"] <= 0.01
    assert result["p_value"]["cross_content_cosine"] <= 0.01


def test_full_space_composition_is_norm_matched_to_reference_direction():
    prototypes = {"a": np.array([3.0, 0.0, 0.0]), "b": np.array([0.0, 4.0, 0.0])}
    reference = np.array([0.0, 0.0, 2.0])
    direction = full_space_composition_direction(
        prototypes, ["a", "b"], reference_direction=reference
    )
    assert np.allclose(direction, np.array([1.2, 1.6, 0.0]))
    assert np.isclose(np.linalg.norm(direction), np.linalg.norm(reference))


def test_full_space_composition_rejects_missing_attribute():
    with pytest.raises(KeyError):
        full_space_composition_direction({"a": np.ones(2)}, ["a", "b"])
