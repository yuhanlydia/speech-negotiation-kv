import numpy as np

from speech_negotiation_kv.parageo import (
    center_within_content,
    fit_content_invariant_basis,
    attribute_prototypes,
    coordinates_from_prototypes,
    compose_coordinates,
    dynamic_coordinate_schedule,
    semantic_subspace,
    orthogonalize_against_semantics,
    cross_content_centroid_accuracy,
)


def synthetic_geometry():
    rng = np.random.default_rng(7)
    attributes = ["low", "high", "warm"]
    attr_vectors = {
        "low": np.array([1.0, 0.0, 0.0, 0.0]),
        "high": np.array([-1.0, 0.0, 0.0, 0.0]),
        "warm": np.array([0.0, 1.0, 0.0, 0.0]),
    }
    rows, content_ids, attrs = [], [], []
    for content in range(8):
        semantic = np.array([0.0, 0.0, 4.0 + content, -2.0 * content])
        for attr in attributes:
            rows.append(semantic + attr_vectors[attr] + rng.normal(scale=0.01, size=4))
            content_ids.append(f"c{content}")
            attrs.append(attr)
    return np.stack(rows), content_ids, attrs


def test_content_centering_removes_content_means():
    X, ids, _ = synthetic_geometry()
    centered = center_within_content(X, ids)
    for content in set(ids):
        assert np.allclose(centered[np.asarray(ids) == content].mean(0), 0.0, atol=1e-10)


def test_fit_recovers_low_rank_transferable_geometry():
    X, ids, attrs = synthetic_geometry()
    fit = fit_content_invariant_basis(X, ids, rank=2)
    assert fit.basis.shape == (4, 2)
    assert cross_content_centroid_accuracy(X, ids, attrs, {"c0", "c1", "c2", "c3"}) == 1.0
    prototypes = attribute_prototypes(fit.centered_features, attrs)
    coordinates = coordinates_from_prototypes(prototypes, fit.basis)
    assert set(coordinates) == {"low", "high", "warm"}


def test_composition_and_dynamic_schedule_are_explicit():
    coordinates = {"warm": np.array([1.0, 0.0]), "slow": np.array([0.0, 2.0])}
    composed = compose_coordinates(coordinates, ["warm", "slow"], weights=[1.0, 0.5])
    assert np.allclose(composed, [1.0, 1.0])
    schedule = dynamic_coordinate_schedule(np.array([1.0, 0.0]), np.array([0.0, 1.0]), steps=5)
    assert np.allclose(schedule[0], [1.0, 0.0])
    assert np.allclose(schedule[-1], [0.0, 1.0])
    step = dynamic_coordinate_schedule(np.array([1.0]), np.array([3.0]), steps=5, mode="step", transition_at=0.5)
    assert np.allclose(step[:, 0], [1.0, 1.0, 3.0, 3.0, 3.0])


def test_semantic_orthogonalization_removes_semantic_projection():
    content_means = np.array([[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 1], [0, 0, 4, 1]], dtype=float)
    S = semantic_subspace(content_means, rank=2)
    B = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    B_orth = orthogonalize_against_semantics(B, S)
    assert np.linalg.norm(S.T @ B_orth) < 1e-8
