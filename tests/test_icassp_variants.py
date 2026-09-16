from pathlib import Path
import numpy as np
import pytest

from speech_negotiation_kv.icassp_variants import (
    compose_variant_coordinate, dynamic_variant_schedule, load_geometry_variant,
    norm_matched_random_directions, resolve_variant, select_layer_chunks,
)


def make_artifact(path: Path):
    rng = np.random.default_rng(3); ambient = 12
    raw, _ = np.linalg.qr(rng.normal(size=(ambient, 4)))
    semantic, _ = np.linalg.qr(rng.normal(size=(ambient, 2)))
    prototypes = rng.normal(size=(3, ambient))
    np.savez(path, attribute_names=np.array(["a", "b", "c"]), attribute_prototypes=prototypes,
             raw_basis_full=raw, semantic_basis=semantic, layers=np.array([16, 20, 24]))


def test_variant_loader_reconstructs_rank_raw_and_orthogonal(tmp_path):
    artifact = tmp_path / "basis.npz"; make_artifact(artifact)
    raw = load_geometry_variant(artifact, basis_kind="raw", rank=2)
    orth = load_geometry_variant(artifact, basis_kind="orthogonal", rank=2)
    assert raw.basis.shape == (12, 2) and orth.basis.shape[0] == 12
    assert set(raw.coordinates) == {"a", "b", "c"}
    assert set(raw.prototypes) == {"a", "b", "c"}
    assert raw.prototypes["a"].shape == (12,)
    assert np.linalg.norm(np.load(artifact)["semantic_basis"].T @ orth.basis) < 1e-8


def test_layer_chunk_selection_is_exact():
    direction = np.arange(12, dtype=float)
    selected = select_layer_chunks(direction, all_layers=[16, 20, 24], selected_layers=[16, 24])
    assert np.allclose(selected, [0, 1, 2, 3, 8, 9, 10, 11])


def test_composition_modes_and_dynamic_controls():
    coords = {"a": np.array([2.0, 0.0]), "b": np.array([0.0, 2.0])}
    assert np.allclose(compose_variant_coordinate(coords, ["a", "b"], mode="sum"), [2, 2])
    assert np.allclose(compose_variant_coordinate(coords, ["a", "b"], mode="mean"), [1, 1])
    assert np.isclose(np.linalg.norm(compose_variant_coordinate(coords, ["a", "b"], mode="normalized")), 1.0)
    scheduled = dynamic_variant_schedule(coords["a"], coords["b"], variant="scheduled", instruction_mode="linear", steps=3, transition_at=.5)
    assert np.allclose(scheduled[[0, -1]], [[2, 0], [0, 2]])
    midpoint = dynamic_variant_schedule(coords["a"], coords["b"], variant="midpoint_static", instruction_mode="linear", steps=3, transition_at=.5)
    assert np.allclose(midpoint, [[1, 1], [1, 1], [1, 1]])


def test_variant_resolution_and_random_norms():
    assert resolve_variant("raw_basis", task="static").basis_kind == "raw"
    assert resolve_variant("rank8", task="static").rank == 8
    assert resolve_variant("layers_late", task="static").layer_set == "late"
    assert resolve_variant("comp_mean", task="composed").composition_mode == "mean"
    assert resolve_variant("dyn_midpoint", task="dynamic").dynamic_variant == "midpoint_static"
    assert resolve_variant("full_space", task="composed").full_space is True
    with pytest.raises(ValueError):
        resolve_variant("full_space", task="static")
    target = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    random = norm_matched_random_directions(target, np.random.default_rng(4))
    assert np.allclose(np.linalg.norm(random, axis=1), [5.0, 2.0])
