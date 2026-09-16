from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .parageo import (
    compose_coordinates,
    coordinates_from_prototypes,
    dynamic_coordinate_schedule,
    orthogonalize_against_semantics,
)


@dataclass(frozen=True)
class GeometryVariant:
    basis: np.ndarray
    coordinates: dict[str, np.ndarray]
    prototypes: dict[str, np.ndarray]
    all_layers: tuple[int, ...]
    selected_layers: tuple[int, ...]
    basis_kind: str
    rank: int


@dataclass(frozen=True)
class VariantSpec:
    name: str
    prompt_only: bool
    random_control: bool
    full_space: bool
    basis_kind: str
    rank: int
    layer_set: str
    composition_mode: str
    dynamic_variant: str


def load_geometry_variant(path: str, *, basis_kind: str = "orthogonal", rank: int = 16,
                          selected_layers: Sequence[int] | None = None) -> GeometryVariant:
    data = np.load(path)
    names = [str(value) for value in data["attribute_names"]]
    prototypes = data["attribute_prototypes"].astype(np.float64)
    all_layers = tuple(int(value) for value in data["layers"])
    semantic = data["semantic_basis"].astype(np.float64)
    raw_full = data["raw_basis_full"].astype(np.float64)
    if rank < 1 or rank > raw_full.shape[1]:
        raise ValueError(f"rank {rank} is unavailable; max saved rank is {raw_full.shape[1]}")
    raw = raw_full[:, : int(rank)]
    if basis_kind == "raw":
        basis = raw
    elif basis_kind == "orthogonal":
        basis = orthogonalize_against_semantics(raw, semantic)
    else:
        raise ValueError("basis_kind must be raw or orthogonal")
    if basis.shape[1] < 1:
        raise ValueError("semantic orthogonalization removed the selected basis")
    proto_map = {name: prototypes[index] for index, name in enumerate(names)}
    coordinates = coordinates_from_prototypes(proto_map, basis)
    if selected_layers is None:
        selected = all_layers
    else:
        selected = tuple(int(value) for value in selected_layers)
        missing = [layer for layer in selected if layer not in all_layers]
        if missing:
            raise ValueError(f"selected layers are not in calibration layers: {missing}")
        if not selected:
            raise ValueError("selected_layers must be non-empty")
    return GeometryVariant(
        basis=basis,
        coordinates=coordinates,
        prototypes=proto_map,
        all_layers=all_layers,
        selected_layers=selected,
        basis_kind=basis_kind,
        rank=int(rank),
    )


def select_layer_chunks(directions: np.ndarray, *, all_layers: Sequence[int],
                        selected_layers: Sequence[int]) -> np.ndarray:
    array = np.asarray(directions, dtype=np.float64)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("directions must be 1D or 2D")
    all_layers = tuple(int(value) for value in all_layers)
    selected_layers = tuple(int(value) for value in selected_layers)
    if not all_layers or array.shape[1] % len(all_layers) != 0:
        raise ValueError("direction width must divide evenly across all_layers")
    missing = [layer for layer in selected_layers if layer not in all_layers]
    if missing:
        raise ValueError(f"unknown selected layers: {missing}")
    per_layer = array.shape[1] // len(all_layers)
    chunks = []
    for layer in selected_layers:
        position = all_layers.index(layer)
        chunks.append(array[:, position * per_layer:(position + 1) * per_layer])
    result = np.concatenate(chunks, axis=1)
    return result[0] if squeeze else result


def compose_variant_coordinate(coordinates: Mapping[str, np.ndarray], names: Sequence[str], *,
                               mode: str = "sum") -> np.ndarray:
    if mode == "sum":
        return compose_coordinates(coordinates, names, normalize=False)
    if mode == "mean":
        return compose_coordinates(coordinates, names, normalize=False) / float(len(names))
    if mode == "normalized":
        return compose_coordinates(coordinates, names, normalize=True)
    raise ValueError("composition mode must be sum, mean, or normalized")


def dynamic_variant_schedule(start: np.ndarray, end: np.ndarray, *, variant: str,
                             instruction_mode: str, steps: int, transition_at: float) -> np.ndarray:
    a = np.asarray(start, dtype=np.float64).reshape(-1)
    b = np.asarray(end, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("start and end coordinates must match")
    if variant == "scheduled":
        return dynamic_coordinate_schedule(
            a, b, steps=steps, mode=instruction_mode, transition_at=transition_at
        )
    if variant == "start_static":
        return np.repeat(a[None, :], int(steps), axis=0)
    if variant == "end_static":
        return np.repeat(b[None, :], int(steps), axis=0)
    if variant == "midpoint_static":
        midpoint = 0.5 * (a + b)
        return np.repeat(midpoint[None, :], int(steps), axis=0)
    raise ValueError("dynamic variant must be scheduled, start_static, end_static, or midpoint_static")


def resolve_variant(name: str, *, task: str, main_rank: int = 16) -> VariantSpec:
    name = str(name)
    if task not in {"static", "composed", "dynamic"}:
        raise ValueError("task must be static, composed, or dynamic")
    spec = VariantSpec(
        name=name,
        prompt_only=name == "prompt_only",
        random_control=name == "random",
        full_space=name == "full_space",
        basis_kind="raw" if name == "raw_basis" else "orthogonal",
        rank=int(name[4:]) if name.startswith("rank") else int(main_rank),
        layer_set=name.removeprefix("layers_") if name.startswith("layers_") else "all",
        composition_mode=(
            "mean" if name == "comp_mean" else
            "normalized" if name == "comp_normalized" else
            "sum"
        ),
        dynamic_variant=(
            "start_static" if name == "dyn_start" else
            "end_static" if name == "dyn_end" else
            "midpoint_static" if name == "dyn_midpoint" else
            "scheduled"
        ),
    )
    if name.startswith("comp_") and task != "composed":
        raise ValueError("composition ablations are only valid for composed task")
    if name == "full_space" and task != "composed":
        raise ValueError("full-space composition baseline is only valid for composed task")
    if name.startswith("dyn_") and task != "dynamic":
        raise ValueError("dynamic ablations are only valid for dynamic task")
    allowed = {
        "prompt_only", "main", "random", "full_space", "raw_basis",
        "rank4", "rank8", "rank16", "rank32",
        "layers_early", "layers_middle", "layers_late", "layers_all",
        "comp_mean", "comp_normalized",
        "dyn_start", "dyn_end", "dyn_midpoint",
    }
    if name not in allowed:
        raise ValueError(f"unknown ICASSP variant: {name}")
    return spec


def norm_matched_random_directions(target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    array = np.asarray(target, dtype=np.float64)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("target directions must be 1D or 2D")
    random = rng.normal(size=array.shape[1])
    random /= max(np.linalg.norm(random), 1e-12)
    norms = np.linalg.norm(array, axis=1)
    result = norms[:, None] * random[None, :]
    return result[0] if squeeze else result
