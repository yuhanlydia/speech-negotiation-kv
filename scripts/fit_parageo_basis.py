#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.parageo import (
    attribute_prototypes,
    choose_rank,
    coordinates_from_prototypes,
    fit_content_invariant_basis,
    leave_one_content_out_centroid_accuracy,
    orthogonalize_against_semantics,
    same_attribute_cross_content_cosine,
    semantic_subspace,
)
from speech_negotiation_kv.records import read_jsonl


def _branch_id(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit content-invariant ParaGeo geometry and ICASSP diagnostics")
    ap.add_argument("--config", default="configs/parageo_speechparaling_pilot.yaml")
    ap.add_argument("--records", default="results/parageo_calibration.jsonl")
    ap.add_argument("--kv", default="results/parageo_calibration_kv.npz")
    ap.add_argument("--output", default="results/parageo_basis.npz")
    ap.add_argument("--summary", default="results/parageo_basis_summary.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = cfg["parageo"]
    rows = {
        str(row["branch_id"]): row
        for row in read_jsonl(args.records)
        if row.get("matched_semantics", False)
    }
    kv = np.load(args.kv)
    all_features = kv["memory"].astype(np.float64).reshape(len(kv["memory"]), -1)
    branch_ids = [_branch_id(value) for value in kv["branch_ids"]]
    indices = [index for index, branch in enumerate(branch_ids) if branch in rows]
    if not indices:
        raise RuntimeError("no matched calibration K/V rows align with records")
    features = all_features[indices]
    aligned = [rows[branch_ids[index]] for index in indices]
    content = [str(row["content_id"]) for row in aligned]
    attrs = [str(row["attribute"]) for row in aligned]

    requested_max = int(pcfg.get("max_basis_rank", 32))
    max_rank = min(requested_max, features.shape[0], features.shape[1])
    full_fit = fit_content_invariant_basis(features, content, rank=max_rank)
    main_rank = min(int(pcfg.get("basis_rank", 16)), full_fit.rank)
    raw_main = full_fit.basis[:, :main_rank]

    unique_contents = sorted(set(content))
    content_array = np.asarray(content, dtype=object)
    content_means = np.stack([
        features[content_array == content_id].mean(axis=0)
        for content_id in unique_contents
    ])
    semantic = semantic_subspace(content_means, rank=int(pcfg.get("semantic_rank", 8)))
    basis_main = orthogonalize_against_semantics(raw_main, semantic)
    basis_full = orthogonalize_against_semantics(full_fit.basis, semantic)

    prototypes = attribute_prototypes(full_fit.centered_features, attrs)
    names = sorted(prototypes)
    prototype_matrix = np.stack([prototypes[name] for name in names])
    coordinates = coordinates_from_prototypes(prototypes, basis_main)
    coordinate_matrix = np.stack([coordinates[name] for name in names])

    loco = leave_one_content_out_centroid_accuracy(features, content, attrs)
    consistency = same_attribute_cross_content_cosine(features, content, attrs)
    rank95 = choose_rank(
        full_fit.singular_values,
        max_rank=max_rank,
        energy=float(pcfg.get("basis_energy", 0.95)),
    )
    cumulative = np.cumsum(full_fit.explained_variance_ratio)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        basis=basis_main.astype(np.float32),
        raw_basis=raw_main.astype(np.float32),
        basis_full=basis_full.astype(np.float32),
        raw_basis_full=full_fit.basis.astype(np.float32),
        attribute_names=np.asarray(names),
        attribute_prototypes=prototype_matrix.astype(np.float32),
        coordinates=coordinate_matrix.astype(np.float32),
        content_ids=np.asarray(unique_contents),
        content_means=content_means.astype(np.float32),
        layers=kv["layers"],
        singular_values=full_fit.singular_values.astype(np.float32),
        explained_variance_ratio=full_fit.explained_variance_ratio.astype(np.float32),
        semantic_basis=semantic.astype(np.float32),
    )
    summary = {
        "rows": len(features),
        "contents": len(unique_contents),
        "attributes": len(names),
        "main_rank_requested": int(pcfg.get("basis_rank", 16)),
        "main_raw_rank": int(main_rank),
        "main_orthogonal_rank": int(basis_main.shape[1]),
        "max_raw_rank": int(full_fit.rank),
        "max_orthogonal_rank": int(basis_full.shape[1]),
        "rank_95_energy": int(rank95),
        "explained_variance_at_main_rank": float(cumulative[main_rank - 1]),
        "leave_one_content_out_centroid_accuracy": float(loco),
        "chance": 1.0 / len(names),
        "same_attribute_cross_content_cosine": consistency,
        "output": str(out),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
