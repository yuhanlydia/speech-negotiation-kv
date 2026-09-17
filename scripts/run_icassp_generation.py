#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from speech_negotiation_kv.glm_official_waveform import GLMOfficialWaveformBackend, OfficialVoiceAssets
from speech_negotiation_kv.icassp_reviewer_controls import full_space_composition_direction
from speech_negotiation_kv.icassp_variants import (
    compose_variant_coordinate,
    dynamic_variant_schedule,
    load_geometry_variant,
    norm_matched_random_directions,
    resolve_variant,
    select_layer_chunks,
)
from speech_negotiation_kv.parageo import direction_from_coordinate
from speech_negotiation_kv.parageo_steering import ScheduledFusedQKVSteerer
from speech_negotiation_kv.speechparaling import (
    SpeechParalingItem,
    attribute_key,
    extract_target_text,
    load_prompt_jsonl,
    match_catalog_controls,
    pair_with_audio,
    parse_dynamic_control,
    select_catalog_covered_items,
    select_icassp_split,
    word_error_rate,
    write_manifest,
)


def _expand(path: str, *, benchmark_root: str | None = None) -> str:
    value = os.path.expandvars(os.path.expanduser(str(path)))
    if benchmark_root and not os.path.isabs(value):
        value = os.path.join(benchmark_root, value)
    return value


def _item_index(item_id: str) -> int:
    if str(item_id).startswith("static_power:"):
        return int.from_bytes(hashlib.sha256(str(item_id).encode("utf-8")).digest()[:4], "big")
    try:
        return int(str(item_id).rsplit(":", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"item_id does not end in an integer index: {item_id}") from exc


def load_external_items(path: str | Path, *, task: str) -> list[SpeechParalingItem]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        audio_path = Path(str(payload["audio_path"]))
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        rows.append(SpeechParalingItem(
            item_id=str(payload["item_id"]),
            prompt=str(payload["prompt"]),
            dimensions=tuple(str(value) for value in payload.get("dimensions", [])),
            task=str(task),
            audio_path=str(audio_path),
        ))
    return rows


def resolve_random_seed(experiment_config: dict, override: int | None) -> int:
    return int(experiment_config["random_seed"] if override is None else override)


def _existing_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[str(row["item_id"])] = row
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate frozen ICASSP 2027 ParaGeo experiment outputs")
    ap.add_argument("--config", default="configs/icassp2027_parageo.yaml")
    ap.add_argument("--task", required=True, choices=["static", "composed", "dynamic"])
    ap.add_argument("--split", required=True, choices=["dev", "heldout", "ablation"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--basis", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-manifest", default=None)
    ap.add_argument("--random-seed", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg, benchmark_cfg = cfg["model"], cfg["benchmark"]
    parageo_cfg, exp_cfg = cfg["parageo"], cfg["experiment"]
    task_cfg = benchmark_cfg["tasks"][args.task]
    benchmark_root = _expand(benchmark_cfg["root"])
    prompt_jsonl = _expand(task_cfg["prompt_jsonl"], benchmark_root=benchmark_root)
    audio_dir = _expand(task_cfg["audio_dir"], benchmark_root=benchmark_root)
    catalog_path = args.catalog or parageo_cfg["catalog"]
    basis_path = args.basis or parageo_cfg["basis"]
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))

    items = (load_external_items(args.dataset_manifest, task=args.task) if args.dataset_manifest else
             pair_with_audio(load_prompt_jsonl(prompt_jsonl, task=args.task), audio_dir))
    items = select_catalog_covered_items(items, catalog, task=args.task)
    if not args.dataset_manifest:
        items = select_icassp_split(
            items, split=args.split, modulus=int(exp_cfg["dev_modulus"]),
            remainder=int(exp_cfg["dev_remainder"]), ablation_limit=int(exp_cfg["ablation_limit"]),
        )
    if not items:
        raise RuntimeError(f"no eligible {args.task} items for split {args.split}")

    spec = resolve_variant(args.variant, task=args.task, main_rank=int(parageo_cfg["main_rank"]))
    selected_layers = tuple(int(v) for v in parageo_cfg["layer_sets"][spec.layer_set])
    geometry = None if spec.prompt_only else load_geometry_variant(
        basis_path, basis_kind=spec.basis_kind, rank=spec.rank, selected_layers=selected_layers
    )

    official_cfg = cfg["official_glm"]
    assets = OfficialVoiceAssets(
        repo_root=_expand(official_cfg["repo_root"]),
        speech_tokenizer_path=_expand(official_cfg["speech_tokenizer_path"]),
        decoder_path=_expand(official_cfg["decoder_path"]),
    )
    backend = GLMOfficialWaveformBackend(
        assets=assets, model_name=model_cfg["name"], quantization=model_cfg.get("quantization", "int4"),
        device=model_cfg.get("device", "cuda:0"), max_new_tokens=model_cfg.get("max_new_tokens", 768),
        temperature=model_cfg.get("temperature", 0.8), top_p=model_cfg.get("top_p", 0.8),
        audio_vocab_size=model_cfg.get("audio_vocab_size", 16384),
    )

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.jsonl"
    if args.fresh and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {} if args.fresh else _existing_records(manifest_path)
    random_seed = resolve_random_seed(exp_cfg, args.random_seed)
    rng = np.random.default_rng(random_seed)
    records: dict[str, dict] = dict(existing)

    for count, item in enumerate(items, start=1):
        output_path = output_dir / Path(item.audio_path).name
        if item.item_id in existing and output_path.exists():
            continue
        steering = None
        selected_attributes: list[str] = []
        schedule_meta = None
        if geometry is not None:
            if args.task in {"static", "composed"}:
                selected_attributes = match_catalog_controls(item.prompt, catalog)
                if args.task == "static":
                    selected_attributes = selected_attributes[:1]
                coordinate = compose_variant_coordinate(
                    geometry.coordinates, selected_attributes,
                    mode=("sum" if args.task == "static" else spec.composition_mode),
                )
                reference_direction = direction_from_coordinate(geometry.basis, coordinate)
                if spec.full_space:
                    full_directions = full_space_composition_direction(
                        geometry.prototypes,
                        selected_attributes,
                        reference_direction=reference_direction,
                    )
                else:
                    full_directions = reference_direction
            else:
                start, end, instruction_mode = parse_dynamic_control(item.prompt)
                dimension = item.dimensions[0]
                start_key, end_key = attribute_key(dimension, start), attribute_key(dimension, end)
                if start_key not in geometry.coordinates or end_key not in geometry.coordinates:
                    raise KeyError(f"missing dynamic endpoint coordinate(s): {start_key}, {end_key}")
                coordinate_schedule = dynamic_variant_schedule(
                    geometry.coordinates[start_key], geometry.coordinates[end_key],
                    variant=spec.dynamic_variant, instruction_mode=instruction_mode,
                    steps=int(exp_cfg["steering_steps"]), transition_at=float(exp_cfg["transition_at"]),
                )
                full_directions = np.stack([
                    direction_from_coordinate(geometry.basis, coordinate)
                    for coordinate in coordinate_schedule
                ])
                selected_attributes = [start_key, end_key]
                schedule_meta = {"instruction_mode": instruction_mode, "variant": spec.dynamic_variant,
                                 "start": start_key, "end": end_key}
            selected_directions = select_layer_chunks(
                full_directions, all_layers=geometry.all_layers, selected_layers=geometry.selected_layers
            )
            if spec.random_control:
                selected_directions = norm_matched_random_directions(selected_directions, rng)
            backend._ensure_loaded()
            mc = backend.model.config
            steering = ScheduledFusedQKVSteerer(
                backend.model, layer_indices=geometry.selected_layers, directions=selected_directions,
                num_attention_heads=int(mc.num_attention_heads), kv_channels=int(mc.kv_channels),
                multi_query_group_num=int(mc.multi_query_group_num), scale=float(args.alpha),
            )

        seed = int(exp_cfg["generation_seed"]) + _item_index(item.item_id)
        generation = backend.generate_from_audio_instruction(item.audio_path, output_path, seed=seed, steering=steering)
        target_text = extract_target_text(item.prompt)
        records[item.item_id] = {
            "item_id": item.item_id, "task": args.task, "split": args.split,
            "variant": args.variant, "alpha": 0.0 if spec.prompt_only else float(args.alpha),
            "basis_kind": spec.basis_kind if not spec.prompt_only else None,
            "direction_space": "full" if spec.full_space else ("geometry" if not spec.prompt_only else None),
            "rank": spec.rank if not spec.prompt_only else None,
            "layers": list(selected_layers) if not spec.prompt_only else [],
            "composition_mode": spec.composition_mode if args.task == "composed" else None,
            "dynamic_variant": spec.dynamic_variant if args.task == "dynamic" else None,
            "dimensions": list(item.dimensions), "prompt": item.prompt, "target_text": target_text,
            "input_audio": item.audio_path, "output_audio": generation.wav_path,
            "selected_attributes": selected_attributes, "dynamic_schedule": schedule_meta,
            "random_seed": random_seed if spec.random_control else None,
            "seed": seed, "transcript": generation.generation.transcript,
            "text_channel_wer": word_error_rate(target_text, generation.generation.transcript),
            "audio_token_count": len(generation.generation.audio_token_ids),
        }
        write_manifest(manifest_path, [records[key] for key in sorted(records)])
        if count % 10 == 0:
            print(f"{count}/{len(items)} requested items processed")

    print(json.dumps({"task": args.task, "split": args.split, "variant": args.variant,
                      "alpha": float(args.alpha), "requested_items": len(items),
                      "manifest": str(manifest_path), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
