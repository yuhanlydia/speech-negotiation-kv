from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


_QUOTED_TEXT_RE = re.compile(r"(?:^|:)\s*['\"](.+?)['\"]\s*$")
_STATIC_CONTROL_RE = re.compile(r"Please read this sentence with (.+?):\s*['\"]", re.IGNORECASE)
_DYNAMIC_START_RE = re.compile(r"starting with (.+?) and (.+?):\s*['\"]", re.IGNORECASE)

_TRANSITION_PREFIXES = (
    "gradually transitioning to ",
    "gradually becoming ",
    "gradually increasing to ",
    "gradually increasing the volume to ",
    "gradually decreasing the volume to ",
    "gradually raising the pitch to ",
    "gradually lowering the pitch to ",
    "gradually slowing down to ",
    "suddenly dropping to ",
    "suddenly jumping to ",
    "suddenly switching to ",
    "suddenly becoming ",
    "suddenly speeding up to ",
    "suddenly increasing the volume in ",
)


@dataclass(frozen=True)
class SpeechParalingItem:
    item_id: str
    prompt: str
    dimensions: tuple[str, ...]
    task: str
    audio_path: str | None = None


def normalize_control_phrase(text: str) -> str:
    value = " ".join(str(text).strip().lower().split())
    for article in ("a ", "an "):
        if value.startswith(article):
            value = value[len(article):]
    return value


def extract_target_text(prompt: str) -> str:
    match = _QUOTED_TEXT_RE.search(prompt.strip())
    if not match:
        raise ValueError(f"could not extract quoted target text: {prompt}")
    return match.group(1)


def parse_static_control(prompt: str) -> str:
    match = _STATIC_CONTROL_RE.search(prompt)
    if not match:
        raise ValueError(f"not a static-control prompt: {prompt}")
    return normalize_control_phrase(match.group(1))


def parse_dynamic_control(prompt: str) -> tuple[str, str, str]:
    match = _DYNAMIC_START_RE.search(prompt)
    if not match:
        raise ValueError(f"not a dynamic-control prompt: {prompt}")
    start = normalize_control_phrase(match.group(1))
    transition = " ".join(match.group(2).strip().lower().split())
    end = transition
    mode = "linear"
    if "suddenly" in transition:
        mode = "step"
    for prefix in _TRANSITION_PREFIXES:
        if transition.startswith(prefix):
            end = transition[len(prefix):]
            break
    else:
        marker = " to "
        if marker in transition:
            end = transition.rsplit(marker, 1)[-1]
    return start, normalize_control_phrase(end), mode


def load_prompt_jsonl(path: str | Path, *, task: str) -> list[SpeechParalingItem]:
    source = Path(path)
    rows: list[SpeechParalingItem] = []
    with source.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(SpeechParalingItem(
                item_id=f"{source.stem}:{index:04d}",
                prompt=str(payload["prompt"]),
                dimensions=tuple(str(value) for value in payload.get("dimensions", [])),
                task=str(task),
            ))
    return rows


def _filename_sample_index(path: Path) -> int | None:
    match = re.search(r"_(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def pair_with_audio(items: Sequence[SpeechParalingItem], audio_dir: str | Path) -> list[SpeechParalingItem]:
    audio_paths = sorted(Path(audio_dir).glob("*.wav"))
    if len(audio_paths) != len(items):
        raise ValueError(f"prompt/audio count mismatch: {len(items)} prompts vs {len(audio_paths)} wav files")
    indices = [_filename_sample_index(path) for path in audio_paths]
    if all(index is not None and 1 <= index <= len(items) for index in indices) and len(set(indices)) == len(indices):
        pairs = [(items[index - 1], path) for index, path in zip(indices, audio_paths)]
    else:
        pairs = list(zip(items, audio_paths))
    return [SpeechParalingItem(
        item_id=item.item_id,
        prompt=item.prompt,
        dimensions=item.dimensions,
        task=item.task,
        audio_path=str(audio_path),
    ) for item, audio_path in pairs]


def select_pilot(items: Iterable[SpeechParalingItem], *, dimensions: Sequence[str], limit: int) -> list[SpeechParalingItem]:
    allowed = {str(value) for value in dimensions}
    selected = [item for item in items if any(dim in allowed for dim in item.dimensions)]
    return selected[: int(limit)]


def attribute_key(dimension: str, control: str) -> str:
    return f"{str(dimension).strip()}::{normalize_control_phrase(control)}"


def match_catalog_controls(prompt: str, catalog: dict[str, dict]) -> list[str]:
    """Return catalog attribute keys whose control phrase appears in the instruction.

    This is intentionally an oracle-control parser for the first ParaGeo pilot:
    it tests steering geometry independently of speech-ASR/routing quality.
    """
    text = " ".join(prompt.lower().split())
    candidates = []
    for key, meta in catalog.items():
        phrase = normalize_control_phrase(meta["control"])
        if phrase and phrase in text:
            candidates.append((len(phrase), str(key)))
    candidates.sort(reverse=True)
    selected: list[str] = []
    used_phrases: list[str] = []
    for _, key in candidates:
        phrase = normalize_control_phrase(catalog[key]["control"])
        if any(phrase in existing or existing in phrase for existing in used_phrases):
            continue
        selected.append(key)
        used_phrases.append(phrase)
    return selected


def select_catalog_covered_items(items: Sequence[SpeechParalingItem], catalog: dict[str, dict], *,
                                 task: str, limit: int | None = None) -> list[SpeechParalingItem]:
    supported_dimensions = {str(meta.get("dimension")) for meta in catalog.values()}
    selected: list[SpeechParalingItem] = []
    for item in items:
        if not set(item.dimensions).issubset(supported_dimensions):
            continue
        if task in {"static", "composed"}:
            matched = match_catalog_controls(item.prompt, catalog)
            required = 1 if task == "static" else max(2, len(item.dimensions))
            if len(matched) < required:
                continue
        elif task == "dynamic":
            if len(item.dimensions) != 1:
                continue
            try:
                start, end, _ = parse_dynamic_control(item.prompt)
            except ValueError:
                continue
            dimension = item.dimensions[0]
            if attribute_key(dimension, start) not in catalog or attribute_key(dimension, end) not in catalog:
                continue
        else:
            raise ValueError(f"unknown task: {task}")
        selected.append(item)
        if limit is not None and len(selected) >= int(limit):
            break
    return selected


def word_error_rate(reference: str, hypothesis: str) -> float:
    def words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())
    ref = words(reference)
    hyp = words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, rword in enumerate(ref, start=1):
        current = [i]
        for j, hword in enumerate(hyp, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + int(rword != hword),
            ))
        previous = current
    return float(previous[-1] / len(ref))
