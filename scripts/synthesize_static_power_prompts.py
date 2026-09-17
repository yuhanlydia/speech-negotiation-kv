#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import wave

import numpy as np


def validate_wav(path: str | Path, *, expected_sample_rate: int = 22050) -> dict:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            payload = handle.readframes(frames)
    except (EOFError, wave.Error) as error:
        raise ValueError(f"invalid wav {source}: {error}") from error
    if channels != 1:
        raise ValueError(f"expected mono wav, got {channels} channels: {source}")
    if sample_width != 2:
        raise ValueError(f"expected 16-bit wav, got {sample_width * 8}-bit: {source}")
    if sample_rate != int(expected_sample_rate):
        raise ValueError(f"unexpected sample rate {sample_rate}, expected {expected_sample_rate}: {source}")
    if frames < sample_rate // 4:
        raise ValueError(f"wav is empty or too short: {source}")
    samples = np.frombuffer(payload, dtype="<i2")
    clipped = bool(np.any(np.abs(samples.astype(np.int32)) >= 32767))
    if clipped:
        raise ValueError(f"wav is clipped: {source}")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        "peak": int(np.abs(samples.astype(np.int32)).max(initial=0)),
        "clipped": False,
    }


def pending_rows(rows: list[dict], *, expected_sample_rate: int = 22050) -> list[dict]:
    pending = []
    for row in rows:
        path = Path(row["audio_path"])
        try:
            validate_wav(path, expected_sample_rate=expected_sample_rate)
        except (FileNotFoundError, ValueError):
            pending.append(row)
    return pending


def retry_delays(attempts: int) -> list[int]:
    return [min(30, 2 ** (index + 1)) for index in range(int(attempts))]


async def synthesize_one(row: dict, *, voice: str, sample_rate: int) -> dict:
    import edge_tts
    import torch
    import torchaudio

    output = Path(row["audio_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        mp3_path = Path(handle.name)
    try:
        communicate = edge_tts.Communicate(row["prompt"], voice)
        await communicate.save(str(mp3_path))
        waveform, source_rate = torchaudio.load(str(mp3_path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if int(source_rate) != int(sample_rate):
            waveform = torchaudio.functional.resample(waveform, int(source_rate), int(sample_rate))
        peak = float(waveform.abs().max())
        if peak >= 0.999:
            waveform = waveform * (0.95 / peak)
        torchaudio.save(str(output), waveform.to(torch.float32), int(sample_rate), encoding="PCM_S", bits_per_sample=16)
        audit = validate_wav(output, expected_sample_rate=sample_rate)
        return {**row, "tts_engine": "edge-tts", "tts_voice": voice, "audio_audit": audit}
    finally:
        mp3_path.unlink(missing_ok=True)


async def synthesize_with_retry(row: dict, *, voice: str, sample_rate: int,
                                attempts: int = 8) -> dict:
    import aiohttp
    import edge_tts

    delays = retry_delays(attempts)
    for index, delay in enumerate(delays):
        try:
            return await synthesize_one(row, voice=voice, sample_rate=sample_rate)
        except (aiohttp.ClientError, edge_tts.exceptions.EdgeTTSException):
            if index == len(delays) - 1:
                raise
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable retry state")


async def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line]
    pending = pending_rows(rows, expected_sample_rate=args.sample_rate)
    audit_by_id = {}
    if Path(args.audit_output).exists():
        audit_by_id = {row["item_id"]: row for row in (
            json.loads(line) for line in Path(args.audit_output).read_text(encoding="utf-8").splitlines() if line
        )}
    for index, row in enumerate(pending, start=1):
        result = await synthesize_with_retry(row, voice=args.voice, sample_rate=args.sample_rate)
        audit_by_id[row["item_id"]] = result
        Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_output).write_text(
            "".join(json.dumps(audit_by_id[key]) + "\n" for key in sorted(audit_by_id)), encoding="utf-8"
        )
        if index % 10 == 0:
            print(f"{index}/{len(pending)} pending prompts synthesized")
        await asyncio.sleep(1.0)
    failures = []
    for row in rows:
        try:
            validate_wav(row["audio_path"], expected_sample_rate=args.sample_rate)
        except (FileNotFoundError, ValueError) as error:
            failures.append({"item_id": row["item_id"], "error": str(error)})
    if failures:
        raise RuntimeError(f"audio audit failed for {len(failures)} items: {failures[:3]}")
    print(json.dumps({"items": len(rows), "synthesized": len(pending), "voice": args.voice, "failures": 0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize and audit static-power input prompts")
    parser.add_argument("--dataset", default="results/icassp2027_static_power/dataset.jsonl")
    parser.add_argument("--audit-output", default="results/icassp2027_static_power/input_audio_audit.jsonl")
    parser.add_argument("--voice", default="en-US-AriaNeural")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
