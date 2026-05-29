from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _common import write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic WAV files with VAD labels.")
    parser.add_argument("--output", type=Path, default=Path("data/synth"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(args.count):
            samples, segments = synthesize_sample(rng, args.sample_rate, args.duration)
            wav_path = args.output / f"sample_{index:03d}.wav"
            write_wav(wav_path, samples, args.sample_rate)
            manifest.write(
                json.dumps(
                    {
                        "audio": wav_path.name,
                        "segments": [[round(s, 3), round(e, 3)] for s, e in segments],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote: {manifest_path}")
    return 0


def synthesize_sample(
    rng: np.random.Generator,
    sample_rate: int,
    duration: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    sample_count = int(sample_rate * duration)
    samples = rng.normal(0.0, rng.uniform(0.001, 0.006), sample_count).astype(np.float32)
    segments: list[tuple[float, float]] = []
    cursor = rng.uniform(0.15, 0.45)
    while cursor < duration - 0.45:
        speech_duration = float(rng.uniform(0.25, 0.85))
        start = cursor
        end = min(duration - 0.1, start + speech_duration)
        add_voiced_region(samples, sample_rate, start, end, rng)
        segments.append((start, end))
        cursor = end + float(rng.uniform(0.2, 0.7))
    add_impulses(samples, rng)
    return np.clip(samples, -1.0, 1.0), segments


def add_voiced_region(
    samples: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    rng: np.random.Generator,
) -> None:
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    length = max(0, end_idx - start_idx)
    if length == 0:
        return
    t = np.arange(length, dtype=np.float32) / sample_rate
    f0 = float(rng.uniform(120.0, 240.0))
    envelope_phase = np.sin(np.linspace(0.0, np.pi, length, dtype=np.float32))
    envelope = np.maximum(envelope_phase, 0.0) ** 0.5
    voice = (
        np.sin(2 * np.pi * f0 * t)
        + 0.45 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.20 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    amplitude = float(rng.uniform(0.12, 0.35))
    samples[start_idx:end_idx] += (amplitude * envelope * voice).astype(np.float32)


def add_impulses(samples: np.ndarray, rng: np.random.Generator) -> None:
    impulse_count = int(rng.integers(0, 4))
    if impulse_count == 0:
        return
    indices = rng.integers(0, len(samples), impulse_count)
    for index in indices:
        width = int(rng.integers(1, 8))
        end = min(len(samples), index + width)
        samples[index:end] += rng.uniform(-0.5, 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
