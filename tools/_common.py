from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from openvad import VadConfig, VoiceActivityDetector, read_audio


@dataclass(frozen=True, slots=True)
class ManifestItem:
    audio: Path
    segments: list[tuple[float, float]]


@dataclass(frozen=True, slots=True)
class Metrics:
    files: int
    audio_seconds: float
    elapsed_seconds: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    @property
    def realtime_factor(self) -> float:
        return self.audio_seconds / self.elapsed_seconds if self.elapsed_seconds else 0.0


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_manifest(path: str | Path) -> list[ManifestItem]:
    manifest_path = Path(path)
    items: list[ManifestItem] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            audio = Path(payload["audio"])
            if not audio.is_absolute():
                audio = manifest_path.parent / audio
            segments = [_validate_segment(item, line_no) for item in payload.get("segments", [])]
            items.append(ManifestItem(audio=audio, segments=segments))
    return items


def _validate_segment(value: object, line_no: int) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"manifest line {line_no}: segment must be [start, end]")
    start = float(value[0])
    end = float(value[1])
    if not 0 <= start < end:
        raise ValueError(f"manifest line {line_no}: invalid segment [{start}, {end}]")
    return start, end


def intervals_to_frames(
    segments: list[tuple[float, float]],
    frame_count: int,
    hop_samples: int,
    frame_samples: int,
    sample_rate: int,
) -> np.ndarray:
    labels = np.zeros(frame_count, dtype=np.bool_)
    for i in range(frame_count):
        frame_start = i * hop_samples / sample_rate
        frame_end = (i * hop_samples + frame_samples) / sample_rate
        for start, end in segments:
            if frame_start < end and frame_end > start:
                labels[i] = True
                break
    return labels


def evaluate_items(items: list[ManifestItem], config: VadConfig) -> Metrics:
    detector = VoiceActivityDetector(config)
    tp = fp = fn = tn = 0
    audio_seconds = 0.0
    start_time = perf_counter()
    for item in items:
        samples, sample_rate = read_audio(item.audio)
        result = detector.analyze(samples, sample_rate)
        expected = intervals_to_frames(
            item.segments,
            len(result.frames.speech),
            result.frames.hop_samples,
            result.frames.frame_samples,
            sample_rate,
        )
        predicted = result.frames.speech.astype(bool, copy=False)
        tp += int(np.logical_and(predicted, expected).sum())
        fp += int(np.logical_and(predicted, ~expected).sum())
        fn += int(np.logical_and(~predicted, expected).sum())
        tn += int(np.logical_and(~predicted, ~expected).sum())
        audio_seconds += result.duration
    return Metrics(
        files=len(items),
        audio_seconds=audio_seconds,
        elapsed_seconds=perf_counter() - start_time,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )


def print_metrics(metrics: Metrics, hop_ms: float) -> None:
    fp_seconds = metrics.false_positive * hop_ms / 1000.0
    fn_seconds = metrics.false_negative * hop_ms / 1000.0
    print(f"files: {metrics.files}")
    print(f"audio_seconds: {metrics.audio_seconds:.3f}")
    print(f"elapsed_seconds: {metrics.elapsed_seconds:.3f}")
    print(f"realtime_factor: {metrics.realtime_factor:.1f}x")
    print(f"precision: {metrics.precision:.4f}")
    print(f"recall: {metrics.recall:.4f}")
    print(f"f1: {metrics.f1:.4f}")
    print(f"false_positive_seconds: {fp_seconds:.3f}")
    print(f"false_negative_seconds: {fn_seconds:.3f}")


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
