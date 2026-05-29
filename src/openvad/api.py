from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import _core
from .io import read_wav
from .types import FrameAnalysis, Segment, VadConfig, VadResult


class VoiceActivityDetector:
    def __init__(self, config: VadConfig | None = None, **overrides: object) -> None:
        if config is None:
            config = VadConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        config.validate()
        self.config = config

    def analyze(self, samples: Iterable[float] | np.ndarray, sample_rate: int) -> VadResult:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("samples must be a mono 1-D array")
        audio = np.ascontiguousarray(np.nan_to_num(audio, copy=False), dtype=np.float32)

        raw = _core.analyze(
            audio,
            sample_rate,
            self.config.frame_ms,
            self.config.hop_ms,
            self.config.onset_threshold,
            self.config.offset_threshold,
            self.config.min_speech_ms,
            self.config.min_silence_ms,
            self.config.speech_pad_ms,
            self.config.aggressiveness,
        )
        frames = FrameAnalysis(
            speech=np.asarray(raw["speech"], dtype=np.uint8),
            probability=np.asarray(raw["probability"], dtype=np.float32),
            energy_db=np.asarray(raw["energy_db"], dtype=np.float32),
            zcr=np.asarray(raw["zcr"], dtype=np.float32),
            frame_samples=int(raw["frame_samples"]),
            hop_samples=int(raw["hop_samples"]),
            noise_floor_db=float(raw["noise_floor_db"]) if "noise_floor_db" in raw else -120.0,
        )
        return VadResult(
            sample_rate=sample_rate,
            samples=int(audio.shape[0]),
            config=self.config,
            frames=frames,
            segments=_segments_from_frames(frames, sample_rate, int(audio.shape[0])),
        )

    def detect_file(self, path: str | Path) -> VadResult:
        samples, sample_rate = read_wav(path)
        return self.analyze(samples, sample_rate)


def detect(
    samples: Iterable[float] | np.ndarray,
    sample_rate: int,
    config: VadConfig | None = None,
    **overrides: object,
) -> VadResult:
    return VoiceActivityDetector(config, **overrides).analyze(samples, sample_rate)


def detect_file(
    path: str | Path,
    config: VadConfig | None = None,
    **overrides: object,
) -> VadResult:
    return VoiceActivityDetector(config, **overrides).detect_file(path)


def _segments_from_frames(
    frames: FrameAnalysis,
    sample_rate: int,
    sample_count: int,
) -> list[Segment]:
    speech = frames.speech.astype(bool, copy=False)
    segments: list[Segment] = []
    i = 0
    n = len(speech)
    while i < n:
        while i < n and not speech[i]:
            i += 1
        start_frame = i
        while i < n and speech[i]:
            i += 1
        end_frame = i
        if end_frame > start_frame:
            start_sample = start_frame * frames.hop_samples
            end_sample = min(
                sample_count,
                (end_frame - 1) * frames.hop_samples + frames.frame_samples,
            )
            confidence = float(np.mean(frames.probability[start_frame:end_frame]))
            segments.append(
                Segment(
                    start=start_sample / sample_rate,
                    end=end_sample / sample_rate,
                    confidence=confidence,
                )
            )
    return segments
