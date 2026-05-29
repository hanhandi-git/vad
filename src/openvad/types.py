from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VadConfig:
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    onset_threshold: float = 0.58
    offset_threshold: float = 0.42
    min_speech_ms: int = 80
    min_silence_ms: int = 120
    speech_pad_ms: int = 40
    aggressiveness: int = 1

    def validate(self) -> None:
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if not 0 <= self.offset_threshold <= self.onset_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= offset <= onset <= 1")
        if min(self.min_speech_ms, self.min_silence_ms, self.speech_pad_ms) < 0:
            raise ValueError("duration settings must be non-negative")
        if not 0 <= self.aggressiveness <= 3:
            raise ValueError("aggressiveness must be in [0, 3]")


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    speech: np.ndarray
    probability: np.ndarray
    energy_db: np.ndarray
    zcr: np.ndarray
    frame_samples: int
    hop_samples: int
    noise_floor_db: float


@dataclass(frozen=True, slots=True)
class VadResult:
    sample_rate: int
    samples: int
    config: VadConfig
    frames: FrameAnalysis
    segments: list[Segment]

    @property
    def duration(self) -> float:
        return self.samples / self.sample_rate
