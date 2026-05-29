from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import SupportsBytes

import numpy as np

from . import _core
from .io import read_wav
from .types import FrameAnalysis, Segment, StreamingVadEvent, VadConfig, VadResult

_PCM_SAMPLE_WIDTHS = {
    "u8": 1,
    "s16le": 2,
    "s16be": 2,
    "f32le": 4,
}


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


class StreamingVoiceActivityDetector:
    """Incremental VAD wrapper for chunked PCM input.

    `push` and `push_pcm` return only segments whose end is far enough behind
    the current write position to be considered stable. Call `flush` once at end
    of stream to emit any remaining segments.
    """

    def __init__(
        self,
        sample_rate: int,
        config: VadConfig | None = None,
        on_start_of_speech: Callable[[StreamingVadEvent], None] | None = None,
        on_end_of_speech: Callable[[StreamingVadEvent], None] | None = None,
        **overrides: object,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self.detector = VoiceActivityDetector(config, **overrides)
        self.on_start_of_speech = on_start_of_speech
        self.on_end_of_speech = on_end_of_speech
        self._audio = np.empty(0, dtype=np.float32)
        self._pcm_remainder = b""
        self._pending_start_sample: int | None = None
        self._emitted_until_samples = 0
        self._closed = False

    def push(self, samples: Iterable[float] | np.ndarray) -> list[Segment]:
        """Append normalized mono float PCM samples and return stable segments."""
        self._ensure_open()
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("samples must be a mono 1-D array")
        if audio.size == 0:
            return []
        audio = np.ascontiguousarray(np.nan_to_num(audio, copy=False), dtype=np.float32)
        self._audio = np.concatenate([self._audio, audio])
        return self._collect_events(final=False)[1]

    def push_events(self, samples: Iterable[float] | np.ndarray) -> list[StreamingVadEvent]:
        """Append normalized mono float PCM samples and return stable VAD events."""
        self._ensure_open()
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("samples must be a mono 1-D array")
        if audio.size == 0:
            return []
        audio = np.ascontiguousarray(np.nan_to_num(audio, copy=False), dtype=np.float32)
        self._audio = np.concatenate([self._audio, audio])
        return self._collect_events(final=False)[0]

    def push_pcm(
        self,
        data: bytes | bytearray | memoryview | SupportsBytes,
        *,
        sample_format: str = "s16le",
        channels: int = 1,
    ) -> list[Segment]:
        """Append raw PCM bytes and return stable segments.

        Supported formats are `s16le`, `s16be`, `f32le`, and `u8`. Multi-channel
        input is downmixed to mono by averaging channels.
        """
        self._ensure_open()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if sample_format not in _PCM_SAMPLE_WIDTHS:
            raise ValueError(f"unsupported raw PCM sample format: {sample_format}")

        raw = self._pcm_remainder + bytes(data)
        frame_width = _PCM_SAMPLE_WIDTHS[sample_format] * channels
        usable = (len(raw) // frame_width) * frame_width
        self._pcm_remainder = raw[usable:]
        if usable == 0:
            return []
        return self.push(_decode_pcm_bytes(raw[:usable], sample_format, channels))

    def push_pcm_events(
        self,
        data: bytes | bytearray | memoryview | SupportsBytes,
        *,
        sample_format: str = "s16le",
        channels: int = 1,
    ) -> list[StreamingVadEvent]:
        """Append raw PCM bytes and return stable VAD events."""
        self._ensure_open()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if sample_format not in _PCM_SAMPLE_WIDTHS:
            raise ValueError(f"unsupported raw PCM sample format: {sample_format}")

        raw = self._pcm_remainder + bytes(data)
        frame_width = _PCM_SAMPLE_WIDTHS[sample_format] * channels
        usable = (len(raw) // frame_width) * frame_width
        self._pcm_remainder = raw[usable:]
        if usable == 0:
            return []
        return self.push_events(_decode_pcm_bytes(raw[:usable], sample_format, channels))

    def flush(self) -> list[Segment]:
        """Mark the stream complete and return all not-yet-emitted segments."""
        self._ensure_open()
        self._closed = True
        if self._pcm_remainder:
            raise ValueError("stream ended with incomplete PCM frame")
        return self._collect_events(final=True)[1]

    def flush_events(self) -> list[StreamingVadEvent]:
        """Mark the stream complete and return all not-yet-emitted VAD events."""
        self._ensure_open()
        self._closed = True
        if self._pcm_remainder:
            raise ValueError("stream ended with incomplete PCM frame")
        return self._collect_events(final=True)[0]

    def reset(self) -> None:
        """Clear buffered audio and allow the detector to be reused."""
        self._audio = np.empty(0, dtype=np.float32)
        self._pcm_remainder = b""
        self._pending_start_sample = None
        self._emitted_until_samples = 0
        self._closed = False

    @property
    def buffered_samples(self) -> int:
        return int(self._audio.shape[0])

    @property
    def emitted_until(self) -> float:
        return self._emitted_until_samples / self.sample_rate

    def _collect_events(self, *, final: bool) -> tuple[list[StreamingVadEvent], list[Segment]]:
        if self._audio.size == 0:
            return [], []

        result = self.detector.analyze(self._audio, self.sample_rate)
        sample_count = int(self._audio.shape[0])
        start_cutoff_samples = sample_count
        end_cutoff_samples = sample_count
        if not final:
            start_cutoff_samples = max(0, sample_count - self._start_settle_samples())
            end_cutoff_samples = max(0, sample_count - self._end_settle_samples())

        events: list[StreamingVadEvent] = []
        emitted: list[Segment] = []
        for segment in result.segments:
            start_sample = int(round(segment.start * self.sample_rate))
            end_sample = int(round(segment.end * self.sample_rate))
            if end_sample <= self._emitted_until_samples:
                continue

            if self._pending_start_sample is None and (
                final or start_sample <= start_cutoff_samples
            ):
                event = StreamingVadEvent(
                    kind="start_of_speech",
                    time=segment.start,
                    segment=None,
                )
                events.append(event)
                self._emit_callback(event)
                self._pending_start_sample = start_sample

            if not final and end_sample > end_cutoff_samples:
                continue

            start = max(start_sample, self._emitted_until_samples) / self.sample_rate
            emitted_segment = Segment(start=start, end=segment.end, confidence=segment.confidence)
            event = StreamingVadEvent(
                kind="end_of_speech",
                time=segment.end,
                segment=emitted_segment,
            )
            events.append(event)
            self._emit_callback(event)
            emitted.append(emitted_segment)
            self._emitted_until_samples = max(self._emitted_until_samples, end_sample)
            self._pending_start_sample = None
        return events, emitted

    def _start_settle_samples(self) -> int:
        config = self.detector.config
        settle_ms = config.frame_ms + config.min_speech_ms
        return int(round(self.sample_rate * settle_ms / 1000.0))

    def _end_settle_samples(self) -> int:
        config = self.detector.config
        settle_ms = config.frame_ms + config.min_silence_ms + config.speech_pad_ms
        return int(round(self.sample_rate * settle_ms / 1000.0))

    def _emit_callback(self, event: StreamingVadEvent) -> None:
        if event.kind == "start_of_speech" and self.on_start_of_speech is not None:
            self.on_start_of_speech(event)
        if event.kind == "end_of_speech" and self.on_end_of_speech is not None:
            self.on_end_of_speech(event)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("cannot push after flush; call reset() to start a new stream")


def _decode_pcm_bytes(raw: bytes, sample_format: str, channels: int) -> np.ndarray:
    if sample_format == "s16le":
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_format == "s16be":
        audio = np.frombuffer(raw, dtype=">i2").astype(np.float32) / 32768.0
    elif sample_format == "f32le":
        audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif sample_format == "u8":
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported raw PCM sample format: {sample_format}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


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
