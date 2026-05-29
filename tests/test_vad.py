from __future__ import annotations

import wave

import numpy as np

from openvad import (
    StreamingVadEvent,
    StreamingVoiceActivityDetector,
    VoiceActivityDetector,
    detect_file,
)


def test_silence_has_no_segments() -> None:
    samples = np.zeros(16_000, dtype=np.float32)
    result = VoiceActivityDetector().analyze(samples, 16_000)
    assert result.segments == []
    assert result.frames.speech.sum() == 0


def test_detects_clear_voiced_region() -> None:
    sample_rate = 16_000
    samples = np.random.default_rng(7).normal(0.0, 0.002, sample_rate).astype(np.float32)
    start = int(0.25 * sample_rate)
    end = int(0.75 * sample_rate)
    t = np.arange(end - start, dtype=np.float32) / sample_rate
    samples[start:end] += 0.22 * np.sin(2 * np.pi * 180 * t).astype(np.float32)

    result = VoiceActivityDetector(aggressiveness=0).analyze(samples, sample_rate)

    assert len(result.segments) == 1
    segment = result.segments[0]
    assert 0.15 <= segment.start <= 0.35
    assert 0.65 <= segment.end <= 0.85
    assert segment.confidence > 0.5


def test_detect_file_reads_pcm16_wav(tmp_path) -> None:
    sample_rate = 16_000
    samples = np.zeros(sample_rate // 2, dtype=np.float32)
    wav_path = tmp_path / "silence.wav"
    pcm = (samples * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    result = detect_file(wav_path)

    assert result.sample_rate == sample_rate
    assert result.duration == 0.5


def test_streaming_detector_matches_batch_segments_after_flush() -> None:
    sample_rate = 16_000
    samples = _speech_like_samples(sample_rate)

    batch = VoiceActivityDetector(aggressiveness=0).analyze(samples, sample_rate)
    stream = StreamingVoiceActivityDetector(sample_rate, aggressiveness=0)

    emitted = []
    for offset in range(0, len(samples), 1377):
        emitted.extend(stream.push(samples[offset : offset + 1377]))
    emitted.extend(stream.flush())

    assert len(emitted) == len(batch.segments) == 1
    assert abs(emitted[0].start - batch.segments[0].start) <= 0.02
    assert abs(emitted[0].end - batch.segments[0].end) <= 0.02
    assert emitted[0].confidence > 0.5


def test_streaming_detector_accepts_chunked_s16le_pcm() -> None:
    sample_rate = 16_000
    samples = _speech_like_samples(sample_rate)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    stream = StreamingVoiceActivityDetector(sample_rate, aggressiveness=0)

    emitted = []
    for offset in range(0, len(pcm), 513):
        emitted.extend(stream.push_pcm(pcm[offset : offset + 513], sample_format="s16le"))
    emitted.extend(stream.flush())

    assert len(emitted) == 1
    assert 0.15 <= emitted[0].start <= 0.35
    assert 0.65 <= emitted[0].end <= 0.90


def test_streaming_detector_emits_start_and_end_events() -> None:
    sample_rate = 16_000
    samples = _speech_like_samples(sample_rate)
    stream = StreamingVoiceActivityDetector(sample_rate, aggressiveness=0)

    events = []
    first_start_index = None
    for offset in range(0, len(samples), 1600):
        chunk_events = stream.push_events(samples[offset : offset + 1600])
        if first_start_index is None and chunk_events:
            first_start_index = len(events)
        events.extend(chunk_events)
    events.extend(stream.flush_events())

    assert [event.kind for event in events] == ["start_of_speech", "end_of_speech"]
    assert first_start_index == 0
    assert events[0].segment is None
    assert events[1].segment is not None
    assert events[0].time <= events[1].time
    assert 0.15 <= events[0].time <= 0.35
    assert 0.65 <= events[1].time <= 0.90


def test_streaming_detector_invokes_callbacks() -> None:
    sample_rate = 16_000
    samples = _speech_like_samples(sample_rate)
    callbacks: list[StreamingVadEvent] = []
    stream = StreamingVoiceActivityDetector(
        sample_rate,
        on_start_of_speech=callbacks.append,
        on_end_of_speech=callbacks.append,
        aggressiveness=0,
    )

    for offset in range(0, len(samples), 1600):
        stream.push(samples[offset : offset + 1600])
    stream.flush()

    assert [event.kind for event in callbacks] == ["start_of_speech", "end_of_speech"]
    assert callbacks[1].segment is not None


def test_streaming_detector_rejects_push_after_flush() -> None:
    stream = StreamingVoiceActivityDetector(16_000)

    assert stream.flush() == []

    try:
        stream.push(np.zeros(160, dtype=np.float32))
    except ValueError as exc:
        assert "reset" in str(exc)
    else:
        raise AssertionError("expected push after flush to fail")


def test_streaming_detector_reports_incomplete_pcm_frame_on_flush() -> None:
    stream = StreamingVoiceActivityDetector(16_000)

    assert stream.push_pcm(b"\x00", sample_format="s16le") == []

    try:
        stream.flush()
    except ValueError as exc:
        assert "incomplete PCM frame" in str(exc)
    else:
        raise AssertionError("expected incomplete PCM frame to fail")


def _speech_like_samples(sample_rate: int) -> np.ndarray:
    samples = np.random.default_rng(11).normal(0.0, 0.002, sample_rate).astype(np.float32)
    start = int(0.25 * sample_rate)
    end = int(0.75 * sample_rate)
    t = np.arange(end - start, dtype=np.float32) / sample_rate
    samples[start:end] += 0.22 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
    return samples
