from __future__ import annotations

import wave

import numpy as np

from openvad import VoiceActivityDetector, detect_file


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
