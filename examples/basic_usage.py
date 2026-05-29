from __future__ import annotations

from openvad import VoiceActivityDetector, read_wav

samples, sample_rate = read_wav("speech.wav")
detector = VoiceActivityDetector(aggressiveness=1)
result = detector.analyze(samples, sample_rate)

for segment in result.segments:
    print(f"{segment.start:.2f}s -> {segment.end:.2f}s confidence={segment.confidence:.2f}")
