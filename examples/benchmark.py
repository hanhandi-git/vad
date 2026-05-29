from __future__ import annotations

from time import perf_counter

import numpy as np

from openvad import VoiceActivityDetector


def main() -> None:
    sample_rate = 16_000
    seconds = 600
    rng = np.random.default_rng(42)
    samples = rng.normal(0.0, 0.004, sample_rate * seconds).astype(np.float32)

    detector = VoiceActivityDetector()
    start = perf_counter()
    result = detector.analyze(samples, sample_rate)
    elapsed = perf_counter() - start

    print(f"audio: {seconds:.1f}s")
    print(f"elapsed: {elapsed:.3f}s")
    print(f"realtime factor: {seconds / elapsed:.1f}x")
    print(f"segments: {len(result.segments)}")


if __name__ == "__main__":
    main()
