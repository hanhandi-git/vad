# openvad

`openvad` is a lightweight voice activity detection library with a C++17 signal
processing core and a Python API. It is designed for low-latency segmentation of
PCM speech audio without model downloads or runtime services.

The detector uses short-time energy, zero-crossing rate, adaptive noise-floor
tracking, hysteresis thresholds, and segment post-processing. This makes it fast,
portable, and explainable. It is a practical baseline for streaming, ASR
pre-processing, diarization pre-filters, and batch dataset cleanup.

## Features

- C++ core exposed through `pybind11`.
- Python 3.10+ API with typed dataclasses.
- `uv`-friendly packaging and development workflow.
- CLI for WAV files.
- No external model weights.
- Frame probabilities, energy traces, and final speech segments.
- Conservative defaults with tunable aggressiveness.

## Install

For local development:

```bash
uv sync --extra dev
uv run pytest
```

Build a wheel:

```bash
uv run python -m build
```

Install editable during development:

```bash
uv pip install -e ".[dev]"
```

## CLI

```bash
uv run openvad input.wav
uv run openvad input.wav --json
uv run openvad input.wav --aggressiveness 2 --onset-threshold 0.62
```

Output example:

```text
   0.240s -    1.830s ( 1.590s, confidence=0.812)
```

## Python API

```python
from openvad import VoiceActivityDetector, read_wav

samples, sample_rate = read_wav("speech.wav")
detector = VoiceActivityDetector(aggressiveness=1)
result = detector.analyze(samples, sample_rate)

for segment in result.segments:
    print(segment.start, segment.end, segment.confidence)
```

You can also pass a mono `numpy.float32` array directly:

```python
from openvad import detect

result = detect(samples, sample_rate=16_000, speech_pad_ms=60)
```

## Configuration

`VadConfig` exposes the main detector controls:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `frame_ms` | `20.0` | Analysis window length. |
| `hop_ms` | `10.0` | Step between adjacent frames. |
| `onset_threshold` | `0.58` | Probability needed to enter speech. |
| `offset_threshold` | `0.42` | Probability needed to remain in speech. |
| `min_speech_ms` | `80` | Drop shorter speech islands. |
| `min_silence_ms` | `120` | Fill shorter silence gaps inside speech. |
| `speech_pad_ms` | `40` | Expand final speech regions on both sides. |
| `aggressiveness` | `1` | `0` is permissive, `3` is strict. |

Recommended starting points:

- Clean microphone speech: `aggressiveness=1`.
- Noisy environment: `aggressiveness=2` or raise `onset_threshold`.
- Avoid clipping words at boundaries: increase `speech_pad_ms` to `60`-`100`.
- Very short commands: lower `min_speech_ms` to `40`.

## Accuracy Notes

This project intentionally starts with a high-performance statistical VAD rather
than a neural model. It performs well when speech energy is measurably above the
local noise floor. It is less suitable for music-heavy audio, overlapping
speakers in loud rooms, or speech buried under non-stationary noise.

For production systems, evaluate on your target audio and tune thresholds with
held-out data. The exposed frame-level `probability`, `energy_db`, and `zcr`
arrays are meant to make this tuning straightforward.

## Architecture

- `native/vad_core.cpp`: frame feature extraction, adaptive probability, and
  post-processing.
- `src/openvad/api.py`: public detector API and frame-to-segment conversion.
- `src/openvad/io.py`: small PCM WAV reader with mono downmixing.
- `src/openvad/cli.py`: command-line interface.
- `tests/`: synthetic regression tests.

Documentation is split by language:

- English: [docs/en](docs/en/README.md)
- 中文：[docs/zh](docs/zh/README.md)

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

The native extension is compiled by `setuptools` and `pybind11`; CMake is not
required.

## Validation Tools

Generate a synthetic labeled dataset:

```bash
uv run python tools/make_synthetic_dataset.py --output data/synth --count 20
```

Evaluate a labeled manifest:

```bash
uv run python tools/evaluate_dataset.py data/synth/manifest.jsonl
```

Sweep thresholds:

```bash
uv run python tools/sweep_thresholds.py data/synth/manifest.jsonl
```

Inspect one file with an HTML report:

```bash
uv run python tools/inspect_file.py data/synth/sample_000.wav --output report.html
```

More details:

- English: [docs/en/tools.md](docs/en/tools.md)
- 中文：[docs/zh/tools.md](docs/zh/tools.md)

## License

MIT
