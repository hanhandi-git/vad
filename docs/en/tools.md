# Validation Tools

The `tools/` directory contains small scripts for checking detector behavior on
synthetic and labeled audio.

## Generate Synthetic Data

```bash
uv run python tools/make_synthetic_dataset.py --output data/synth --count 20
```

This creates WAV files plus `manifest.jsonl` labels. Each label row contains the
audio path and reference speech intervals in seconds.

## Evaluate Labeled Audio

```bash
uv run python tools/evaluate_dataset.py data/synth/manifest.jsonl
```

The evaluator reports frame-level precision, recall, F1, false-positive time,
false-negative time, and processing speed.

Manifest format:

```json
{"audio": "sample_000.wav", "segments": [[0.3, 1.1], [1.7, 2.4]]}
```

Relative audio paths are resolved relative to the manifest file.

## Sweep Parameters

```bash
uv run python tools/sweep_thresholds.py data/synth/manifest.jsonl \
  --aggressiveness 0,1,2 \
  --onset 0.50,0.58,0.66 \
  --offset 0.34,0.42,0.50
```

Use this before changing defaults or tuning for a specific audio domain.

## Visualize One File

```bash
uv run python tools/inspect_file.py speech.wav --output report.html
```

The HTML report includes detected segments, frame probabilities, energy, and
zero-crossing rate. If a `--labels labels.json` file is supplied, reference
intervals are shown alongside predictions.

Label file format:

```json
[[0.3, 1.1], [1.7, 2.4]]
```
