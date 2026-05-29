# AISHELL-4 Benchmark

AISHELL-4 is a Mandarin meeting dataset and is useful for evaluating `openvad`
on far-field speech, multiple speakers, overlapping speech, and real meeting
noise. OpenSLR SLR111 provides the full data, and a Hugging Face mirror is also
available. The official data specification describes 16 kHz, 16-bit audio, about
120 hours total, with about 107.50 hours for training and 12.72 hours for the
evaluation set.

## Install Dependencies

```bash
uv sync --extra dev --extra bench
```

The `bench` extra installs:

- `huggingface_hub` for optional AISHELL-4 downloads from Hugging Face.
- `soundfile` for FLAC/OGG audio in the Hugging Face mirror.

## Data Sources

OpenSLR:

```text
https://www.openslr.org/111/
```

Hugging Face:

```text
https://huggingface.co/datasets/AISHELL/AISHELL-4
```

The full `test` split is about 5.2G, so it should not be downloaded in ordinary
CI jobs.

## Expected Layout

```text
data/aishell4/
  test/
    wav/
    TextGrid/
```

If you download from OpenSLR manually, make sure each audio file can be matched
to a `.rttm` annotation by exact stem or suffix stem.

## Download Test Split From Hugging Face

```bash
uv run python tools/aishell4_prepare.py \
  --root data/aishell4 \
  --split test \
  --download-hf
```

This writes:

```text
data/aishell4/test_manifest.jsonl
```

## Download Only A Few Recordings

List file names in the Hugging Face `test` split:

```bash
uv run python tools/aishell4_prepare.py --split test --list-hf-files
```

Then choose a few recording stems and download only matching audio and RTTM
files:

```bash
uv run python tools/aishell4_prepare.py \
  --root data/aishell4 \
  --split test \
  --download-hf \
  --hf-stems L_R003S01C02,L_R003S02C01 \
  --output data/aishell4/test_small_manifest.jsonl
```

`--hf-stems` is comma-separated and should not include file extensions. The
script matches:

- `test/wav/{stem}.*`
- `test/wav/*{stem}.*`
- `test/TextGrid/{stem}.rttm`
- `test/TextGrid/*{stem}.rttm`

## Smoke Benchmark

```bash
uv run python tools/aishell4_prepare.py \
  --root data/aishell4 \
  --split test \
  --max-files 3 \
  --output data/aishell4/test_smoke_manifest.jsonl

uv run python tools/evaluate_dataset.py data/aishell4/test_smoke_manifest.jsonl
```

## Full Evaluation

```bash
uv run python tools/evaluate_dataset.py data/aishell4/test_manifest.jsonl
```

## Parameter Sweep

```bash
uv run python tools/sweep_thresholds.py data/aishell4/test_manifest.jsonl \
  --aggressiveness 0,1,2,3 \
  --onset 0.50,0.54,0.58,0.62,0.66 \
  --offset 0.34,0.38,0.42,0.46,0.50 \
  --top 20
```

## Visualize One Meeting

```bash
uv run python tools/inspect_file.py data/aishell4/test/wav/example.flac \
  --output tests/reports/aishell4_example.html
```

## Notes

- AISHELL-4 RTTM files may contain overlapping speakers. The preparation script
  unions all speaker regions into one speech/non-speech reference track.
- This is appropriate for VAD, but not enough for speaker diarization.
- `evaluate_dataset.py` currently uses frame overlap without a collar. A future
  improvement is adding a 100-250 ms boundary collar option.

## Sources

- OpenSLR SLR111: <https://www.openslr.org/111/>
- Hugging Face AISHELL-4: <https://huggingface.co/datasets/AISHELL/AISHELL-4>
- AISHELL-4 paper: <https://arxiv.org/abs/2104.03603>
- AISHELL-4 data specification: <https://aishell-4.oss-cn-hangzhou.aliyuncs.com/AISHELL-4%20Data-Specification.pdf>
