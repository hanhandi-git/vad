# Test Audio Data

`tests/data/pcm/` contains raw PCM files copied from
`/app/agora_rtc/tests/acg_data`.

Assumed format:

- sample rate: 16 kHz
- channels: mono
- sample format: signed 16-bit little-endian PCM (`s16le`)

Generate visual VAD reports:

```bash
uv run python tools/batch_inspect.py tests/data/pcm/*.pcm --output-dir tests/reports
```

The command writes one HTML report per file plus `tests/reports/summary.json`.
