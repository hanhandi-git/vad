# Contributing

## Development Setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Native Extension

The core detector lives in `native/vad_core.cpp` and is exposed to Python through
`pybind11`. Keep sample-level loops in C++ unless Python improves readability
without affecting runtime.

## Testing

Add regression tests for:

- Silence and low-level noise.
- Clear speech-like regions.
- Short impulse noise.
- Segment merging and padding behavior.
- WAV decoding edge cases.

Synthetic tests are useful for invariants, but accuracy changes should also be
checked against real target-domain audio before release.

## Style

- Prefer small public APIs with typed dataclasses.
- Keep defaults conservative.
- Document tuning behavior when changing thresholds.
- Avoid adding large runtime dependencies without a clear accuracy or usability
  gain.
