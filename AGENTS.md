# Repository Guidelines

## Project Structure & Module Organization

`openvad` is a Python 3.10+ package with a C++17 VAD core exposed through
`pybind11`. Public Python code lives in `src/openvad/`: `api.py` contains the
detector API, `types.py` defines dataclasses, `io.py` handles WAV input, and
`cli.py` provides the `openvad` command. Native signal-processing code is in
`native/vad_core.cpp`. Tests and PCM fixtures are under `tests/`; generated
reports are in `tests/reports/`. Developer utilities are in `tools/`, examples
in `examples/`, and bilingual documentation in `docs/en/` and `docs/zh/`.

## Build, Test, and Development Commands

- `uv sync --extra dev`: create/update the development environment.
- `uv run pytest`: run the full test suite configured by `pyproject.toml`.
- `uv run ruff check .`: run linting and import-order checks.
- `uv run python -m build`: build source and wheel distributions.
- `uv run openvad input.wav --json`: exercise the CLI against a WAV file.
- `uv run python tools/inspect_file.py input.wav --output report.html`: produce
  a tuning/debugging report.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and keep lines at or below 100 characters.
Ruff enforces `E`, `F`, `I`, `UP`, and `B` rules. Prefer typed public APIs,
dataclasses for structured results, and small functions with explicit names.
Test functions should use `test_<behavior>` names. Keep performance-sensitive
sample/frame loops in `native/vad_core.cpp` unless Python is clearly simpler and
fast enough.

## Testing Guidelines

Use `pytest` for regression coverage. Add or update tests in `tests/test_vad.py`
for silence, noise, voiced regions, impulse noise, segment merging/padding, and
WAV decoding changes. Synthetic tests are useful for invariants; threshold or
accuracy changes should also be checked with representative real audio and the
tools in `tools/`. Keep committed fixtures small.

## Commit & Pull Request Guidelines

The current history uses short, imperative commit summaries such as
`Initial commit: Add project structure...`. Keep new commit messages concise and
action-oriented; include scope when helpful, for example `docs: clarify tuning
workflow` or `api: expose frame probabilities`.

Pull requests should describe the behavior change, list validation commands
(`uv run pytest`, `uv run ruff check .`), and link related issues. Include CLI
output or report screenshots when changing user-facing behavior or tuning
defaults.

## Security & Configuration Tips

Do not commit large audio datasets, model weights, private recordings, or local
environment files. Prefer small synthetic fixtures or documented external data
sources. Avoid adding heavy runtime dependencies without a clear accuracy or
usability benefit.
