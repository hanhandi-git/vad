from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from openvad import VadConfig, VoiceActivityDetector, read_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an HTML inspection report for one WAV file."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("openvad_report.html"))
    parser.add_argument("--labels", type=Path, help="Optional JSON file with [[start, end], ...].")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--sample-format",
        default="s16le",
        choices=["s16le", "s16be", "f32le", "u8"],
    )
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--aggressiveness", type=int, choices=range(4), default=1)
    parser.add_argument("--onset-threshold", type=float, default=0.58)
    parser.add_argument("--offset-threshold", type=float, default=0.42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = load_labels(args.labels) if args.labels else []
    config = VadConfig(
        aggressiveness=args.aggressiveness,
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
    )
    samples, sample_rate = read_audio(
        args.audio,
        sample_rate=args.sample_rate,
        sample_format=args.sample_format,
        channels=args.channels,
    )
    result = VoiceActivityDetector(config).analyze(samples, sample_rate)
    args.output.write_text(render_html(args.audio, result, labels), encoding="utf-8")
    print(f"wrote: {args.output}")
    return 0


def load_labels(path: Path) -> list[tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(float(start), float(end)) for start, end in payload]


def render_html(audio_path: Path, result: object, labels: list[tuple[float, float]]) -> str:
    frames = result.frames
    times = (
        np.arange(len(frames.probability), dtype=np.float32)
        * frames.hop_samples
        / result.sample_rate
    )
    probability = downsample(frames.probability)
    energy = normalize_series(frames.energy_db)
    zcr = normalize_series(frames.zcr)
    speech_runs = interval_bars(
        [(s.start, s.end) for s in result.segments],
        result.duration,
        "#1f9d55",
    )
    label_runs = interval_bars(labels, result.duration, "#d97706")
    escaped_name = html.escape(str(audio_path))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>openvad report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #172033; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{ background: #f6f8fb; border: 1px solid #dbe2ea; border-radius: 10px; padding: 14px; }}
    .label {{ color: #62708a; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .track {{ position: relative; height: 28px; background: #edf1f7;
      border-radius: 6px; margin: 8px 0 22px; }}
    .bar {{ position: absolute; top: 0; bottom: 0; border-radius: 6px;
      opacity: .86; }}
    svg {{ width: 100%; height: 220px; border: 1px solid #dbe2ea;
      border-radius: 10px; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #dbe2ea; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>openvad report</h1>
  <p>{escaped_name}</p>
  <div class="summary">
    <div class="card">
      <div class="label">Duration</div><div class="value">{result.duration:.3f}s</div>
    </div>
    <div class="card">
      <div class="label">Sample rate</div><div class="value">{result.sample_rate}</div>
    </div>
    <div class="card">
      <div class="label">Segments</div><div class="value">{len(result.segments)}</div>
    </div>
    <div class="card">
      <div class="label">Noise floor</div>
      <div class="value">{frames.noise_floor_db:.1f} dB</div>
    </div>
  </div>
  <h2>Predicted speech</h2>
  <div class="track">{speech_runs}</div>
  <h2>Reference labels</h2>
  <div class="track">{label_runs}</div>
  <h2>Frame traces</h2>
  <svg viewBox="0 0 1000 220" preserveAspectRatio="none">
    <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{polyline(probability)}" />
    <polyline fill="none" stroke="#dc2626" stroke-width="2" points="{polyline(energy)}" />
    <polyline fill="none" stroke="#16a34a" stroke-width="2" points="{polyline(zcr)}" />
  </svg>
  <p>Blue: probability. Red: normalized energy. Green: normalized zero-crossing rate.</p>
  <h2>Segments</h2>
  {segments_table(result.segments)}
  <script type="application/json" id="openvad-times">{json.dumps(times.tolist())}</script>
</body>
</html>
"""


def downsample(values: np.ndarray, limit: int = 1000) -> np.ndarray:
    if len(values) <= limit:
        return values.astype(np.float32, copy=False)
    indices = np.linspace(0, len(values) - 1, limit).astype(np.int64)
    return values[indices].astype(np.float32, copy=False)


def normalize_series(values: np.ndarray) -> np.ndarray:
    values = downsample(values)
    lo = float(np.min(values)) if len(values) else 0.0
    hi = float(np.max(values)) if len(values) else 1.0
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def polyline(values: np.ndarray) -> str:
    if len(values) == 0:
        return ""
    xs = np.linspace(0, 1000, len(values))
    ys = 210 - np.clip(values, 0.0, 1.0) * 200
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=True))


def interval_bars(segments: list[tuple[float, float]], duration: float, color: str) -> str:
    if duration <= 0:
        return ""
    bars = []
    for start, end in segments:
        left = max(0.0, min(100.0, 100.0 * start / duration))
        width = max(0.0, min(100.0 - left, 100.0 * (end - start) / duration))
        bars.append(
            '<div class="bar" '
            f'style="left:{left:.3f}%;width:{width:.3f}%;background:{color}">'
            "</div>"
        )
    return "\n".join(bars)


def segments_table(segments: object) -> str:
    if not segments:
        return "<p>No speech segments detected.</p>"
    rows = "\n".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{segment.start:.3f}</td>"
        f"<td>{segment.end:.3f}</td>"
        f"<td>{segment.duration:.3f}</td>"
        f"<td>{segment.confidence:.3f}</td>"
        "</tr>"
        for index, segment in enumerate(segments, start=1)
    )
    return (
        "<table><thead><tr><th>#</th><th>Start</th><th>End</th>"
        "<th>Duration</th><th>Confidence</th></tr></thead><tbody>"
        f"{rows}</tbody></table>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
