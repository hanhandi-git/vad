from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_file import render_html

from openvad import VadConfig, VoiceActivityDetector, read_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create VAD reports for multiple audio files.")
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("tests/reports"))
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = VadConfig(
        aggressiveness=args.aggressiveness,
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
    )
    detector = VoiceActivityDetector(config)
    rows = []
    for audio_path in args.audio:
        samples, sample_rate = read_audio(
            audio_path,
            sample_rate=args.sample_rate,
            sample_format=args.sample_format,
            channels=args.channels,
        )
        result = detector.analyze(samples, sample_rate)
        report_name = f"{audio_path.stem}.vad.html"
        report_path = args.output_dir / report_name
        report_path.write_text(render_html(audio_path, result, []), encoding="utf-8")
        rows.append(
            {
                "audio": str(audio_path),
                "report": str(report_path),
                "sample_rate": result.sample_rate,
                "duration": result.duration,
                "speech_seconds": sum(segment.duration for segment in result.segments),
                "speech_ratio": (
                    sum(segment.duration for segment in result.segments) / result.duration
                    if result.duration
                    else 0.0
                ),
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.duration,
                        "confidence": segment.confidence,
                    }
                    for segment in result.segments
                ],
            }
        )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {summary_path}")
    for row in rows:
        print(
            f"{Path(row['audio']).name}: duration={row['duration']:.3f}s "
            f"speech={row['speech_seconds']:.3f}s segments={len(row['segments'])} "
            f"report={row['report']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
