from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import detect_file
from .types import VadConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast voice activity detection for WAV files.")
    parser.add_argument("path", type=Path, help="Input PCM WAV file.")
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--onset-threshold", type=float, default=0.58)
    parser.add_argument("--offset-threshold", type=float, default=0.42)
    parser.add_argument("--min-speech-ms", type=int, default=80)
    parser.add_argument("--min-silence-ms", type=int, default=120)
    parser.add_argument("--speech-pad-ms", type=int, default=40)
    parser.add_argument("--aggressiveness", type=int, choices=range(4), default=1)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VadConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        aggressiveness=args.aggressiveness,
    )
    result = detect_file(args.path, config)
    payload = {
        "sample_rate": result.sample_rate,
        "duration": result.duration,
        "noise_floor_db": result.frames.noise_floor_db,
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
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for segment in result.segments:
            print(
                f"{segment.start:8.3f}s - {segment.end:8.3f}s "
                f"({segment.duration:6.3f}s, confidence={segment.confidence:.3f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
