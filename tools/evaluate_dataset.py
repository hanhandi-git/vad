from __future__ import annotations

import argparse
from pathlib import Path

from _common import evaluate_items, load_manifest, print_metrics

from openvad import VadConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate openvad on a JSONL manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--onset-threshold", type=float, default=0.58)
    parser.add_argument("--offset-threshold", type=float, default=0.42)
    parser.add_argument("--min-speech-ms", type=int, default=80)
    parser.add_argument("--min-silence-ms", type=int, default=120)
    parser.add_argument("--speech-pad-ms", type=int, default=40)
    parser.add_argument("--aggressiveness", type=int, choices=range(4), default=1)
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
    items = load_manifest(args.manifest)
    metrics = evaluate_items(items, config)
    print_metrics(metrics, config.hop_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
