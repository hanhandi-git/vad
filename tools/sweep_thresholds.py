from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from _common import evaluate_items, load_manifest, parse_float_list, parse_int_list

from openvad import VadConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep openvad parameters on a labeled manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--aggressiveness", default="0,1,2,3")
    parser.add_argument("--onset", default="0.50,0.54,0.58,0.62,0.66")
    parser.add_argument("--offset", default="0.34,0.38,0.42,0.46,0.50")
    parser.add_argument("--min-speech-ms", type=int, default=80)
    parser.add_argument("--min-silence-ms", type=int, default=120)
    parser.add_argument("--speech-pad-ms", type=int, default=40)
    parser.add_argument("--top", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_manifest(args.manifest)
    rows: list[tuple[float, float, float, int, float, float, float]] = []
    for aggressiveness, onset, offset in itertools.product(
        parse_int_list(args.aggressiveness),
        parse_float_list(args.onset),
        parse_float_list(args.offset),
    ):
        if offset > onset:
            continue
        config = VadConfig(
            onset_threshold=onset,
            offset_threshold=offset,
            aggressiveness=aggressiveness,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
        )
        metrics = evaluate_items(items, config)
        rows.append(
            (
                metrics.f1,
                metrics.precision,
                metrics.recall,
                aggressiveness,
                onset,
                offset,
                metrics.realtime_factor,
            )
        )
    rows.sort(reverse=True)
    print("f1\tprecision\trecall\taggr\tonset\toffset\trtf")
    for f1, precision, recall, aggressiveness, onset, offset, rtf in rows[: args.top]:
        print(
            f"{f1:.4f}\t{precision:.4f}\t{recall:.4f}\t"
            f"{aggressiveness}\t{onset:.2f}\t{offset:.2f}\t{rtf:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
