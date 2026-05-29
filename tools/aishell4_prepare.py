from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare AISHELL-4 RTTM annotations as an openvad manifest."
    )
    parser.add_argument("--root", type=Path, default=Path("data/aishell4"))
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train_S", "train_M", "train_L"],
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--download-hf", action="store_true")
    parser.add_argument(
        "--hf-stems",
        default="",
        help="Comma-separated recording stems to download from Hugging Face.",
    )
    parser.add_argument(
        "--list-hf-files",
        action="store_true",
        help="List matching Hugging Face files for the selected split and exit.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit files for smoke tests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stems = parse_stems(args.hf_stems)
    if args.list_hf_files:
        list_huggingface_files(args.split)
        return 0
    if args.download_hf:
        download_from_huggingface(args.root, args.split, stems)

    split_root = args.root / args.split
    manifest_path = args.output or args.root / f"{args.split}_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rttm_files = sorted((split_root / "TextGrid").glob("*.rttm"))
    if not rttm_files:
        rttm_files = sorted(args.root.glob(f"**/{args.split}/TextGrid/*.rttm"))
    if not rttm_files:
        raise FileNotFoundError(f"no RTTM files found under {split_root}")

    audio_index = build_audio_index(split_root)
    rows = []
    skipped = []
    for rttm_path in rttm_files:
        audio_path = match_audio(rttm_path.stem, audio_index)
        if audio_path is None:
            skipped.append(rttm_path.name)
            continue
        segments = parse_rttm_as_speech(rttm_path)
        rows.append(
            {
                "audio": str(audio_path.relative_to(manifest_path.parent)),
                "segments": segments,
            }
        )
        if args.max_files and len(rows) >= args.max_files:
            break

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote: {manifest_path}")
    print(f"items: {len(rows)}")
    if skipped:
        print(f"skipped_without_audio: {len(skipped)}")
        for name in skipped[:10]:
            print(f"  {name}")
    return 0


def parse_stems(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def list_huggingface_files(split: str) -> None:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise ImportError("Install benchmark dependencies first: uv sync --extra bench") from exc

    files = list_repo_files("AISHELL/AISHELL-4", repo_type="dataset")
    for path in files:
        if path.startswith(f"{split}/wav/") or path.startswith(f"{split}/TextGrid/"):
            print(path)


def download_from_huggingface(root: Path, split: str, stems: list[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install benchmark dependencies first: uv sync --extra bench") from exc

    root.mkdir(parents=True, exist_ok=True)
    if stems:
        allow_patterns = []
        for stem in stems:
            allow_patterns.extend(
                [
                    f"{split}/wav/{stem}.*",
                    f"{split}/wav/*{stem}.*",
                    f"{split}/TextGrid/{stem}.rttm",
                    f"{split}/TextGrid/*{stem}.rttm",
                    f"{split}/TextGrid/{stem}.TextGrid",
                    f"{split}/TextGrid/*{stem}.TextGrid",
                ]
            )
    else:
        allow_patterns = [
            f"{split}/wav/*",
            f"{split}/TextGrid/*.rttm",
            f"{split}/TextGrid/*.TextGrid",
        ]

    snapshot_download(
        repo_id="AISHELL/AISHELL-4",
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=allow_patterns,
    )


def build_audio_index(split_root: Path) -> dict[str, Path]:
    audio_files = []
    for suffix in ("*.wav", "*.flac", "*.ogg"):
        audio_files.extend(split_root.glob(f"wav/{suffix}"))
        audio_files.extend(split_root.glob(f"**/wav/{suffix}"))
    return {path.stem: path for path in sorted(audio_files)}


def match_audio(rttm_stem: str, audio_index: dict[str, Path]) -> Path | None:
    if rttm_stem in audio_index:
        return audio_index[rttm_stem]
    suffix = "_" + rttm_stem
    for stem, path in audio_index.items():
        if stem.endswith(suffix):
            return path
    return None


def parse_rttm_as_speech(path: Path) -> list[list[float]]:
    intervals: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            if duration > 0:
                intervals.append((start, start + duration))
    return [[round(start, 3), round(end, 3)] for start, end in merge_intervals(intervals)]


def merge_intervals(
    intervals: list[tuple[float, float]],
    gap: float = 0.0,
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
