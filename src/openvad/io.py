from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_audio(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    sample_format: str = "s16le",
    channels: int = 1,
) -> tuple[np.ndarray, int]:
    audio_path = Path(path)
    if audio_path.suffix.lower() == ".wav":
        return read_wav(audio_path)
    if audio_path.suffix.lower() == ".pcm":
        if sample_rate is None:
            raise ValueError("sample_rate is required for raw PCM files")
        return read_pcm(audio_path, sample_rate, sample_format=sample_format, channels=channels)
    if audio_path.suffix.lower() in {".flac", ".ogg"}:
        return read_sound_file(audio_path)
    raise ValueError(f"unsupported audio file extension: {audio_path.suffix}")


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        audio = _decode_pcm24(raw)
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width} bytes")

    if channels <= 0:
        raise ValueError("invalid WAV channel count")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def read_pcm(
    path: str | Path,
    sample_rate: int,
    *,
    sample_format: str = "s16le",
    channels: int = 1,
) -> tuple[np.ndarray, int]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")

    raw = Path(path).read_bytes()
    if sample_format == "s16le":
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_format == "s16be":
        audio = np.frombuffer(raw, dtype=">i2").astype(np.float32) / 32768.0
    elif sample_format == "f32le":
        audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif sample_format == "u8":
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported raw PCM sample format: {sample_format}")

    if channels > 1:
        usable = (len(audio) // channels) * channels
        audio = audio[:usable].reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def read_sound_file(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Reading FLAC/OGG requires the benchmark extra: uv sync --extra bench"
        ) from exc

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        mono = audio.mean(axis=1, dtype=np.float32)
    else:
        mono = audio[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate)


def _decode_pcm24(raw: bytes) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.uint8)
    if len(data) % 3 != 0:
        raise ValueError("invalid 24-bit PCM data length")
    triples = data.reshape(-1, 3).astype(np.int32)
    values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    sign_bit = 1 << 23
    values = (values ^ sign_bit) - sign_bit
    return values.astype(np.float32) / 8388608.0
