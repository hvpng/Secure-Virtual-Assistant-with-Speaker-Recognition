"""Fast audio-header inspection for Module A dataset preparation."""

from __future__ import annotations

import json
import statistics
import subprocess
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from module_a.src.dataset_discovery import DiscoveredRecord, duplicate_paths


@dataclass(frozen=True)
class AudioMetadataRecord:
    path: str
    speaker_id: str
    sample_rate: int | None
    duration_sec: float | None
    channels: int | None
    format: str
    readable: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _inspect_wav(path: Path) -> tuple[int, float, int]:
    with wave.open(str(path), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        frame_count = audio.getnframes()
        if sample_rate <= 0 or channels <= 0 or frame_count < 0:
            raise ValueError("invalid WAV header values")
        return sample_rate, frame_count / sample_rate, channels


def _inspect_with_ffprobe(path: Path) -> tuple[int, float, int]:
    """Read non-WAV metadata without decoding the full audio payload."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe is required to inspect non-WAV audio; install ffmpeg/ffprobe."
        ) from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("ffprobe could not read the audio header") from exc

    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration = float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("ffprobe returned incomplete audio metadata") from exc
    if sample_rate <= 0 or channels <= 0 or duration < 0:
        raise ValueError("invalid audio metadata values")
    return sample_rate, duration, channels


def inspect_audio(record: DiscoveredRecord) -> AudioMetadataRecord:
    """Inspect one header and convert all corruption/tool errors into record state."""

    path = Path(record.path)
    audio_format = path.suffix.lower().lstrip(".")
    try:
        if path.suffix.lower() == ".wav":
            sample_rate, duration, channels = _inspect_wav(path)
        else:
            sample_rate, duration, channels = _inspect_with_ffprobe(path)
        return AudioMetadataRecord(
            path=str(path.resolve()),
            speaker_id=record.speaker_id,
            sample_rate=sample_rate,
            duration_sec=duration,
            channels=channels,
            format=audio_format,
            readable=True,
        )
    except Exception as exc:  # Per-file corruption must not abort the inspection job.
        return AudioMetadataRecord(
            path=str(path.resolve()),
            speaker_id=record.speaker_id,
            sample_rate=None,
            duration_sec=None,
            channels=None,
            format=audio_format,
            readable=False,
            error=str(exc) or exc.__class__.__name__,
        )


def inspect_audio_records(records: Iterable[DiscoveredRecord]) -> list[AudioMetadataRecord]:
    return [inspect_audio(record) for record in records]


def _numeric_stats(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _distribution(values: Iterable[int | str]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def build_dataset_summary(
    records: list[AudioMetadataRecord],
    *,
    dataset_name: str,
    dataset_root: str | Path,
    seed: int,
) -> dict[str, object]:
    usable = [record for record in records if record.readable]
    corrupt = [record for record in records if not record.readable]
    per_speaker: dict[str, int] = defaultdict(int)
    for record in usable:
        per_speaker[record.speaker_id] += 1

    utterance_counts = list(per_speaker.values())
    durations = [
        record.duration_sec for record in usable if record.duration_sec is not None
    ]
    discovered = [DiscoveredRecord(record.path, record.speaker_id) for record in records]
    root = Path(dataset_root).expanduser().resolve()

    def relative(path: str) -> str:
        try:
            return Path(path).resolve().relative_to(root).as_posix()
        except ValueError:
            return Path(path).name

    return {
        "dataset_name": dataset_name,
        "dataset_root": str(root),
        "seed": seed,
        "total_discovered_files": len(records),
        "usable_files": len(usable),
        "corrupt_files": len(corrupt),
        "duplicate_paths": [relative(path) for path in duplicate_paths(discovered)],
        "num_speakers": len(per_speaker),
        "utterances_per_speaker": _numeric_stats(utterance_counts),
        "speaker_utterance_counts": dict(sorted(per_speaker.items())),
        "sample_rate_distribution": _distribution(
            record.sample_rate for record in usable if record.sample_rate is not None
        ),
        "channel_distribution": _distribution(
            record.channels for record in usable if record.channels is not None
        ),
        "duration_sec": _numeric_stats(durations),
        "extension_distribution": _distribution(record.format for record in records),
        "corrupt_records": [
            {"path": relative(record.path), "error": record.error} for record in corrupt
        ],
    }
