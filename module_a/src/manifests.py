"""Speaker-disjoint allocation, manifest writing, and leakage validation."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from module_a.src.audio_metadata import AudioMetadataRecord
from module_a.src.config import SplitConfig


SPLIT_NAMES = ("train", "val", "test")
MANIFEST_COLUMNS = (
    "path",
    "speaker_id",
    "split",
    "duration_sec",
    "sample_rate",
    "channels",
)


class ManifestError(RuntimeError):
    """Raised when a split or manifest violates the A1 data contract."""


@dataclass(frozen=True)
class SpeakerAllocation:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    speaker_id: str
    split: str
    duration_sec: float
    sample_rate: int
    channels: int
    readable: bool = True

    def csv_dict(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("readable")
        return values


def allocate_split_counts(total_speakers: int, config: SplitConfig) -> dict[str, int]:
    """Use largest remainders, then deterministically repair empty splits.

    Floors are taken from the exact 80/10/10 quotas. Remaining speakers go to
    the largest fractional remainders with train/val/test as the stable tie
    order. For small but valid datasets, an empty split receives one speaker
    from the currently largest split. Fewer than three speakers fail because a
    non-empty, speaker-disjoint three-way split is impossible.
    """

    if total_speakers < 3:
        raise ManifestError(
            "At least 3 eligible speakers are required for non-empty train/val/test splits."
        )
    ratios = {
        "train": config.train_ratio,
        "val": config.val_ratio,
        "test": config.test_ratio,
    }
    if any(value <= 0 for value in ratios.values()) or not math.isclose(
        sum(ratios.values()), 1.0, abs_tol=1e-9
    ):
        raise ManifestError("Split ratios must be positive and sum to 1.0.")

    quotas = {name: total_speakers * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(quota) for name, quota in quotas.items()}
    remaining = total_speakers - sum(counts.values())
    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(quotas[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in remainder_order[:remaining]:
        counts[name] += 1

    for empty_name in (name for name in SPLIT_NAMES if counts[name] == 0):
        donor = max(
            (name for name in SPLIT_NAMES if counts[name] > 1),
            key=lambda name: (counts[name], -SPLIT_NAMES.index(name)),
            default=None,
        )
        if donor is None:
            raise ManifestError("Cannot repair an empty split without losing another split.")
        counts[donor] -= 1
        counts[empty_name] += 1

    if sum(counts.values()) != total_speakers or any(
        counts[name] <= 0 for name in SPLIT_NAMES
    ):
        raise ManifestError("Internal split allocation error.")
    return counts


def split_speakers(
    speaker_ids: Iterable[str], config: SplitConfig
) -> SpeakerAllocation:
    speakers = sorted(set(speaker_ids))
    if any(not speaker.strip() for speaker in speakers):
        raise ManifestError("Speaker IDs must be non-empty.")
    counts = allocate_split_counts(len(speakers), config)
    shuffled = speakers.copy()
    random.Random(config.seed).shuffle(shuffled)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    allocation = SpeakerAllocation(
        train=tuple(sorted(shuffled[:train_end])),
        val=tuple(sorted(shuffled[train_end:val_end])),
        test=tuple(sorted(shuffled[val_end:])),
    )
    assert_speaker_disjoint(allocation)
    return allocation


def assert_speaker_disjoint(allocation: SpeakerAllocation) -> None:
    groups = {name: set(values) for name, values in allocation.as_dict().items()}
    if groups["train"] & groups["val"]:
        raise ManifestError("Speaker leakage detected between train and val.")
    if groups["train"] & groups["test"]:
        raise ManifestError("Speaker leakage detected between train and test.")
    if groups["val"] & groups["test"]:
        raise ManifestError("Speaker leakage detected between val and test.")
    if any(not group for group in groups.values()):
        raise ManifestError("Train, val, and test must each contain at least one speaker.")


def filter_eligible_speakers(
    records: Iterable[AudioMetadataRecord], min_utterances_per_speaker: int
) -> tuple[list[AudioMetadataRecord], dict[str, int]]:
    if min_utterances_per_speaker <= 0:
        raise ManifestError("min_utterances_per_speaker must be positive.")
    usable = [record for record in records if record.readable]
    counts = Counter(record.speaker_id for record in usable)
    excluded = {
        speaker_id: count
        for speaker_id, count in sorted(counts.items())
        if count < min_utterances_per_speaker
    }
    eligible = [record for record in usable if record.speaker_id not in excluded]
    return eligible, excluded


def build_manifest_records(
    records: Iterable[AudioMetadataRecord],
    allocation: SpeakerAllocation,
    dataset_root: str | Path,
) -> list[ManifestRecord]:
    root = Path(dataset_root).expanduser().resolve()
    split_by_speaker = {
        speaker: split
        for split, speakers in allocation.as_dict().items()
        for speaker in speakers
    }
    manifest_records: list[ManifestRecord] = []
    for record in records:
        if not record.readable:
            raise ManifestError(f"Corrupt audio cannot enter a manifest: {record.path}")
        if record.speaker_id not in split_by_speaker:
            raise ManifestError(f"Speaker was not assigned to a split: {record.speaker_id}")
        if record.duration_sec is None or record.sample_rate is None or record.channels is None:
            raise ManifestError(f"Readable record has incomplete metadata: {record.path}")
        try:
            relative_path = Path(record.path).resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ManifestError(f"Audio path is outside dataset root: {record.path}") from exc
        manifest_records.append(
            ManifestRecord(
                path=relative_path,
                speaker_id=record.speaker_id,
                split=split_by_speaker[record.speaker_id],
                duration_sec=round(record.duration_sec, 6),
                sample_rate=record.sample_rate,
                channels=record.channels,
            )
        )
    return sorted(
        manifest_records,
        key=lambda record: (
            SPLIT_NAMES.index(record.split),
            record.speaker_id,
            record.path,
        ),
    )


def validate_manifests(
    records: Iterable[ManifestRecord],
    *,
    dataset_root: str | Path | None = None,
    expected_paths: set[str] | None = None,
    corrupt_paths: set[str] | None = None,
) -> None:
    """Fail closed on missing files, duplicates, corruption, or speaker leakage."""

    rows = list(records)
    paths: set[str] = set()
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    row_counts = Counter(record.split for record in rows)
    root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
    normalized_corrupt = {Path(path).as_posix() for path in (corrupt_paths or set())}

    for record in rows:
        if record.split not in SPLIT_NAMES:
            raise ManifestError(f"Invalid split value: {record.split}")
        if not record.path.strip():
            raise ManifestError("Manifest path must not be empty.")
        if not record.speaker_id.strip():
            raise ManifestError("Manifest speaker_id must not be empty.")
        normalized_path = Path(record.path).as_posix()
        manifest_path = Path(record.path)
        if manifest_path.is_absolute() or ".." in manifest_path.parts:
            raise ManifestError(
                f"Manifest paths must stay relative to dataset root: {normalized_path}"
            )
        if normalized_path in paths:
            raise ManifestError(f"Duplicate file path across manifests: {normalized_path}")
        if not record.readable or normalized_path in normalized_corrupt:
            raise ManifestError(f"Corrupt audio appears in a manifest: {normalized_path}")
        if root is not None:
            candidate = (root / manifest_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ManifestError(
                    f"Manifest path is outside dataset root: {normalized_path}"
                ) from exc
            if not candidate.is_file():
                raise ManifestError(f"Manifest file does not exist: {normalized_path}")
        paths.add(normalized_path)
        speakers_by_split[record.split].add(record.speaker_id)

    if any(row_counts[name] == 0 for name in SPLIT_NAMES):
        raise ManifestError("Train, val, and test manifests must all be non-empty.")
    if speakers_by_split["train"] & speakers_by_split["val"]:
        raise ManifestError("Speaker leakage detected between train and val manifests.")
    if speakers_by_split["train"] & speakers_by_split["test"]:
        raise ManifestError("Speaker leakage detected between train and test manifests.")
    if speakers_by_split["val"] & speakers_by_split["test"]:
        raise ManifestError("Speaker leakage detected between val and test manifests.")
    if expected_paths is not None:
        normalized_expected = {Path(path).as_posix() for path in expected_paths}
        if paths != normalized_expected:
            missing = sorted(normalized_expected - paths)
            unexpected = sorted(paths - normalized_expected)
            raise ManifestError(
                f"Manifest coverage mismatch; missing={missing}, unexpected={unexpected}"
            )


def write_manifests(records: Iterable[ManifestRecord], output_root: str | Path) -> None:
    rows = list(records)
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        path = output / f"{split}_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for record in rows:
                if record.split == split:
                    writer.writerow(record.csv_dict())


def build_split_summary(
    records: Iterable[ManifestRecord],
    allocation: SpeakerAllocation,
    *,
    seed: int,
    excluded_speakers: Mapping[str, int] | None = None,
) -> dict[str, object]:
    assert_speaker_disjoint(allocation)
    rows = list(records)
    summary: dict[str, object] = {
        "seed": seed,
        "speaker_disjoint": True,
        "excluded_low_utterance_speakers": dict(excluded_speakers or {}),
    }
    for split, speakers in allocation.as_dict().items():
        summary[split] = {
            "speakers": len(speakers),
            "utterances": sum(record.split == split for record in rows),
        }
    return summary


def write_json(payload: Mapping[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
