"""Reusable A1 orchestration shared by CLI scripts and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from module_a.src.audio_metadata import (
    AudioMetadataRecord,
    build_dataset_summary,
    inspect_audio_records,
)
from module_a.src.config import ModuleAConfig
from module_a.src.dataset_discovery import DiscoveredRecord, discover_audio_files
from module_a.src.manifests import (
    ManifestRecord,
    SpeakerAllocation,
    build_manifest_records,
    build_split_summary,
    filter_eligible_speakers,
    split_speakers,
    validate_manifests,
    write_json,
    write_manifests,
)
from module_a.src.reproducibility import seed_everything


@dataclass(frozen=True)
class InspectionResult:
    discovered: list[DiscoveredRecord]
    metadata: list[AudioMetadataRecord]
    summary: dict[str, object]


@dataclass(frozen=True)
class PreparationResult:
    inspection: InspectionResult
    eligible: list[AudioMetadataRecord]
    excluded_speakers: dict[str, int]
    allocation: SpeakerAllocation
    manifests: list[ManifestRecord]
    split_summary: dict[str, object]


def inspect_dataset(config: ModuleAConfig, *, write_output: bool = True) -> InspectionResult:
    root = config.require_dataset_root()
    seed_everything(config.split.seed)
    discovered = discover_audio_files(
        root,
        config.dataset,
        max_files=config.inspection.max_files,
    )
    metadata = inspect_audio_records(discovered)
    summary = build_dataset_summary(
        metadata,
        dataset_name=config.dataset.name,
        dataset_root=root,
        seed=config.split.seed,
    )
    if write_output:
        write_json(summary, config.output_root / "dataset_summary.json")
    return InspectionResult(discovered=discovered, metadata=metadata, summary=summary)


def prepare_manifests(config: ModuleAConfig) -> PreparationResult:
    root = config.require_dataset_root()
    inspection = inspect_dataset(config, write_output=True)
    eligible, excluded = filter_eligible_speakers(
        inspection.metadata,
        config.dataset.min_utterances_per_speaker,
    )
    allocation = split_speakers(
        (record.speaker_id for record in eligible), config.split
    )
    manifests = build_manifest_records(eligible, allocation, root)
    expected_paths = {
        Path(record.path).resolve().relative_to(root).as_posix() for record in eligible
    }
    corrupt_paths = {
        Path(record.path).resolve().relative_to(root).as_posix()
        for record in inspection.metadata
        if not record.readable
    }
    validate_manifests(
        manifests,
        dataset_root=root,
        expected_paths=expected_paths,
        corrupt_paths=corrupt_paths,
    )
    write_manifests(manifests, config.output_root)
    split_summary = build_split_summary(
        manifests,
        allocation,
        seed=config.split.seed,
        excluded_speakers=excluded,
    )
    write_json(split_summary, config.output_root / "split_summary.json")
    return PreparationResult(
        inspection=inspection,
        eligible=eligible,
        excluded_speakers=excluded,
        allocation=allocation,
        manifests=manifests,
        split_summary=split_summary,
    )
