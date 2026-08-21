"""Deterministic recursive audio discovery with explicit speaker-ID strategies."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from module_a.src.config import DatasetConfig


class DatasetDiscoveryError(RuntimeError):
    """Raised when files or speaker identities cannot be discovered safely."""


@dataclass(frozen=True)
class DiscoveredRecord:
    path: str
    speaker_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _relative_key(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _validate_speaker_id(value: str, path: Path) -> str:
    speaker_id = value.strip()
    if not speaker_id or speaker_id in {".", ".."}:
        raise DatasetDiscoveryError(f"Cannot determine a non-empty speaker ID for: {path}")
    if "/" in speaker_id or "\\" in speaker_id:
        raise DatasetDiscoveryError(f"Speaker ID must be one path component for: {path}")
    return speaker_id


def _load_metadata_mapping(root: Path, config: DatasetConfig) -> dict[str, str]:
    metadata_path = config.speaker_metadata_csv
    if metadata_path is None:
        raise DatasetDiscoveryError(
            "speaker_id_source=metadata_csv requires dataset.speaker_metadata_csv."
        )
    if not metadata_path.is_file():
        raise DatasetDiscoveryError(f"Speaker metadata CSV does not exist: {metadata_path}")

    mapping: dict[str, str] = {}
    try:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                config.speaker_metadata_path_column,
                config.speaker_metadata_id_column,
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise DatasetDiscoveryError(
                    f"Metadata CSV must contain columns: {sorted(required)}"
                )
            for row in reader:
                raw_path = (row.get(config.speaker_metadata_path_column) or "").strip()
                raw_speaker = (row.get(config.speaker_metadata_id_column) or "").strip()
                if not raw_path or not raw_speaker:
                    raise DatasetDiscoveryError("Metadata CSV contains an empty path or speaker ID.")
                candidate = Path(raw_path).expanduser()
                if candidate.is_absolute():
                    try:
                        key = candidate.resolve().relative_to(root).as_posix()
                    except ValueError as exc:
                        raise DatasetDiscoveryError(
                            f"Metadata path is outside dataset root: {candidate}"
                        ) from exc
                else:
                    key = Path(raw_path.replace("\\", "/")).as_posix()
                    while key.startswith("./"):
                        key = key[2:]
                speaker_id = _validate_speaker_id(raw_speaker, candidate)
                if key in mapping and mapping[key] != speaker_id:
                    raise DatasetDiscoveryError(f"Conflicting speaker IDs for metadata path: {key}")
                mapping[key] = speaker_id
    except OSError as exc:
        raise DatasetDiscoveryError(f"Cannot read speaker metadata CSV: {metadata_path}") from exc
    return mapping


def extract_speaker_id(
    path: Path,
    root: Path,
    config: DatasetConfig,
    *,
    metadata_mapping: dict[str, str] | None = None,
) -> str:
    """Extract a speaker ID using the configured, explicit layout strategy."""

    relative = path.resolve().relative_to(root.resolve())
    source = config.speaker_id_source
    if source == "parent_dir":
        if len(relative.parts) < 2:
            raise DatasetDiscoveryError(
                "parent_dir cannot infer a speaker for audio directly under dataset root; "
                "inspect the layout and choose path_component or metadata_csv."
            )
        return _validate_speaker_id(relative.parent.name, path)

    if source == "path_component":
        index = config.speaker_id_path_component
        if index is None:
            raise DatasetDiscoveryError(
                "speaker_id_source=path_component requires speaker_id_path_component."
            )
        directory_parts = relative.parts[:-1]
        try:
            value = directory_parts[index]
        except IndexError as exc:
            raise DatasetDiscoveryError(
                f"Speaker path component {index} does not exist for: {relative.as_posix()}"
            ) from exc
        return _validate_speaker_id(value, path)

    if source == "metadata_csv":
        mapping = metadata_mapping or {}
        key = relative.as_posix()
        if key not in mapping:
            raise DatasetDiscoveryError(f"Audio file has no speaker mapping in metadata CSV: {key}")
        return _validate_speaker_id(mapping[key], path)

    raise DatasetDiscoveryError(f"Unsupported speaker ID strategy: {source}")


def discover_audio_files(
    root: str | Path,
    config: DatasetConfig,
    *,
    max_files: int | None = None,
) -> list[DiscoveredRecord]:
    """Recursively discover supported audio in a filesystem-order-independent way."""

    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise DatasetDiscoveryError(
            f"Dataset root does not exist or is not a directory: {dataset_root}"
        )
    if max_files is not None and max_files <= 0:
        raise DatasetDiscoveryError("max_files must be positive when provided.")

    extensions = {extension.lower() for extension in config.audio_extensions}
    paths = sorted(
        (
            path.resolve()
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: _relative_key(path, dataset_root),
    )
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise DatasetDiscoveryError(
            f"No supported audio files found under {dataset_root}; expected {sorted(extensions)}."
        )

    metadata_mapping = (
        _load_metadata_mapping(dataset_root, config)
        if config.speaker_id_source == "metadata_csv"
        else None
    )
    return [
        DiscoveredRecord(
            path=str(path),
            speaker_id=extract_speaker_id(
                path,
                dataset_root,
                config,
                metadata_mapping=metadata_mapping,
            ),
        )
        for path in paths
    ]


def duplicate_paths(records: Iterable[DiscoveredRecord]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        normalized = str(Path(record.path).resolve())
        if normalized in seen:
            duplicates.add(normalized)
        seen.add(normalized)
    return sorted(duplicates)
