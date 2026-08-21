from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from module_a.src.audio_metadata import AudioMetadataRecord
from module_a.src.config import load_config
from module_a.src.manifests import (
    MANIFEST_COLUMNS,
    ManifestError,
    ManifestRecord,
    allocate_split_counts,
    assert_speaker_disjoint,
    build_manifest_records,
    filter_eligible_speakers,
    split_speakers,
    validate_manifests,
    write_manifests,
)
from module_a.src.pipeline import prepare_manifests


def metadata(path: Path, speaker: str, *, readable: bool = True) -> AudioMetadataRecord:
    return AudioMetadataRecord(
        path=str(path),
        speaker_id=speaker,
        sample_rate=16_000 if readable else None,
        duration_sec=0.1 if readable else None,
        channels=1 if readable else None,
        format="wav",
        readable=readable,
        error=None if readable else "corrupt",
    )


def manifest(path: str, speaker: str, split: str) -> ManifestRecord:
    return ManifestRecord(path, speaker, split, 0.1, 16_000, 1)


def test_exact_80_10_10_allocation_for_100_speakers():
    assert allocate_split_counts(100, load_config().split) == {
        "train": 80,
        "val": 10,
        "test": 10,
    }


def test_small_dataset_policy_is_non_empty_and_too_small_fails():
    config = load_config().split
    assert allocate_split_counts(3, config) == {"train": 1, "val": 1, "test": 1}
    assert sum(allocate_split_counts(4, config).values()) == 4
    with pytest.raises(ManifestError, match="At least 3"):
        allocate_split_counts(2, config)


def test_split_is_deterministic_assigns_every_speaker_once_and_has_no_overlap():
    speakers = [f"speaker_{index:03d}" for index in range(37)]
    first = split_speakers(reversed(speakers), load_config().split)
    second = split_speakers(speakers, load_config().split)

    assert first == second
    assigned = first.train + first.val + first.test
    assert set(assigned) == set(speakers)
    assert len(assigned) == len(set(assigned))
    assert_speaker_disjoint(first)


def test_different_seed_changes_speaker_assignment():
    speakers = [f"speaker_{index:03d}" for index in range(30)]
    config = load_config().split

    assert split_speakers(speakers, config) != split_speakers(
        speakers, replace(config, seed=99)
    )


def test_low_utterance_and_corrupt_records_are_excluded(tmp_path):
    records = [
        metadata(tmp_path / "alice_1.wav", "alice"),
        metadata(tmp_path / "alice_2.wav", "alice"),
        metadata(tmp_path / "bob_1.wav", "bob"),
        metadata(tmp_path / "bob_bad.wav", "bob", readable=False),
    ]

    eligible, excluded = filter_eligible_speakers(records, 2)

    assert {record.speaker_id for record in eligible} == {"alice"}
    assert excluded == {"bob": 1}


def test_manifest_columns_relative_paths_and_split_values(tmp_path, write_wav):
    records = []
    speakers = ["alice", "bob", "carol"]
    for speaker in speakers:
        path = write_wav(tmp_path / speaker / "one.wav")
        records.append(metadata(path, speaker))
    allocation = split_speakers(speakers, load_config().split)
    rows = build_manifest_records(records, allocation, tmp_path)
    output = tmp_path / "outputs"
    write_manifests(rows, output)

    for split in ("train", "val", "test"):
        with (output / f"{split}_manifest.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            written = list(reader)
        assert tuple(reader.fieldnames or ()) == MANIFEST_COLUMNS
        assert written
        assert {row["split"] for row in written} == {split}
        assert all(not Path(row["path"]).is_absolute() for row in written)


def test_validator_rejects_speaker_leakage():
    rows = [
        manifest("train.wav", "alice", "train"),
        manifest("val.wav", "alice", "val"),
        manifest("test.wav", "bob", "test"),
    ]

    with pytest.raises(ManifestError, match="Speaker leakage"):
        validate_manifests(rows)


def test_validator_rejects_duplicate_file_across_splits():
    rows = [
        manifest("same.wav", "alice", "train"),
        manifest("same.wav", "bob", "val"),
        manifest("test.wav", "carol", "test"),
    ]

    with pytest.raises(ManifestError, match="Duplicate file path"):
        validate_manifests(rows)


def test_validator_rejects_missing_file_and_corrupt_marker(tmp_path):
    existing = tmp_path / "existing.wav"
    existing.write_bytes(b"placeholder")
    rows = [
        manifest("existing.wav", "alice", "train"),
        manifest("missing.wav", "bob", "val"),
        manifest("test.wav", "carol", "test"),
    ]
    with pytest.raises(ManifestError, match="does not exist"):
        validate_manifests(rows, dataset_root=tmp_path)

    corrupt_rows = [
        manifest("train.wav", "alice", "train"),
        manifest("val.wav", "bob", "val"),
        manifest("test.wav", "carol", "test"),
    ]
    with pytest.raises(ManifestError, match="Corrupt audio"):
        validate_manifests(corrupt_rows, corrupt_paths={"val.wav"})


def test_validator_requires_exact_expected_coverage():
    rows = [
        manifest("train.wav", "alice", "train"),
        manifest("val.wav", "bob", "val"),
        manifest("test.wav", "carol", "test"),
    ]
    with pytest.raises(ManifestError, match="coverage mismatch"):
        validate_manifests(
            rows,
            expected_paths={"train.wav", "val.wav", "test.wav", "extra.wav"},
        )


def test_pipeline_summary_counts_and_manifests_are_reproducible(tmp_path, write_wav):
    dataset_root = tmp_path / "dataset"
    for speaker_index in range(10):
        for utterance_index in range(2):
            write_wav(
                dataset_root
                / f"speaker_{speaker_index:02d}"
                / f"utterance_{utterance_index}.wav"
            )
    (dataset_root / "speaker_00" / "bad.wav").write_bytes(b"bad")
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = prepare_manifests(
        load_config(dataset_root=dataset_root, output_root=first_output)
    )
    second = prepare_manifests(
        load_config(dataset_root=dataset_root, output_root=second_output)
    )

    assert first.inspection.summary["total_discovered_files"] == 21
    assert first.inspection.summary["usable_files"] == 20
    assert first.inspection.summary["corrupt_files"] == 1
    assert first.split_summary["train"] == {"speakers": 8, "utterances": 16}
    assert first.split_summary["val"] == {"speakers": 1, "utterances": 2}
    assert first.split_summary["test"] == {"speakers": 1, "utterances": 2}
    assert first.manifests == second.manifests
    for split in ("train", "val", "test"):
        assert (first_output / f"{split}_manifest.csv").read_bytes() == (
            second_output / f"{split}_manifest.csv"
        ).read_bytes()
