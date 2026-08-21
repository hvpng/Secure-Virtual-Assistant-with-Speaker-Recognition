from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from module_a.src.config import load_config
from module_a.src.dataset_discovery import DatasetDiscoveryError, discover_audio_files


def test_recursive_discovery_ignores_non_audio_and_is_sorted(tmp_path, write_wav):
    config = load_config().dataset
    write_wav(tmp_path / "speaker_b" / "two.WAV")
    write_wav(tmp_path / "speaker_a" / "one.wav")
    (tmp_path / "speaker_a" / "notes.txt").write_text("ignore", encoding="utf-8")

    records = discover_audio_files(tmp_path, config)

    assert [Path(record.path).name for record in records] == ["one.wav", "two.WAV"]
    assert [record.speaker_id for record in records] == ["speaker_a", "speaker_b"]


def test_path_component_strategy_extracts_explicit_component(tmp_path, write_wav):
    base = load_config().dataset
    config = replace(
        base,
        speaker_id_source="path_component",
        speaker_id_path_component=0,
    )
    write_wav(tmp_path / "spk001" / "session_a" / "one.wav")

    assert discover_audio_files(tmp_path, config)[0].speaker_id == "spk001"


def test_parent_strategy_fails_for_audio_directly_under_root(tmp_path, write_wav):
    write_wav(tmp_path / "orphan.wav")

    with pytest.raises(DatasetDiscoveryError, match="cannot infer a speaker"):
        discover_audio_files(tmp_path, load_config().dataset)


def test_metadata_csv_strategy_requires_complete_mapping(tmp_path, write_wav):
    first = write_wav(tmp_path / "clips" / "one.wav")
    second = write_wav(tmp_path / "clips" / "two.wav")
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "speaker_id"])
        writer.writeheader()
        writer.writerow({"path": first.relative_to(tmp_path).as_posix(), "speaker_id": "alice"})
        writer.writerow({"path": second.relative_to(tmp_path).as_posix(), "speaker_id": "bob"})
    config = replace(
        load_config().dataset,
        speaker_id_source="metadata_csv",
        speaker_metadata_csv=metadata,
    )

    assert [record.speaker_id for record in discover_audio_files(tmp_path, config)] == [
        "alice",
        "bob",
    ]


@pytest.mark.parametrize(
    "module_name",
    [
        "module_a.scripts.inspect_dataset",
        "module_a.scripts.prepare_manifests",
    ],
)
def test_cli_help_runs_offline(module_name):
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "--dataset-root" in result.stdout
