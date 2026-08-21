from __future__ import annotations

import pytest

from module_a.src.audio_metadata import (
    build_dataset_summary,
    inspect_audio,
    inspect_audio_records,
)
from module_a.src.dataset_discovery import DiscoveredRecord


def test_wav_metadata_reads_duration_sample_rate_and_channels(tmp_path, write_wav):
    path = write_wav(
        tmp_path / "speaker" / "stereo.wav",
        sample_rate=8_000,
        duration_sec=0.25,
        channels=2,
    )

    result = inspect_audio(DiscoveredRecord(str(path), "speaker"))

    assert result.readable is True
    assert result.sample_rate == 8_000
    assert result.channels == 2
    assert result.duration_sec == pytest.approx(0.25)
    assert result.format == "wav"


def test_corrupt_audio_is_controlled_and_does_not_abort_batch(tmp_path, write_wav):
    good = write_wav(tmp_path / "speaker" / "good.wav")
    corrupt = tmp_path / "speaker" / "corrupt.wav"
    corrupt.write_bytes(b"not-wave")

    results = inspect_audio_records(
        [
            DiscoveredRecord(str(good), "speaker"),
            DiscoveredRecord(str(corrupt), "speaker"),
        ]
    )

    assert [record.readable for record in results] == [True, False]
    assert results[1].sample_rate is None
    assert results[1].error


def test_dataset_summary_counts_usable_corrupt_and_distributions(tmp_path, write_wav):
    alice = write_wav(tmp_path / "alice" / "a.wav", sample_rate=16_000)
    bob = write_wav(tmp_path / "bob" / "b.wav", sample_rate=8_000, channels=2)
    corrupt = tmp_path / "bob" / "bad.wav"
    corrupt.write_bytes(b"bad")
    metadata = inspect_audio_records(
        [
            DiscoveredRecord(str(alice), "alice"),
            DiscoveredRecord(str(bob), "bob"),
            DiscoveredRecord(str(corrupt), "bob"),
        ]
    )

    summary = build_dataset_summary(
        metadata,
        dataset_name="fixture",
        dataset_root=tmp_path,
        seed=42,
    )

    assert summary["total_discovered_files"] == 3
    assert summary["usable_files"] == 2
    assert summary["corrupt_files"] == 1
    assert summary["num_speakers"] == 2
    assert summary["sample_rate_distribution"] == {"16000": 1, "8000": 1}
    assert summary["channel_distribution"] == {"1": 1, "2": 1}
    assert summary["duration_sec"]["mean"] == pytest.approx(0.1)
