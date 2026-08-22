from __future__ import annotations

import pytest

from module_a.src.training_data import (
    TrainingDataError,
    TrainingRecord,
    build_speaker_to_index,
    select_train_records,
    split_train_monitor,
)


def record(path: str, speaker: str, split: str = "train") -> TrainingRecord:
    return TrainingRecord(path, speaker, split)


def test_speaker_mapping_uses_sorted_train_speakers_only():
    records = [
        record("b.wav", "bob"),
        record("a.wav", "alice"),
        record("v.wav", "validation_only", "val"),
        record("t.wav", "test_only", "test"),
    ]

    assert build_speaker_to_index(records) == {"alice": 0, "bob": 1}


def test_empty_train_speaker_set_is_rejected():
    with pytest.raises(TrainingDataError, match="train speakers"):
        build_speaker_to_index(
            [record("v.wav", "validation_only", "val")]
        )


def test_seeded_speaker_limit_is_deterministic():
    records = [record(f"{speaker}/a.wav", speaker) for speaker in "abcdef"]
    first, first_speakers = select_train_records(records, max_speakers=3, seed=42)
    second, second_speakers = select_train_records(
        list(reversed(records)), max_speakers=3, seed=42
    )

    assert first_speakers == second_speakers
    assert first == second
    assert len(first_speakers) == 3


def test_train_monitor_split_is_deterministic_disjoint_and_keeps_fit_speakers():
    records = [
        record(f"{speaker}/{index}.wav", speaker)
        for speaker in ("alice", "bob", "carol")
        for index in range(4)
    ]

    first = split_train_monitor(
        records, holdout_ratio=0.25, seed=42, max_monitor_speakers=2
    )
    second = split_train_monitor(
        list(reversed(records)),
        holdout_ratio=0.25,
        seed=42,
        max_monitor_speakers=2,
    )

    assert first == second
    assert not ({item.path for item in first.fit} & {item.path for item in first.monitor})
    assert {item.speaker_id for item in first.fit} == {"alice", "bob", "carol"}
    assert {item.speaker_id for item in first.monitor} == set(first.monitor_speakers)
    assert all(item.split == "train" for item in (*first.fit, *first.monitor))


def test_single_utterance_speaker_stays_fit_but_is_not_monitored():
    records = [
        record("alice/only.wav", "alice"),
        record("bob/one.wav", "bob"),
        record("bob/two.wav", "bob"),
    ]

    split = split_train_monitor(records, holdout_ratio=0.5, seed=42)

    assert "alice" not in split.monitor_speakers
    assert any(item.speaker_id == "alice" for item in split.fit)
