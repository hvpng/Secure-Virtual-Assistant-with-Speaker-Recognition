from __future__ import annotations

from collections import Counter

from module_a.src.data import (
    AudioRecord,
    SpeakerBalancedBatchSampler,
    discover_records,
    split_train_validation,
)


def test_deterministic_90_10_speaker_disjoint_split():
    records = [
        AudioRecord(f"spk{speaker:02d}/{index}.wav", f"spk{speaker:02d}", "source")
        for speaker in range(20)
        for index in range(2)
    ]
    first = split_train_validation(records, seed=42, validation_ratio=0.1)
    second = split_train_validation(list(reversed(records)), seed=42, validation_ratio=0.1)
    assert first == second
    train, validation = first
    train_speakers = {record.speaker_id for record in train}
    validation_speakers = {record.speaker_id for record in validation}
    assert len(train_speakers) == 18
    assert len(validation_speakers) == 2
    assert train_speakers.isdisjoint(validation_speakers)


def test_discovery_uses_relative_paths_and_configured_speaker_component(tmp_path, write_wav):
    root = tmp_path / "VoxVietnam-T"
    write_wav(root / "speaker_a" / "one.wav")
    (root / "speaker_a" / "ignore.txt").write_text("x", encoding="utf-8")
    records = discover_records(
        root,
        split="source",
        audio_extensions=[".wav"],
        speaker_component_from_end=2,
    )
    assert records == [AudioRecord("speaker_a/one.wav", "speaker_a", "source")]


def test_balanced_sampler_is_deterministic_and_balances_each_batch():
    records = [
        AudioRecord(f"a/{index}.wav", "a", "train") for index in range(10)
    ] + [AudioRecord("b/0.wav", "b", "train"), AudioRecord("c/0.wav", "c", "train")]
    sampler = SpeakerBalancedBatchSampler(
        records,
        speakers_per_batch=2,
        utterances_per_speaker=2,
        seed=42,
        batches_per_epoch=4,
    )
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first == second
    for batch in first:
        speakers = [records[index].speaker_id for index in batch]
        assert sorted(Counter(speakers).values()) == [2, 2]
    sampler.set_epoch(1)
    assert list(iter(sampler)) != first

