from __future__ import annotations

import pytest

from module_a.src.training_data import (
    SpeakerBalancedBatchSampler,
    TrainingDataError,
    TrainingRecord,
)


def test_balanced_sampler_is_reproducible_per_epoch_and_changes_across_epochs():
    records = [
        TrainingRecord(f"{speaker}/{index}.wav", speaker)
        for speaker in ("a", "b", "c", "d")
        for index in range(3)
    ]
    sampler = SpeakerBalancedBatchSampler(
        records,
        speakers_per_batch=2,
        utterances_per_speaker=2,
        seed=42,
        batches_per_epoch=8,
    )
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_each_batch_has_exact_speaker_and_utterance_counts_with_replacement():
    records = [
        TrainingRecord("low/only.wav", "low"),
        *[
            TrainingRecord(f"high/{index}.wav", "high")
            for index in range(20)
        ],
        TrainingRecord("medium/one.wav", "medium"),
        TrainingRecord("medium/two.wav", "medium"),
    ]
    sampler = SpeakerBalancedBatchSampler(
        records,
        speakers_per_batch=3,
        utterances_per_speaker=2,
        seed=7,
        batches_per_epoch=5,
    )

    for batch in sampler:
        speakers = [records[index].speaker_id for index in batch]
        assert len(batch) == 6
        assert {speaker: speakers.count(speaker) for speaker in set(speakers)} == {
            "low": 2,
            "medium": 2,
            "high": 2,
        }
        low_indices = [index for index in batch if records[index].speaker_id == "low"]
        assert low_indices[0] == low_indices[1]


@pytest.mark.parametrize("speakers,utterances", [(0, 2), (2, 0)])
def test_invalid_balanced_batch_settings_are_rejected(speakers, utterances):
    records = [TrainingRecord("a.wav", "a"), TrainingRecord("b.wav", "b")]
    with pytest.raises(TrainingDataError, match="positive"):
        SpeakerBalancedBatchSampler(
            records,
            speakers_per_batch=speakers,
            utterances_per_speaker=utterances,
            seed=42,
        )
