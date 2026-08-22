from __future__ import annotations

import numpy as np

from module_a.src.sv_evaluation import (
    SVScore,
    build_sv_trials,
    calibrate_sv_threshold,
    compute_sv_metrics,
    rates_at_threshold,
    score_sv_trials,
)
from module_a.src.training_data import TrainingRecord


def _records(split="val"):
    return [
        TrainingRecord(f"{split}/{speaker}/{index}.wav", speaker, split)
        for speaker in (f"{split}_a", f"{split}_b", f"{split}_c")
        for index in range(3)
    ]


def test_sv_trials_are_balanced_unique_valid_and_deterministic():
    first = build_sv_trials(_records(), seed=42, max_positive_per_speaker=2)
    second = build_sv_trials(list(reversed(_records())), seed=42, max_positive_per_speaker=2)
    assert first == second
    positives = [trial for trial in first if trial.label == 1]
    negatives = [trial for trial in first if trial.label == 0]
    assert len(positives) == len(negatives) == 6
    assert all(trial.path_a != trial.path_b for trial in first)
    assert all(trial.speaker_a == trial.speaker_b for trial in positives)
    assert all(trial.speaker_a != trial.speaker_b for trial in negatives)
    assert len({tuple(sorted((trial.path_a, trial.path_b))) for trial in first}) == len(first)


def test_sv_cosine_scoring_uses_normalized_dot_product():
    records = _records()
    trials = build_sv_trials(records, seed=42, max_positive_per_speaker=1)
    bases = {"val_a": 0, "val_b": 1, "val_c": 2}
    embeddings = {}
    for record in records:
        vector = np.zeros(4, dtype=np.float32)
        vector[bases[record.speaker_id]] = 1.0
        embeddings[record.path] = vector
    scores = score_sv_trials(trials, embeddings)
    assert all(score.score == (1.0 if score.label else 0.0) for score in scores)


def test_sv_far_frr_and_empirical_eer_have_known_values():
    scores = [
        SVScore("p1", "p2", 1, 0.9),
        SVScore("p3", "p4", 1, 0.8),
        SVScore("n1", "n2", 0, 0.2),
        SVScore("n3", "n4", 0, 0.1),
    ]
    assert rates_at_threshold(scores, 0.8) == {"far": 0.0, "frr": 0.0, "tpr": 1.0}
    metrics = compute_sv_metrics(scores)
    calibration = calibrate_sv_threshold(scores)
    assert metrics["eer"] == 0.0
    assert metrics["eer_threshold"] == 0.8
    assert calibration["source_split"] == "validation"
    assert calibration["selected_threshold"] == 0.8


def test_sv_threshold_accepts_scores_equal_to_threshold():
    scores = [SVScore("p1", "p2", 1, 0.5), SVScore("n1", "n2", 0, 0.5)]
    rates = rates_at_threshold(scores, 0.5)
    assert rates["frr"] == 0.0
    assert rates["far"] == 1.0
