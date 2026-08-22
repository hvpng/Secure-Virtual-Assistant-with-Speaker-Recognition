from __future__ import annotations

import numpy as np
import pytest

from module_a.src.evaluation import EvaluationError
from module_a.src.sv_evaluation import (
    SVScore,
    build_sv_trials,
    calibrate_sv_threshold,
    compute_sv_metrics,
    load_frozen_sv_calibration,
    rates_at_threshold,
    score_sv_trials,
    select_sv_target_far_operating_point,
    write_sv_calibration,
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
    assert calibration["eer_threshold"] == 0.8
    assert calibration["deployment_threshold"] == 0.8


def test_sv_threshold_accepts_scores_equal_to_threshold():
    scores = [SVScore("p1", "p2", 1, 0.5), SVScore("n1", "n2", 0, 0.5)]
    rates = rates_at_threshold(scores, 0.5)
    assert rates["frr"] == 0.0
    assert rates["far"] == 1.0


def test_target_far_selects_lowest_frr_feasible_empirical_threshold():
    scores = [
        SVScore("p1", "p2", 1, 0.9),
        SVScore("p3", "p4", 1, 0.7),
        SVScore("p5", "p6", 1, 0.4),
        SVScore("p7", "p8", 1, 0.2),
        SVScore("n1", "n2", 0, 0.8),
        SVScore("n3", "n4", 0, 0.6),
        SVScore("n5", "n6", 0, 0.3),
        SVScore("n7", "n8", 0, 0.1),
    ]
    selected = select_sv_target_far_operating_point(scores, 0.25)
    assert selected == {"threshold": 0.7, "far": 0.25, "frr": 0.5, "tpr": 0.5}
    feasible = [
        rates_at_threshold(scores, threshold)
        for threshold in {item.score for item in scores}
        if rates_at_threshold(scores, threshold)["far"] <= 0.25
    ]
    assert selected["frr"] == min(point["frr"] for point in feasible)


def test_target_far_tie_prefers_higher_threshold():
    scores = [
        SVScore("p1", "p2", 1, 0.9),
        SVScore("n1", "n2", 0, 0.8),
        SVScore("n3", "n4", 0, 0.7),
    ]
    selected = select_sv_target_far_operating_point(scores, 1.0)
    assert selected["threshold"] == 0.9
    assert selected["frr"] == 0.0


def test_target_far_sentinel_provides_far_zero_fallback():
    scores = [
        SVScore("p1", "p2", 1, 0.5),
        SVScore("n1", "n2", 0, 0.9),
    ]
    selected = select_sv_target_far_operating_point(scores, 0.0)
    assert selected["threshold"] > 0.9
    assert selected["far"] == 0.0
    assert selected["frr"] == 1.0


def test_deployment_calibration_does_not_change_empirical_eer():
    scores = [
        SVScore("p1", "p2", 1, 0.9),
        SVScore("p3", "p4", 1, 0.4),
        SVScore("n1", "n2", 0, 0.7),
        SVScore("n3", "n4", 0, 0.2),
    ]
    intrinsic = compute_sv_metrics(scores)
    calibration = calibrate_sv_threshold(scores, target_far=0.0)
    assert calibration["validation_eer"] == intrinsic["eer"]
    assert calibration["eer_threshold"] == intrinsic["eer_threshold"]
    assert calibration["deployment_threshold"] != calibration["eer_threshold"]


def test_target_far_is_serialized_and_bound_by_provenance(tmp_path):
    scores = [
        SVScore("p1", "p2", 1, 0.9),
        SVScore("n1", "n2", 0, 0.2),
    ]
    path = tmp_path / "sv_calibration.json"
    written = write_sv_calibration(
        path,
        scores,
        target_far=0.05,
        provenance={"sv_target_far": 0.05},
    )
    assert written["deployment_target_far"] == 0.05
    assert load_frozen_sv_calibration(
        path, expected_provenance={"sv_target_far": 0.05}
    )["deployment_target_far"] == 0.05
    with pytest.raises(EvaluationError, match="provenance mismatch"):
        load_frozen_sv_calibration(
            path, expected_provenance={"sv_target_far": 0.10}
        )
