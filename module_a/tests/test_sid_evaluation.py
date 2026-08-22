from __future__ import annotations

import numpy as np

from module_a.src.sid_evaluation import (
    SIDProbeScore,
    build_sid_protocol,
    build_sid_prototypes,
    calibrate_sid_threshold,
    score_sid_probes,
    sid_metrics_at_threshold,
)
from module_a.src.training_data import TrainingRecord


def _records(split: str):
    return [
        TrainingRecord(f"{split}/{speaker}/{index}.wav", speaker, split)
        for speaker in (f"{split}_a", f"{split}_b", f"{split}_c", f"{split}_d", f"{split}_e")
        for index in range(6)
    ]


def test_sid_protocol_is_deterministic_disjoint_and_bounded():
    first = build_sid_protocol(_records("val"), split="val", seed=42, known_ratio=0.8)
    second = build_sid_protocol(
        list(reversed(_records("val"))), split="val", seed=42, known_ratio=0.8
    )
    assert first == second
    assert len(first.known_speakers) == 4
    assert len(first.unknown_speakers) == 1
    enrollment = {path for paths in first.enrollment_paths.values() for path in paths}
    probes = {probe.path for probe in first.probes}
    assert enrollment.isdisjoint(probes)
    assert all(1 <= len(paths) <= 5 for paths in first.enrollment_paths.values())
    assert all(speaker not in first.enrollment_paths for speaker in first.unknown_speakers)


def test_val_and_test_sid_protocols_are_independent():
    val = build_sid_protocol(_records("val"), split="val", seed=42)
    test = build_sid_protocol(_records("test"), split="test", seed=42)
    val_paths = {path for paths in val.enrollment_paths.values() for path in paths} | {
        probe.path for probe in val.probes
    }
    test_paths = {path for paths in test.enrollment_paths.values() for path in paths} | {
        probe.path for probe in test.probes
    }
    assert set(val.known_speakers).isdisjoint(test.known_speakers)
    assert val_paths.isdisjoint(test_paths)


def test_sid_prototype_is_normalized_mean_and_argmax_scores_correct():
    protocol = build_sid_protocol(_records("val"), split="val", seed=42)
    speaker_axis = {
        speaker: index for index, speaker in enumerate(sorted(set(protocol.known_speakers) | set(protocol.unknown_speakers)))
    }
    embeddings = {}
    for record in _records("val"):
        vector = np.zeros(8, dtype=np.float32)
        vector[speaker_axis[record.speaker_id]] = 1.0
        embeddings[record.path] = vector
    prototypes = build_sid_prototypes(protocol, embeddings)
    assert all(np.isclose(np.linalg.norm(vector), 1.0) for vector in prototypes.values())
    scores = score_sid_probes(protocol, embeddings, prototypes)
    assert all(
        score.best_speaker_id == score.true_speaker_id
        for score in scores
        if score.true_status == "known"
    )


def test_sid_metrics_and_rejection_are_exact_on_synthetic_scores():
    scores = [
        SIDProbeScore("k1", "a", "known", "a", 0.9),
        SIDProbeScore("k2", "b", "known", "a", 0.8),
        SIDProbeScore("u1", "u", "unknown", "a", 0.4),
        SIDProbeScore("u2", "v", "unknown", "b", 0.3),
    ]
    metrics = sid_metrics_at_threshold(scores, 0.5)
    assert metrics["known_top1_identity_accuracy"] == 0.5
    assert metrics["known_accepted_correct_rate"] == 0.5
    assert metrics["unknown_rejection_rate"] == 1.0
    assert metrics["unknown_false_accept_rate"] == 0.0
    assert metrics["overall_open_set_accuracy"] == 0.75


def test_sid_calibration_tie_prefers_higher_threshold():
    scores = [
        SIDProbeScore("k", "a", "known", "a", 0.5),
        SIDProbeScore("u", "u", "unknown", "a", 0.5),
    ]
    calibration = calibrate_sid_threshold(scores)
    assert calibration["source_split"] == "validation"
    assert calibration["selected_threshold"] > 0.5
    assert calibration["tie_breaker"] == "higher threshold"
