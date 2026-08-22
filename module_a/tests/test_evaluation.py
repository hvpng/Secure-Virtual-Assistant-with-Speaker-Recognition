from __future__ import annotations

import numpy as np

from module_a.src.data import AudioRecord
from module_a.src.evaluation import (
    SIDProbeScore,
    SVScore,
    build_sid_protocol,
    build_sv_trials,
    calibrate_sid,
    calibrate_sv,
    compute_sv_metrics,
    sid_metrics,
)


def test_sv_trials_are_deterministic_balanced_and_valid():
    records = [
        AudioRecord(f"{speaker}/{index}.wav", speaker, "validation")
        for speaker in ("a", "b", "c")
        for index in range(3)
    ]
    first = build_sv_trials(records, seed=42, max_positive_per_speaker=2)
    second = build_sv_trials(list(reversed(records)), seed=42, max_positive_per_speaker=2)
    assert first == second
    assert sum(trial.label == 1 for trial in first) == sum(trial.label == 0 for trial in first)
    assert all(trial.path_a != trial.path_b for trial in first)


def test_sv_metrics_and_target_far_calibration_are_empirical():
    scores = [
        SVScore("p1", "p2", 1, 0.9),
        SVScore("p3", "p4", 1, 0.7),
        SVScore("n1", "n2", 0, 0.8),
        SVScore("n3", "n4", 0, 0.1),
    ]
    metrics = compute_sv_metrics(scores)
    calibration = calibrate_sv(scores, 0.0)
    assert 0 <= metrics["eer"] <= 1
    assert calibration["far"] == 0.0
    assert calibration["source"] == "validation"


def test_sid_target_unknown_far_uses_same_embedding_score_contract():
    scores = [
        SIDProbeScore("k1", "a", "known", "a", 0.9),
        SIDProbeScore("k2", "b", "known", "b", 0.7),
        SIDProbeScore("u1", "u", "unknown", "a", 0.8),
        SIDProbeScore("u2", "v", "unknown", "b", 0.2),
    ]
    calibration = calibrate_sid(scores, 0.5)
    metrics = sid_metrics(scores, calibration["threshold"])
    assert metrics["unknown_false_accept_rate"] <= 0.5
    assert metrics["known_top1_identity_accuracy"] == 1.0


def test_sid_protocol_is_open_set_and_deterministic():
    records = [
        AudioRecord(f"{speaker}/{index}.wav", speaker, "validation")
        for speaker in ("a", "b", "c", "d")
        for index in range(3)
    ]
    first = build_sid_protocol(records, seed=42, known_ratio=0.5, max_enrollment=1)
    second = build_sid_protocol(list(reversed(records)), seed=42, known_ratio=0.5, max_enrollment=1)
    assert first == second
    assert set(first["known_speakers"]).isdisjoint(first["unknown_speakers"])
    enrollment = {path for paths in first["enrollment"].values() for path in paths}
    probes = {probe["path"] for probe in first["probes"]}
    assert enrollment.isdisjoint(probes)

