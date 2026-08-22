"""Phase-safe A4 validation calibration and frozen-threshold test evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from module_a.src.evaluation import EvaluationError, write_json_atomic
from module_a.src.sid_evaluation import (
    build_sid_protocol,
    build_sid_prototypes,
    calibrate_sid_threshold,
    load_frozen_sid_calibration,
    score_sid_probes,
    sid_metrics_at_threshold,
    sid_score_distributions,
    write_sid_calibration,
    write_sid_protocol,
    write_sid_scores,
)
from module_a.src.sv_evaluation import (
    build_sv_trials,
    compute_sv_metrics,
    load_frozen_sv_calibration,
    rates_at_threshold,
    score_sv_trials,
    write_sv_calibration,
    write_sv_scores,
    write_sv_trials,
)
from module_a.src.training_data import TrainingRecord


def _paths(output_dir: str | Path, split: str) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    return {
        "sv_trials": root / "trials" / f"sv_{split}_trials.csv",
        "sid_protocol": root / "protocols" / f"sid_{split}_protocol.json",
        "sv_scores": root / "scores" / f"sv_{split}_scores.csv",
        "sid_scores": root / "scores" / f"sid_{split}_scores.csv",
        "sv_metrics": root / "metrics" / f"sv_{split}_metrics.json",
        "sid_metrics": root / "metrics" / f"sid_{split}_metrics.json",
        "sv_calibration": root / "calibration" / "sv_calibration.json",
        "sid_calibration": root / "calibration" / "sid_calibration.json",
    }


def require_frozen_calibrations(
    output_dir: str | Path,
    *,
    expected_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load validation artifacts before test evaluation is allowed to begin."""

    paths = _paths(output_dir, "test")
    return (
        load_frozen_sv_calibration(
            paths["sv_calibration"], expected_provenance=expected_provenance
        ),
        load_frozen_sid_calibration(
            paths["sid_calibration"], expected_provenance=expected_provenance
        ),
    )


def run_validation_protocols(
    records: Sequence[TrainingRecord],
    embeddings: Mapping[str, np.ndarray],
    *,
    output_dir: str | Path,
    seed: int,
    max_sv_positive_per_speaker: int,
    sid_known_ratio: float,
    sid_max_enrollment: int,
    calibration_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate SV and SID exclusively from A1 validation speakers."""

    if not records or any(record.split != "val" for record in records):
        raise EvaluationError("Validation calibration accepts only val records.")
    paths = _paths(output_dir, "val")

    sv_trials = build_sv_trials(
        records,
        seed=seed,
        max_positive_per_speaker=max_sv_positive_per_speaker,
    )
    write_sv_trials(paths["sv_trials"], sv_trials)
    sv_scores = score_sv_trials(sv_trials, embeddings)
    write_sv_scores(paths["sv_scores"], sv_scores)
    sv_metrics = compute_sv_metrics(sv_scores)
    write_json_atomic(paths["sv_metrics"], sv_metrics)
    sv_calibration = write_sv_calibration(
        paths["sv_calibration"], sv_scores, provenance=calibration_provenance
    )

    sid_protocol = build_sid_protocol(
        records,
        split="val",
        seed=seed,
        known_ratio=sid_known_ratio,
        max_enrollment_utterances=sid_max_enrollment,
    )
    write_sid_protocol(paths["sid_protocol"], sid_protocol)
    prototypes = build_sid_prototypes(sid_protocol, embeddings)
    sid_scores = score_sid_probes(sid_protocol, embeddings, prototypes)
    sid_calibration = write_sid_calibration(
        paths["sid_calibration"], sid_scores, provenance=calibration_provenance
    )
    sid_threshold = float(sid_calibration["selected_threshold"])
    sid_metrics = sid_metrics_at_threshold(sid_scores, sid_threshold)
    sid_metrics.update(sid_score_distributions(sid_scores))
    sid_metrics["source_threshold"] = "validation_calibration"
    sid_metrics["calibration_objective"] = sid_calibration["objective"]
    write_sid_scores(paths["sid_scores"], sid_scores, threshold=sid_threshold)
    write_json_atomic(paths["sid_metrics"], sid_metrics)

    return {
        "split": "validation",
        "sv": {"metrics": sv_metrics, "calibration": sv_calibration},
        "sid": {
            "metrics": sid_metrics,
            "calibration": sid_calibration,
            "known_speakers": len(sid_protocol.known_speakers),
            "unknown_speakers": len(sid_protocol.unknown_speakers),
        },
    }


def run_test_protocols(
    records: Sequence[TrainingRecord],
    embeddings: Mapping[str, np.ndarray],
    *,
    output_dir: str | Path,
    seed: int,
    max_sv_positive_per_speaker: int,
    sid_known_ratio: float,
    sid_max_enrollment: int,
    expected_calibration_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate test using persisted validation thresholds; never recalibrate."""

    if not records or any(record.split != "test" for record in records):
        raise EvaluationError("Test evaluation accepts only test records.")
    # This guard deliberately runs before creating any test protocol or score artifact.
    sv_calibration, sid_calibration = require_frozen_calibrations(
        output_dir, expected_provenance=expected_calibration_provenance
    )
    sv_threshold = float(sv_calibration["selected_threshold"])
    sid_threshold = float(sid_calibration["selected_threshold"])
    paths = _paths(output_dir, "test")

    sv_trials = build_sv_trials(
        records,
        seed=seed,
        max_positive_per_speaker=max_sv_positive_per_speaker,
    )
    write_sv_trials(paths["sv_trials"], sv_trials)
    sv_scores = score_sv_trials(sv_trials, embeddings)
    write_sv_scores(paths["sv_scores"], sv_scores)
    descriptive = compute_sv_metrics(sv_scores)
    frozen_rates = rates_at_threshold(sv_scores, sv_threshold)
    sv_metrics = {
        "test_eer": descriptive["eer"],
        "test_eer_threshold": descriptive["eer_threshold"],
        "test_eer_threshold_policy": "descriptive_only_not_deployment",
        "frozen_validation_sv_threshold": sv_threshold,
        "test_far_at_frozen_threshold": frozen_rates["far"],
        "test_frr_at_frozen_threshold": frozen_rates["frr"],
        "auc": descriptive["auc"],
        "positive_trials": descriptive["positive_trials"],
        "negative_trials": descriptive["negative_trials"],
        "positive_score_distribution": descriptive["positive_score_distribution"],
        "negative_score_distribution": descriptive["negative_score_distribution"],
        "source_threshold": "persisted_validation_calibration",
    }
    write_json_atomic(paths["sv_metrics"], sv_metrics)

    sid_protocol = build_sid_protocol(
        records,
        split="test",
        seed=seed,
        known_ratio=sid_known_ratio,
        max_enrollment_utterances=sid_max_enrollment,
    )
    write_sid_protocol(paths["sid_protocol"], sid_protocol)
    prototypes = build_sid_prototypes(sid_protocol, embeddings)
    sid_scores = score_sid_probes(sid_protocol, embeddings, prototypes)
    sid_metrics = sid_metrics_at_threshold(sid_scores, sid_threshold)
    sid_metrics.update(sid_score_distributions(sid_scores))
    sid_metrics["frozen_validation_sid_threshold"] = sid_threshold
    sid_metrics["source_threshold"] = "persisted_validation_calibration"
    write_sid_scores(paths["sid_scores"], sid_scores, threshold=sid_threshold)
    write_json_atomic(paths["sid_metrics"], sid_metrics)

    return {
        "split": "test",
        "sv": {"metrics": sv_metrics},
        "sid": {
            "metrics": sid_metrics,
            "known_speakers": len(sid_protocol.known_speakers),
            "unknown_speakers": len(sid_protocol.unknown_speakers),
        },
    }
