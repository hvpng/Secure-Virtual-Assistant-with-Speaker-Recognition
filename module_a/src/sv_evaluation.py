"""Deterministic A4 speaker-verification trials, scoring, and calibration."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from module_a.src.evaluation import EvaluationError, write_json_atomic
from module_a.src.training_data import TrainingRecord


SV_CALIBRATION_SCHEMA_VERSION = 2
SV_CALIBRATION_POLICY_VERSION = "sv_eer_and_target_far_empirical_v1"


@dataclass(frozen=True)
class SVTrial:
    path_a: str
    path_b: str
    speaker_a: str
    speaker_b: str
    label: int


@dataclass(frozen=True)
class SVScore:
    path_a: str
    path_b: str
    label: int
    score: float


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _canonical_trial(
    first: TrainingRecord, second: TrainingRecord, label: int
) -> SVTrial:
    if first.path <= second.path:
        left, right = first, second
    else:
        left, right = second, first
    return SVTrial(left.path, right.path, left.speaker_id, right.speaker_id, label)


def build_sv_trials(
    records: Sequence[TrainingRecord],
    *,
    seed: int,
    max_positive_per_speaker: int,
    total_negative_trials: int | None = None,
) -> list[SVTrial]:
    """Build bounded positives and an equal deterministic set of negatives."""

    if max_positive_per_speaker <= 0:
        raise EvaluationError("max_positive_per_speaker must be positive.")
    grouped: dict[str, list[TrainingRecord]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record)
    if len(grouped) < 2:
        raise EvaluationError("SV trials require at least two speakers.")
    for items in grouped.values():
        items.sort(key=lambda item: item.path)

    positives: list[SVTrial] = []
    for speaker_id in sorted(grouped):
        pairs = list(combinations(grouped[speaker_id], 2))
        if len(pairs) > max_positive_per_speaker:
            pairs = random.Random(_stable_seed(seed, speaker_id)).sample(
                pairs, max_positive_per_speaker
            )
        positives.extend(_canonical_trial(first, second, 1) for first, second in pairs)
    positives.sort(key=lambda trial: (trial.path_a, trial.path_b))
    if not positives:
        raise EvaluationError("SV protocol contains no positive trials.")

    speakers = sorted(grouped)
    maximum_negative = sum(
        len(grouped[first]) * len(grouped[second])
        for first, second in combinations(speakers, 2)
    )
    target_negative = len(positives) if total_negative_trials is None else total_negative_trials
    if target_negative <= 0:
        raise EvaluationError("total_negative_trials must be positive.")
    if target_negative > maximum_negative:
        raise EvaluationError("Requested more unique negative trials than are available.")

    rng = random.Random(seed)
    negative_by_pair: dict[tuple[str, str], SVTrial] = {}
    attempts = 0
    maximum_attempts = target_negative * 100 + 1_000
    while len(negative_by_pair) < target_negative and attempts < maximum_attempts:
        speaker_a, speaker_b = rng.sample(speakers, 2)
        first = rng.choice(grouped[speaker_a])
        second = rng.choice(grouped[speaker_b])
        trial = _canonical_trial(first, second, 0)
        negative_by_pair[(trial.path_a, trial.path_b)] = trial
        attempts += 1
    if len(negative_by_pair) < target_negative:
        for speaker_a, speaker_b in combinations(speakers, 2):
            for first in grouped[speaker_a]:
                for second in grouped[speaker_b]:
                    trial = _canonical_trial(first, second, 0)
                    negative_by_pair[(trial.path_a, trial.path_b)] = trial
                    if len(negative_by_pair) == target_negative:
                        break
                if len(negative_by_pair) == target_negative:
                    break
            if len(negative_by_pair) == target_negative:
                break
    negatives = sorted(
        negative_by_pair.values(), key=lambda trial: (trial.path_a, trial.path_b)
    )
    trials = positives + negatives
    validate_sv_trials(trials)
    if total_negative_trials is None and len(positives) != len(negatives):
        raise EvaluationError("Default SV protocol must balance positive/negative trials.")
    return trials


def validate_sv_trials(trials: Sequence[SVTrial]) -> None:
    if not trials:
        raise EvaluationError("SV trial protocol is empty.")
    seen: set[tuple[str, str]] = set()
    labels: set[int] = set()
    for trial in trials:
        if trial.label not in {0, 1}:
            raise EvaluationError("SV label must be 0 or 1.")
        if trial.path_a == trial.path_b:
            raise EvaluationError("SV self-pair is forbidden.")
        key = tuple(sorted((trial.path_a, trial.path_b)))
        if key in seen:
            raise EvaluationError("Duplicate unordered SV trial pair detected.")
        seen.add(key)
        labels.add(trial.label)
        if (trial.speaker_a == trial.speaker_b) != (trial.label == 1):
            raise EvaluationError("SV trial label conflicts with speaker identity.")
    if labels != {0, 1}:
        raise EvaluationError("SV protocol requires positive and negative trials.")


def write_sv_trials(path: str | Path, trials: Sequence[SVTrial]) -> Path:
    validate_sv_trials(trials)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("path_a", "path_b", "speaker_a", "speaker_b", "label"),
        )
        writer.writeheader()
        writer.writerows(asdict(trial) for trial in trials)
    return output


def score_sv_trials(
    trials: Sequence[SVTrial], embeddings: Mapping[str, np.ndarray]
) -> list[SVScore]:
    validate_sv_trials(trials)
    scores: list[SVScore] = []
    for trial in trials:
        if trial.path_a not in embeddings or trial.path_b not in embeddings:
            raise EvaluationError("SV trial references a missing embedding.")
        score = float(np.dot(embeddings[trial.path_a], embeddings[trial.path_b]))
        if not math.isfinite(score) or not -1.0001 <= score <= 1.0001:
            raise EvaluationError("SV cosine score is non-finite or outside cosine range.")
        scores.append(SVScore(trial.path_a, trial.path_b, trial.label, score))
    return scores


def write_sv_scores(path: str | Path, scores: Sequence[SVScore]) -> Path:
    if not scores:
        raise EvaluationError("SV scores are empty.")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("path_a", "path_b", "label", "score")
        )
        writer.writeheader()
        writer.writerows(asdict(score) for score in scores)
    return output


def rates_at_threshold(scores: Sequence[SVScore], threshold: float) -> dict[str, float]:
    positives = [item.score for item in scores if item.label == 1]
    negatives = [item.score for item in scores if item.label == 0]
    if not positives or not negatives or not math.isfinite(threshold):
        raise EvaluationError("FAR/FRR requires finite threshold and both trial classes.")
    far = sum(score >= threshold for score in negatives) / len(negatives)
    frr = sum(score < threshold for score in positives) / len(positives)
    return {"far": float(far), "frr": float(frr), "tpr": float(1.0 - frr)}


def _threshold_grid(scores: Sequence[SVScore]) -> list[float]:
    values = sorted({float(item.score) for item in scores}, reverse=True)
    if not values or not all(math.isfinite(value) for value in values):
        raise EvaluationError("SV threshold grid requires finite scores.")
    return [math.nextafter(values[0], math.inf), *values]


def _auc(roc: Sequence[Mapping[str, float]]) -> float:
    points = sorted((float(item["far"]), float(item["tpr"])) for item in roc)
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return float(area)


def score_distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return {"count": int(array.size), "mean": None, "std": None, "quantiles": {}}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "quantiles": {
            "q05": float(np.quantile(array, 0.05)),
            "q25": float(np.quantile(array, 0.25)),
            "q50": float(np.quantile(array, 0.50)),
            "q75": float(np.quantile(array, 0.75)),
            "q95": float(np.quantile(array, 0.95)),
        },
    }


def compute_sv_metrics(scores: Sequence[SVScore]) -> dict[str, Any]:
    if not scores:
        raise EvaluationError("Cannot compute SV metrics from empty scores.")
    roc: list[dict[str, float]] = []
    for threshold in _threshold_grid(scores):
        rates = rates_at_threshold(scores, threshold)
        roc.append({"threshold": threshold, **rates})
    operating = min(
        roc,
        key=lambda point: (
            abs(point["far"] - point["frr"]),
            -point["threshold"],
        ),
    )
    positives = [item.score for item in scores if item.label == 1]
    negatives = [item.score for item in scores if item.label == 0]
    return {
        "eer": float((operating["far"] + operating["frr"]) / 2.0),
        "eer_threshold": float(operating["threshold"]),
        "far_at_eer_threshold": float(operating["far"]),
        "frr_at_eer_threshold": float(operating["frr"]),
        "eer_method": "closest empirical FAR/FRR point; no interpolation; higher-threshold tie-break",
        "auc": _auc(roc),
        "roc_curve": roc,
        "positive_trials": len(positives),
        "negative_trials": len(negatives),
        "positive_score_distribution": score_distribution(positives),
        "negative_score_distribution": score_distribution(negatives),
    }


def select_sv_target_far_operating_point(
    scores: Sequence[SVScore], target_far: float
) -> dict[str, float]:
    """Select the empirical target-FAR point without interpolation.

    Candidates include one threshold strictly above the maximum observed score, so
    FAR=0 is always feasible. Among points with FAR <= ``target_far``, the point
    with the lowest FRR is selected; equal-FRR behavior prefers the higher threshold.
    """

    if not math.isfinite(target_far) or not 0.0 <= target_far <= 1.0:
        raise EvaluationError("SV target FAR must be finite and between 0 and 1.")
    candidates = [
        {"threshold": threshold, **rates_at_threshold(scores, threshold)}
        for threshold in _threshold_grid(scores)
    ]
    feasible = [point for point in candidates if point["far"] <= target_far]
    if not feasible:  # Defensive: the above-maximum sentinel should make this impossible.
        raise EvaluationError("No empirical SV threshold satisfies the target FAR.")
    return min(feasible, key=lambda point: (point["frr"], -point["threshold"]))


def calibrate_sv_threshold(
    scores: Sequence[SVScore], *, target_far: float = 0.05
) -> dict[str, Any]:
    metrics = compute_sv_metrics(scores)
    deployment = select_sv_target_far_operating_point(scores, target_far)
    return {
        "calibration_schema_version": SV_CALIBRATION_SCHEMA_VERSION,
        "calibration_policy_version": SV_CALIBRATION_POLICY_VERSION,
        "source_split": "validation",
        "intrinsic_policy": "empirical_eer",
        "validation_eer": metrics["eer"],
        "eer_threshold": metrics["eer_threshold"],
        "far_at_eer_threshold": metrics["far_at_eer_threshold"],
        "frr_at_eer_threshold": metrics["frr_at_eer_threshold"],
        "deployment_policy": "target_far",
        "deployment_objective": (
            "lowest empirical FRR subject to FAR <= target; higher-threshold tie-break"
        ),
        "deployment_target_far": float(target_far),
        "deployment_threshold": float(deployment["threshold"]),
        "deployment_far": float(deployment["far"]),
        "deployment_frr": float(deployment["frr"]),
        "deployment_tpr": float(deployment["tpr"]),
        "deployment_tar": float(deployment["tpr"]),
        "positive_trials": metrics["positive_trials"],
        "negative_trials": metrics["negative_trials"],
        "positive_score_distribution": metrics["positive_score_distribution"],
        "negative_score_distribution": metrics["negative_score_distribution"],
    }


def load_frozen_sv_calibration(
    path: str | Path,
    *,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from module_a.src.evaluation import read_json_object

    calibration = read_json_object(path)
    threshold = calibration.get("deployment_threshold")
    target_far = calibration.get("deployment_target_far")
    if (
        calibration.get("calibration_schema_version")
        != SV_CALIBRATION_SCHEMA_VERSION
        or calibration.get("calibration_policy_version")
        != SV_CALIBRATION_POLICY_VERSION
        or calibration.get("source_split") != "validation"
        or calibration.get("deployment_policy") != "target_far"
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not isinstance(target_far, (int, float))
        or not math.isfinite(float(target_far))
        or not 0 <= float(target_far) <= 1
    ):
        raise EvaluationError("SV calibration is not a frozen validation artifact.")
    for key, expected in (expected_provenance or {}).items():
        if calibration.get(key) != expected:
            raise EvaluationError(f"SV calibration provenance mismatch: {key}")
    return calibration


def write_sv_calibration(
    path: str | Path,
    scores: Sequence[SVScore],
    *,
    target_far: float = 0.05,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calibration = dict(provenance or {})
    calibration.update(calibrate_sv_threshold(scores, target_far=target_far))
    calibration["source_split"] = "validation"
    write_json_atomic(path, calibration)
    return calibration
