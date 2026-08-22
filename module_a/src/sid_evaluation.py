"""Deterministic open-set A4 SID gallery/probe evaluation and calibration."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from module_a.src.evaluation import (
    EvaluationError,
    read_json_object,
    validate_embedding_vector,
    write_json_atomic,
)
from module_a.src.sv_evaluation import score_distribution
from module_a.src.training_data import TrainingRecord


UNKNOWN_IDENTITY = "UNKNOWN"
SID_CALIBRATION_SCHEMA_VERSION = 3
SID_CALIBRATION_POLICY_VERSION = "sid_target_unknown_far_v1"


@dataclass(frozen=True)
class SIDProbe:
    path: str
    speaker_id: str
    status: str


@dataclass(frozen=True)
class SIDProtocol:
    split: str
    seed: int
    known_ratio: float
    max_enrollment_utterances: int
    known_speakers: tuple[str, ...]
    unknown_speakers: tuple[str, ...]
    enrollment_paths: dict[str, tuple[str, ...]]
    probes: tuple[SIDProbe, ...]

    def json_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "seed": self.seed,
            "known_ratio": self.known_ratio,
            "max_enrollment_utterances": self.max_enrollment_utterances,
            "known_speakers": list(self.known_speakers),
            "unknown_speakers": list(self.unknown_speakers),
            "enrollment_paths": {
                speaker: list(paths)
                for speaker, paths in sorted(self.enrollment_paths.items())
            },
            "probes": [
                {
                    "path": probe.path,
                    "speaker_id": probe.speaker_id,
                    "status": probe.status,
                }
                for probe in self.probes
            ],
        }


@dataclass(frozen=True)
class SIDProbeScore:
    probe_path: str
    true_speaker_id: str
    true_status: str
    best_speaker_id: str
    best_score: float


def _stable_seed(seed: int, split: str, speaker_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}:{speaker_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_sid_protocol(
    records: Sequence[TrainingRecord],
    *,
    split: str,
    seed: int,
    known_ratio: float = 0.8,
    max_enrollment_utterances: int = 5,
) -> SIDProtocol:
    if split not in {"val", "test"} or any(record.split != split for record in records):
        raise EvaluationError("SID protocol records must belong to one val/test split.")
    if not 0 < known_ratio < 1 or max_enrollment_utterances <= 0:
        raise EvaluationError("SID known ratio and enrollment cap are invalid.")
    grouped: dict[str, list[TrainingRecord]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record)
    speakers = sorted(grouped)
    if len(speakers) < 2:
        raise EvaluationError("Open-set SID requires at least two speakers.")
    known_count = min(
        len(speakers) - 1,
        max(1, math.floor(len(speakers) * known_ratio + 0.5)),
    )
    shuffled = speakers.copy()
    random.Random(seed).shuffle(shuffled)
    known_speakers = tuple(sorted(shuffled[:known_count]))
    unknown_speakers = tuple(sorted(shuffled[known_count:]))

    enrollment_paths: dict[str, tuple[str, ...]] = {}
    probes: list[SIDProbe] = []
    for speaker_id in known_speakers:
        items = sorted(grouped[speaker_id], key=lambda item: item.path)
        if len(items) < 2:
            raise EvaluationError(
                f"Known SID speaker requires enrollment and probe audio: {speaker_id}"
            )
        shuffled_items = items.copy()
        random.Random(_stable_seed(seed, split, speaker_id)).shuffle(shuffled_items)
        enrollment_count = min(max_enrollment_utterances, len(items) - 1)
        enrollment = tuple(sorted(item.path for item in shuffled_items[:enrollment_count]))
        enrollment_set = set(enrollment)
        enrollment_paths[speaker_id] = enrollment
        probes.extend(
            SIDProbe(item.path, speaker_id, "known")
            for item in items
            if item.path not in enrollment_set
        )
    for speaker_id in unknown_speakers:
        probes.extend(
            SIDProbe(item.path, speaker_id, "unknown")
            for item in sorted(grouped[speaker_id], key=lambda item: item.path)
        )
    protocol = SIDProtocol(
        split=split,
        seed=seed,
        known_ratio=known_ratio,
        max_enrollment_utterances=max_enrollment_utterances,
        known_speakers=known_speakers,
        unknown_speakers=unknown_speakers,
        enrollment_paths=enrollment_paths,
        probes=tuple(sorted(probes, key=lambda item: (item.status, item.speaker_id, item.path))),
    )
    validate_sid_protocol(protocol)
    return protocol


def validate_sid_protocol(protocol: SIDProtocol) -> None:
    known = set(protocol.known_speakers)
    unknown = set(protocol.unknown_speakers)
    if not known or not unknown or known & unknown:
        raise EvaluationError("SID known/unknown speaker sets must be non-empty and disjoint.")
    if set(protocol.enrollment_paths) != known:
        raise EvaluationError("Every and only known SID speakers require enrollment.")
    enrollment = {
        path for paths in protocol.enrollment_paths.values() for path in paths
    }
    if len(enrollment) != sum(len(paths) for paths in protocol.enrollment_paths.values()):
        raise EvaluationError("SID enrollment paths contain duplicates.")
    probe_paths = {probe.path for probe in protocol.probes}
    if len(probe_paths) != len(protocol.probes) or enrollment & probe_paths:
        raise EvaluationError("SID enrollment/probe paths must be unique and disjoint.")
    for speaker, paths in protocol.enrollment_paths.items():
        if not paths or len(paths) > protocol.max_enrollment_utterances:
            raise EvaluationError(f"SID enrollment count is invalid for {speaker}.")
    for probe in protocol.probes:
        expected = "known" if probe.speaker_id in known else "unknown"
        if probe.status != expected:
            raise EvaluationError("SID probe status conflicts with known/unknown split.")
    if not any(probe.status == "known" for probe in protocol.probes) or not any(
        probe.status == "unknown" for probe in protocol.probes
    ):
        raise EvaluationError("SID requires both known and unknown probes.")


def write_sid_protocol(path: str | Path, protocol: SIDProtocol) -> Path:
    validate_sid_protocol(protocol)
    return write_json_atomic(path, protocol.json_dict())


def build_sid_prototypes(
    protocol: SIDProtocol,
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    validate_sid_protocol(protocol)
    if not embeddings:
        raise EvaluationError("SID embeddings are empty.")
    dimension = len(next(iter(embeddings.values())))
    prototypes: dict[str, np.ndarray] = {}
    for speaker_id in protocol.known_speakers:
        paths = protocol.enrollment_paths[speaker_id]
        if any(path not in embeddings for path in paths):
            raise EvaluationError("SID enrollment references a missing embedding.")
        matrix = np.stack(
            [validate_embedding_vector(embeddings[path], dimension) for path in paths]
        ).astype(np.float32, copy=False)
        prototype = matrix.mean(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(prototype))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise EvaluationError(f"Cannot normalize SID prototype: {speaker_id}")
        prototype = (prototype / norm).astype(np.float32, copy=False)
        prototypes[speaker_id] = validate_embedding_vector(prototype, dimension).copy()
    return prototypes


def score_sid_probes(
    protocol: SIDProtocol,
    embeddings: Mapping[str, np.ndarray],
    prototypes: Mapping[str, np.ndarray],
) -> list[SIDProbeScore]:
    validate_sid_protocol(protocol)
    speakers = sorted(prototypes)
    if speakers != list(protocol.known_speakers):
        raise EvaluationError("SID prototypes do not match the known speaker gallery.")
    prototype_matrix = np.stack([prototypes[speaker] for speaker in speakers])
    scores: list[SIDProbeScore] = []
    for probe in protocol.probes:
        if probe.path not in embeddings:
            raise EvaluationError("SID probe references a missing embedding.")
        similarities = prototype_matrix @ embeddings[probe.path]
        if similarities.ndim != 1 or not np.isfinite(similarities).all():
            raise EvaluationError("SID probe produced invalid cosine scores.")
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        if not -1.0001 <= best_score <= 1.0001:
            raise EvaluationError("SID best score is outside cosine range.")
        scores.append(
            SIDProbeScore(
                probe_path=probe.path,
                true_speaker_id=probe.speaker_id,
                true_status=probe.status,
                best_speaker_id=speakers[best_index],
                best_score=best_score,
            )
        )
    if not scores:
        raise EvaluationError("SID probe score set is empty.")
    return scores


def sid_metrics_at_threshold(
    scores: Sequence[SIDProbeScore], threshold: float
) -> dict[str, Any]:
    known = [score for score in scores if score.true_status == "known"]
    unknown = [score for score in scores if score.true_status == "unknown"]
    if not known or not unknown or not math.isfinite(threshold):
        raise EvaluationError("SID metrics require known/unknown probes and finite threshold.")
    known_top1 = sum(
        score.best_speaker_id == score.true_speaker_id for score in known
    ) / len(known)
    known_accepted_correct = sum(
        score.best_score >= threshold
        and score.best_speaker_id == score.true_speaker_id
        for score in known
    )
    known_wrong_accept = sum(
        score.best_score >= threshold
        and score.best_speaker_id != score.true_speaker_id
        for score in known
    )
    known_rejected = sum(score.best_score < threshold for score in known)
    unknown_rejected = sum(score.best_score < threshold for score in unknown)
    unknown_false_accept = len(unknown) - unknown_rejected
    correct = known_accepted_correct + unknown_rejected
    known_accepted_correct_rate = known_accepted_correct / len(known)
    unknown_rejection_rate = unknown_rejected / len(unknown)
    return {
        "threshold": float(threshold),
        "known_probe_count": len(known),
        "unknown_probe_count": len(unknown),
        "known_top1_identity_accuracy": float(known_top1),
        "known_accepted_correct_rate": float(known_accepted_correct_rate),
        "known_wrong_accept_rate": float(known_wrong_accept / len(known)),
        "known_rejection_rate": float(known_rejected / len(known)),
        "unknown_rejection_rate": float(unknown_rejection_rate),
        "unknown_false_accept_rate": float(unknown_false_accept / len(unknown)),
        "balanced_open_set_accuracy": float(
            0.5 * known_accepted_correct_rate + 0.5 * unknown_rejection_rate
        ),
        "overall_open_set_accuracy": float(correct / len(scores)),
    }


def _sid_thresholds(scores: Sequence[SIDProbeScore]) -> list[float]:
    values = sorted({score.best_score for score in scores}, reverse=True)
    if not values or not all(math.isfinite(value) for value in values):
        raise EvaluationError("SID calibration requires finite scores.")
    return [math.nextafter(values[0], math.inf), *values]


def sid_score_distributions(scores: Sequence[SIDProbeScore]) -> dict[str, Any]:
    known_correct_scores = [
        score.best_score
        for score in scores
        if score.true_status == "known"
        and score.best_speaker_id == score.true_speaker_id
    ]
    known_incorrect_scores = [
        score.best_score
        for score in scores
        if score.true_status == "known"
        and score.best_speaker_id != score.true_speaker_id
    ]
    unknown_scores = [score.best_score for score in scores if score.true_status == "unknown"]
    return {
        "known_correct_score_distribution": score_distribution(known_correct_scores),
        "known_incorrect_score_distribution": score_distribution(known_incorrect_scores),
        "unknown_score_distribution": score_distribution(unknown_scores),
    }


def select_sid_target_unknown_far_operating_point(
    scores: Sequence[SIDProbeScore], target_unknown_far: float
) -> dict[str, Any]:
    """Select the empirical SID deployment point without interpolation.

    The threshold maximizes the known accepted-correct rate among candidates whose
    unknown false-accept rate does not exceed ``target_unknown_far``. Equal primary
    objectives prefer the higher threshold. The above-maximum sentinel guarantees a
    reject-all candidate with zero unknown false accepts.
    """

    if (
        not math.isfinite(target_unknown_far)
        or not 0.0 <= target_unknown_far <= 1.0
    ):
        raise EvaluationError(
            "SID target unknown FAR must be finite and between 0 and 1."
        )
    candidates = [
        sid_metrics_at_threshold(scores, value) for value in _sid_thresholds(scores)
    ]
    feasible = [
        metrics
        for metrics in candidates
        if metrics["unknown_false_accept_rate"] <= target_unknown_far
    ]
    if not feasible:  # Defensive: reject-all sentinel should always be feasible.
        raise EvaluationError("No empirical SID threshold satisfies target unknown FAR.")
    return max(
        feasible,
        key=lambda metrics: (
            metrics["known_accepted_correct_rate"],
            metrics["threshold"],
        ),
    )


def calibrate_sid_threshold(
    scores: Sequence[SIDProbeScore], *, target_unknown_far: float = 0.05
) -> dict[str, Any]:
    selected = select_sid_target_unknown_far_operating_point(
        scores, target_unknown_far
    )
    return {
        "calibration_schema_version": SID_CALIBRATION_SCHEMA_VERSION,
        "calibration_policy_version": SID_CALIBRATION_POLICY_VERSION,
        "source_split": "validation",
        "objective": (
            "maximize validation known accepted-correct rate subject to "
            "unknown FAR target"
        ),
        "deployment_policy": "target_unknown_far",
        "deployment_target_unknown_far": float(target_unknown_far),
        "tie_breaker": "higher threshold",
        "selected_threshold": selected["threshold"],
        **{key: value for key, value in selected.items() if key != "threshold"},
        **sid_score_distributions(scores),
    }


def load_frozen_sid_calibration(
    path: str | Path,
    *,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calibration = read_json_object(path)
    threshold = calibration.get("selected_threshold")
    target_unknown_far = calibration.get("deployment_target_unknown_far")
    if (
        calibration.get("calibration_schema_version")
        != SID_CALIBRATION_SCHEMA_VERSION
        or calibration.get("calibration_policy_version")
        != SID_CALIBRATION_POLICY_VERSION
        or calibration.get("source_split") != "validation"
        or calibration.get("objective")
        != (
            "maximize validation known accepted-correct rate subject to "
            "unknown FAR target"
        )
        or calibration.get("deployment_policy") != "target_unknown_far"
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not isinstance(target_unknown_far, (int, float))
        or not math.isfinite(float(target_unknown_far))
        or not 0 <= float(target_unknown_far) <= 1
    ):
        raise EvaluationError("SID calibration is not a frozen validation artifact.")
    for key, expected in (expected_provenance or {}).items():
        if calibration.get(key) != expected:
            raise EvaluationError(f"SID calibration provenance mismatch: {key}")
    return calibration


def write_sid_calibration(
    path: str | Path,
    scores: Sequence[SIDProbeScore],
    *,
    target_unknown_far: float = 0.05,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calibration = dict(provenance or {})
    calibration.update(
        calibrate_sid_threshold(scores, target_unknown_far=target_unknown_far)
    )
    calibration["source_split"] = "validation"
    write_json_atomic(path, calibration)
    return calibration


def write_sid_scores(
    path: str | Path,
    scores: Sequence[SIDProbeScore],
    *,
    threshold: float,
) -> Path:
    if not scores or not math.isfinite(threshold):
        raise EvaluationError("SID scored output requires scores and a finite threshold.")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "probe_path",
        "true_speaker_id",
        "true_status",
        "best_speaker_id",
        "best_score",
        "predicted_speaker_id",
        "correct",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for score in scores:
            accepted = score.best_score >= threshold
            predicted = score.best_speaker_id if accepted else UNKNOWN_IDENTITY
            correct = (
                predicted == score.true_speaker_id
                if score.true_status == "known"
                else predicted == UNKNOWN_IDENTITY
            )
            writer.writerow(
                {
                    "probe_path": score.probe_path,
                    "true_speaker_id": score.true_speaker_id,
                    "true_status": score.true_status,
                    "best_speaker_id": score.best_speaker_id,
                    "best_score": score.best_score,
                    "predicted_speaker_id": predicted,
                    "correct": correct,
                }
            )
    return output
