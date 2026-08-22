"""Shared embedding extraction, SV metrics, and open-set SID evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from module_a.src.config import save_json
from module_a.src.data import AudioRecord, VoxVietnamDataset, records_fingerprint
from module_a.src.ecapa import SpeakerEmbeddingModel


class EvaluationError(RuntimeError):
    """Raised when an evaluation protocol or metric is invalid."""


@dataclass(frozen=True)
class SVTrial:
    path_a: str
    path_b: str
    label: int


@dataclass(frozen=True)
class SVScore:
    path_a: str
    path_b: str
    label: int
    score: float


@dataclass(frozen=True)
class SIDProbeScore:
    path: str
    true_speaker: str
    status: str
    best_speaker: str
    best_score: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_embeddings(
    model: SpeakerEmbeddingModel,
    records: Sequence[AudioRecord],
    dataset_root: str | Path,
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = VoxVietnamDataset(
        records,
        dataset_root,
        sample_rate=int(config["audio"]["sample_rate"]),
        segment_seconds=float(config["audio"]["segment_seconds"]),
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    model.eval()
    embeddings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            vectors = model.extract_embedding(batch["waveform"].to(device)).float().cpu().numpy()
            for path, vector in zip(batch["path"], vectors, strict=True):
                array = np.asarray(vector, dtype=np.float32)
                if (
                    array.shape != (int(config["model"]["embedding_dim"]),)
                    or not np.isfinite(array).all()
                    or not np.isclose(np.linalg.norm(array), 1.0, atol=1e-4)
                ):
                    raise EvaluationError(f"Invalid embedding extracted for: {path}")
                embeddings[str(path)] = array
    if len(embeddings) != len(records):
        raise EvaluationError("Embedding extraction did not cover every manifest record.")
    return embeddings


def build_sv_trials(
    records: Sequence[AudioRecord], *, seed: int, max_positive_per_speaker: int
) -> list[SVTrial]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record.path)
    if len(grouped) < 2 or max_positive_per_speaker <= 0:
        raise EvaluationError("SV trials require at least two speakers and a positive cap.")
    positives: list[SVTrial] = []
    for paths in grouped.values():
        paths.sort()
    for speaker in sorted(grouped):
        pairs = list(combinations(grouped[speaker], 2))
        if len(pairs) > max_positive_per_speaker:
            speaker_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{speaker}".encode()).digest()[:8], "big"
            )
            pairs = random.Random(speaker_seed).sample(pairs, max_positive_per_speaker)
        positives.extend(SVTrial(min(a, b), max(a, b), 1) for a, b in pairs)
    positives.sort(key=lambda trial: (trial.path_a, trial.path_b))
    if not positives:
        raise EvaluationError("SV protocol contains no positive trials.")

    speakers = sorted(grouped)
    rng = random.Random(seed)
    negatives: dict[tuple[str, str], SVTrial] = {}
    maximum_attempts = len(positives) * 100 + 1000
    for _ in range(maximum_attempts):
        if len(negatives) == len(positives):
            break
        left_speaker, right_speaker = rng.sample(speakers, 2)
        a = rng.choice(grouped[left_speaker])
        b = rng.choice(grouped[right_speaker])
        key = tuple(sorted((a, b)))
        negatives[key] = SVTrial(key[0], key[1], 0)
    if len(negatives) != len(positives):
        raise EvaluationError("Cannot create enough unique negative SV trials.")
    trials = positives + sorted(negatives.values(), key=lambda item: (item.path_a, item.path_b))
    validate_sv_trials(trials)
    return trials


def validate_sv_trials(trials: Sequence[SVTrial]) -> None:
    if not trials or {trial.label for trial in trials} != {0, 1}:
        raise EvaluationError("SV trials require both positive and negative labels.")
    seen: set[tuple[str, str]] = set()
    for trial in trials:
        if trial.path_a == trial.path_b or trial.label not in {0, 1}:
            raise EvaluationError("SV trial contains a self-pair or invalid label.")
        key = tuple(sorted((trial.path_a, trial.path_b)))
        if key in seen:
            raise EvaluationError("Duplicate unordered SV trial pair detected.")
        seen.add(key)


def load_official_sv_trials(
    path: str | Path, available_paths: set[str]
) -> list[SVTrial]:
    trial_path = Path(path).expanduser().resolve()
    if not trial_path.is_file():
        raise EvaluationError(f"Official trial file does not exist: {trial_path}")
    trials: list[SVTrial] = []
    lines = trial_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [part.strip() for part in (line.split(",") if "," in line else line.split())]
        if line_number == 1 and {value.lower() for value in values} >= {"label"}:
            continue
        if len(values) != 3:
            raise EvaluationError(f"Unsupported official trial row at line {line_number}.")
        if values[0].lower() in {"0", "1", "target", "nontarget", "same", "different"}:
            raw_label, first, second = values
        else:
            first, second, raw_label = values
        normalized_label = raw_label.lower()
        if normalized_label in {"1", "target", "same"}:
            label = 1
        elif normalized_label in {"0", "nontarget", "different"}:
            label = 0
        else:
            raise EvaluationError(f"Unknown official trial label: {raw_label}")
        first = first.replace("\\", "/").lstrip("./")
        second = second.replace("\\", "/").lstrip("./")
        if first not in available_paths or second not in available_paths:
            raise EvaluationError(
                "Official VoxVietnam-O trial references audio absent from the mounted dataset."
            )
        trials.append(SVTrial(first, second, label))
    validate_sv_trials(trials)
    return trials


def score_sv_trials(
    trials: Sequence[SVTrial], embeddings: Mapping[str, np.ndarray]
) -> list[SVScore]:
    scores: list[SVScore] = []
    for trial in trials:
        try:
            score = float(np.dot(embeddings[trial.path_a], embeddings[trial.path_b]))
        except KeyError as exc:
            raise EvaluationError("SV trial references a missing embedding.") from exc
        if not math.isfinite(score) or not -1.0001 <= score <= 1.0001:
            raise EvaluationError("SV cosine score is invalid.")
        scores.append(SVScore(trial.path_a, trial.path_b, trial.label, score))
    return scores


def _thresholds(values: Sequence[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values), reverse=True)
    if not unique or not all(math.isfinite(value) for value in unique):
        raise EvaluationError("Threshold candidates require finite scores.")
    return [math.nextafter(unique[0], math.inf), *unique]


def sv_rates(scores: Sequence[SVScore], threshold: float) -> dict[str, float]:
    positives = [score.score for score in scores if score.label == 1]
    negatives = [score.score for score in scores if score.label == 0]
    if not positives or not negatives:
        raise EvaluationError("SV rates require both trial classes.")
    far = sum(value >= threshold for value in negatives) / len(negatives)
    frr = sum(value < threshold for value in positives) / len(positives)
    return {"far": float(far), "frr": float(frr), "tpr": float(1.0 - frr)}


def compute_sv_metrics(scores: Sequence[SVScore]) -> dict[str, Any]:
    roc = [
        {"threshold": threshold, **sv_rates(scores, threshold)}
        for threshold in _thresholds([score.score for score in scores])
    ]
    eer_point = min(
        roc, key=lambda item: (abs(item["far"] - item["frr"]), -item["threshold"])
    )
    points = sorted((item["far"], item["tpr"]) for item in roc)
    auc = sum(
        (x1 - x0) * (y0 + y1) / 2
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )
    return {
        "eer": float((eer_point["far"] + eer_point["frr"]) / 2),
        "eer_threshold": float(eer_point["threshold"]),
        "auc": float(auc),
        "far_at_eer": float(eer_point["far"]),
        "frr_at_eer": float(eer_point["frr"]),
        "positive_trials": sum(score.label == 1 for score in scores),
        "negative_trials": sum(score.label == 0 for score in scores),
        "eer_method": "closest empirical FAR/FRR; no interpolation; higher-threshold tie-break",
    }


def calibrate_sv(scores: Sequence[SVScore], target_far: float) -> dict[str, Any]:
    if not 0 <= target_far <= 1:
        raise EvaluationError("SV target FAR must be between 0 and 1.")
    candidates = [
        {"threshold": threshold, **sv_rates(scores, threshold)}
        for threshold in _thresholds([score.score for score in scores])
    ]
    feasible = [candidate for candidate in candidates if candidate["far"] <= target_far]
    selected = min(feasible, key=lambda item: (item["frr"], -item["threshold"]))
    intrinsic = compute_sv_metrics(scores)
    return {
        "schema_version": 1,
        "source": "validation",
        "policy": "target_far_empirical_v1",
        "target_far": float(target_far),
        "threshold": float(selected["threshold"]),
        "far": float(selected["far"]),
        "frr": float(selected["frr"]),
        "tar": float(selected["tpr"]),
        "validation_eer": intrinsic["eer"],
        "validation_eer_threshold": intrinsic["eer_threshold"],
    }


def build_sid_protocol(
    records: Sequence[AudioRecord],
    *,
    seed: int,
    known_ratio: float,
    max_enrollment: int,
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record.path)
    speakers = sorted(grouped)
    if len(speakers) < 2 or not 0 < known_ratio < 1 or max_enrollment <= 0:
        raise EvaluationError("Invalid open-set SID protocol configuration.")
    shuffled = speakers.copy()
    random.Random(seed).shuffle(shuffled)
    known_count = max(1, min(len(speakers) - 1, round(len(speakers) * known_ratio)))
    known = sorted(shuffled[:known_count])
    unknown = sorted(shuffled[known_count:])
    enrollment: dict[str, list[str]] = {}
    probes: list[dict[str, str]] = []
    for speaker in known:
        paths = sorted(grouped[speaker])
        if len(paths) < 2:
            raise EvaluationError(f"Known SID speaker needs enrollment and probe: {speaker}")
        local = paths.copy()
        speaker_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{speaker}".encode()).digest()[:8], "big"
        )
        random.Random(speaker_seed).shuffle(local)
        count = min(max_enrollment, len(local) - 1)
        selected = sorted(local[:count])
        enrollment[speaker] = selected
        selected_set = set(selected)
        probes.extend(
            {"path": path, "speaker_id": speaker, "status": "known"}
            for path in paths
            if path not in selected_set
        )
    for speaker in unknown:
        probes.extend(
            {"path": path, "speaker_id": speaker, "status": "unknown"}
            for path in sorted(grouped[speaker])
        )
    return {
        "protocol": "custom_open_set_sid_v1",
        "seed": seed,
        "known_ratio": known_ratio,
        "max_enrollment": max_enrollment,
        "known_speakers": known,
        "unknown_speakers": unknown,
        "enrollment": enrollment,
        "probes": sorted(probes, key=lambda item: (item["status"], item["speaker_id"], item["path"])),
    }


def build_sid_prototypes(
    protocol: Mapping[str, Any], embeddings: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    prototypes: dict[str, np.ndarray] = {}
    for speaker, paths in protocol["enrollment"].items():
        try:
            prototype = np.stack([embeddings[path] for path in paths]).mean(axis=0)
        except KeyError as exc:
            raise EvaluationError("SID enrollment references a missing embedding.") from exc
        norm = float(np.linalg.norm(prototype))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise EvaluationError("SID prototype has zero or invalid norm.")
        prototypes[speaker] = np.asarray(prototype / norm, dtype=np.float32)
    return prototypes


def score_sid(
    protocol: Mapping[str, Any],
    embeddings: Mapping[str, np.ndarray],
    prototypes: Mapping[str, np.ndarray],
) -> list[SIDProbeScore]:
    speakers = sorted(prototypes)
    matrix = np.stack([prototypes[speaker] for speaker in speakers])
    scores: list[SIDProbeScore] = []
    for probe in protocol["probes"]:
        if probe["path"] not in embeddings:
            raise EvaluationError("SID probe references a missing embedding.")
        similarities = matrix @ embeddings[probe["path"]]
        best = int(np.argmax(similarities))
        score = float(similarities[best])
        if not math.isfinite(score):
            raise EvaluationError("SID produced a non-finite score.")
        scores.append(
            SIDProbeScore(
                probe["path"], probe["speaker_id"], probe["status"], speakers[best], score
            )
        )
    return scores


def sid_metrics(scores: Sequence[SIDProbeScore], threshold: float) -> dict[str, Any]:
    known = [score for score in scores if score.status == "known"]
    unknown = [score for score in scores if score.status == "unknown"]
    if not known or not unknown:
        raise EvaluationError("SID metrics require known and unknown probes.")
    top1 = sum(score.best_speaker == score.true_speaker for score in known)
    accepted_correct = sum(
        score.best_score >= threshold and score.best_speaker == score.true_speaker
        for score in known
    )
    wrong_accept = sum(
        score.best_score >= threshold and score.best_speaker != score.true_speaker
        for score in known
    )
    known_reject = sum(score.best_score < threshold for score in known)
    unknown_reject = sum(score.best_score < threshold for score in unknown)
    return {
        "threshold": float(threshold),
        "known_probe_count": len(known),
        "unknown_probe_count": len(unknown),
        "known_top1_identity_accuracy": float(top1 / len(known)),
        "known_accepted_correct_rate": float(accepted_correct / len(known)),
        "known_wrong_accept_rate": float(wrong_accept / len(known)),
        "known_rejection_rate": float(known_reject / len(known)),
        "unknown_false_accept_rate": float((len(unknown) - unknown_reject) / len(unknown)),
        "unknown_rejection_rate": float(unknown_reject / len(unknown)),
    }


def calibrate_sid(scores: Sequence[SIDProbeScore], target_unknown_far: float) -> dict[str, Any]:
    if not 0 <= target_unknown_far <= 1:
        raise EvaluationError("SID target unknown FAR must be between 0 and 1.")
    candidates = [
        sid_metrics(scores, threshold)
        for threshold in _thresholds([score.best_score for score in scores])
    ]
    feasible = [
        candidate
        for candidate in candidates
        if candidate["unknown_false_accept_rate"] <= target_unknown_far
    ]
    selected = max(
        feasible,
        key=lambda item: (item["known_accepted_correct_rate"], item["threshold"]),
    )
    return {
        "schema_version": 1,
        "source": "validation",
        "policy": "target_unknown_far_empirical_v1",
        "target_unknown_far": float(target_unknown_far),
        **selected,
    }


def evaluate_validation(
    records: Sequence[AudioRecord],
    embeddings: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    seed = int(config["seed"])
    training = config["training"]
    calibration_config = config["calibration"]
    sv_trials = build_sv_trials(
        records,
        seed=seed,
        max_positive_per_speaker=int(training["max_positive_trials_per_speaker"]),
    )
    sv_scores = score_sv_trials(sv_trials, embeddings)
    sv_metric = compute_sv_metrics(sv_scores)
    sv_calibration = calibrate_sv(sv_scores, float(calibration_config["sv_target_far"]))
    sid_protocol = build_sid_protocol(
        records,
        seed=seed,
        known_ratio=float(calibration_config["sid_known_ratio"]),
        max_enrollment=int(calibration_config["sid_max_enrollment"]),
    )
    sid_scores = score_sid(
        sid_protocol, embeddings, build_sid_prototypes(sid_protocol, embeddings)
    )
    sid_calibration = calibrate_sid(
        sid_scores, float(calibration_config["sid_target_unknown_far"])
    )
    sid_metric = sid_metrics(sid_scores, float(sid_calibration["threshold"]))
    provenance = {
        "checkpoint_sha256": checkpoint_sha256,
        "validation_manifest_fingerprint": records_fingerprint(records),
        "seed": seed,
    }
    sv_calibration.update(provenance)
    sid_calibration.update(provenance)
    output = Path(output_dir).expanduser().resolve()
    save_json(output / "calibration" / "sv_calibration.json", sv_calibration)
    save_json(output / "calibration" / "sid_calibration.json", sid_calibration)
    metrics = {
        "source": "speaker_disjoint_validation_from_voxvietnam_t",
        "sv": {**sv_metric, "deployment": sv_calibration},
        "sid": {**sid_metric, "protocol": "custom_open_set_sid_v1"},
    }
    save_json(output / "metrics" / "validation_metrics.json", metrics)
    return metrics


def load_frozen_calibrations(
    output_dir: str | Path, *, checkpoint_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(output_dir).expanduser().resolve() / "calibration"
    try:
        sv = json.loads((root / "sv_calibration.json").read_text(encoding="utf-8"))
        sid = json.loads((root / "sid_calibration.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("Frozen validation calibration artifacts are missing or invalid.") from exc
    if (
        sv.get("source") != "validation"
        or sv.get("policy") != "target_far_empirical_v1"
        or sid.get("source") != "validation"
        or sid.get("policy") != "target_unknown_far_empirical_v1"
        or sv.get("checkpoint_sha256") != checkpoint_sha256
        or sid.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise EvaluationError("Frozen calibration is incompatible with best.pt.")
    return sv, sid


def write_sv_scores(path: str | Path, scores: Sequence[SVScore]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path_a", "path_b", "label", "score"))
        writer.writeheader()
        writer.writerows(asdict(score) for score in scores)
    return output
