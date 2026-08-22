"""A4 checkpoint inference, split isolation, and deterministic embedding caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from module_a.src.audio_batch import (
    DETERMINISTIC_SEGMENT_POLICY_VERSION,
    load_waveform,
    prepare_deterministic_segment,
)
from module_a.src.checkpoint import CHECKPOINT_FIELDS
from module_a.src.config import ModuleAConfig, config_to_dict, load_config
from module_a.src.device import autocast_context
from module_a.src.model_factory import build_model
from module_a.src.training_data import TrainingRecord, load_manifest


class EvaluationError(RuntimeError):
    """Raised when A4 data, checkpoint, cache, or inference is unsafe."""


EMBEDDING_PREPROCESSING_VERSION = DETERMINISTIC_SEGMENT_POLICY_VERSION


@dataclass(frozen=True)
class EvaluationModelBundle:
    model: nn.Module
    config: ModuleAConfig
    checkpoint_path: Path
    checkpoint_sha256: str
    num_classes: int
    speaker_to_index: dict[str, int]
    device: torch.device


@dataclass(frozen=True)
class EmbeddingCache:
    embeddings: dict[str, np.ndarray]
    metadata: dict[str, Any]


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise EvaluationError(f"Required JSON artifact does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read JSON artifact: {source}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"JSON artifact must contain an object: {source}")
    return payload


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise EvaluationError(f"File does not exist for fingerprinting: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationError(f"Cannot fingerprint file: {source}") from exc
    return digest.hexdigest()


def fingerprint_records(records: Sequence[TrainingRecord]) -> str:
    canonical = [
        {"path": record.path, "speaker_id": record.speaker_id, "split": record.split}
        for record in sorted(records, key=lambda item: (item.path, item.speaker_id))
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def fingerprint_evaluation_config(config: ModuleAConfig) -> str:
    serialized = config_to_dict(config)
    relevant = {
        "model": serialized["model"],
        "audio": serialized["audio"],
        "loss": serialized["loss"],
        "evaluation_mixed_precision": serialized["evaluation"]["mixed_precision"],
        "embedding_preprocessing": EMBEDDING_PREPROCESSING_VERSION,
    }
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_split_manifest(path: str | Path, expected_split: str) -> list[TrainingRecord]:
    if expected_split not in {"val", "test"}:
        raise EvaluationError("A4 accepts only val or test manifests.")
    try:
        records = load_manifest(path)
    except Exception as exc:
        raise EvaluationError(f"Cannot load {expected_split} manifest: {path}") from exc
    if any(record.split != expected_split for record in records):
        raise EvaluationError(
            f"{expected_split} manifest contains records from another split."
        )
    paths = [record.path for record in records]
    if len(paths) != len(set(paths)):
        raise EvaluationError(f"{expected_split} manifest contains duplicate paths.")
    if len({record.speaker_id for record in records}) < 2:
        raise EvaluationError(f"{expected_split} evaluation requires at least two speakers.")
    return sorted(records, key=lambda item: (item.speaker_id, item.path))


def validate_evaluation_isolation(
    *,
    train_speakers: Sequence[str],
    validation_records: Sequence[TrainingRecord] | None = None,
    test_records: Sequence[TrainingRecord] | None = None,
) -> None:
    train = set(train_speakers)
    validation = {record.speaker_id for record in validation_records or ()}
    test = {record.speaker_id for record in test_records or ()}
    if train & validation:
        raise EvaluationError("Train speaker leaked into A4 validation evaluation.")
    if train & test:
        raise EvaluationError("Train speaker leaked into A4 test evaluation.")
    if validation & test:
        raise EvaluationError("Validation/test speaker contamination detected.")
    validation_paths = {record.path for record in validation_records or ()}
    test_paths = {record.path for record in test_records or ()}
    if validation_paths & test_paths:
        raise EvaluationError("Validation/test path contamination detected.")


def _replace_dataclass(instance: Any, values: Mapping[str, Any]) -> Any:
    expected = {field.name for field in fields(instance)}
    missing = expected - set(values)
    if missing:
        raise EvaluationError(f"Checkpoint config is missing fields: {sorted(missing)}")
    replacements = {name: values[name] for name in expected}
    if "campp_block_layers" in replacements:
        replacements["campp_block_layers"] = tuple(replacements["campp_block_layers"])
    return replace(instance, **replacements)


def _config_from_checkpoint(payload: Mapping[str, Any]) -> ModuleAConfig:
    saved = payload.get("config")
    if not isinstance(saved, Mapping):
        raise EvaluationError("Checkpoint config is missing or malformed.")
    try:
        base = load_config()
        model = _replace_dataclass(base.model, saved["model"])
        audio = _replace_dataclass(base.audio, saved["audio"])
        loss = _replace_dataclass(base.loss, saved["loss"])
        seed = int(saved["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("Checkpoint model/audio/loss config is malformed.") from exc
    config = replace(base, seed=seed, model=model, audio=audio, loss=loss)
    if not config.model.wavlm_frozen or config.model.stage2_enabled:
        raise EvaluationError("A4 requires the frozen Stage-1 WavLM checkpoint.")
    return config


def load_evaluation_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    local_files_only: bool = False,
    frontend: nn.Module | None = None,
) -> EvaluationModelBundle:
    """Reconstruct A3 exactly, load strictly, then expose eval-only embeddings."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise EvaluationError(f"Checkpoint does not exist: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise EvaluationError(f"Cannot load checkpoint: {checkpoint}") from exc
    if not isinstance(payload, dict) or not CHECKPOINT_FIELDS.issubset(payload):
        raise EvaluationError("Checkpoint is missing required A3 fields.")
    config = _config_from_checkpoint(payload)
    num_classes = payload.get("num_classes")
    speaker_to_index = payload.get("speaker_to_index")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 1
        or not isinstance(speaker_to_index, dict)
        or len(speaker_to_index) != num_classes
        or set(speaker_to_index.values()) != set(range(num_classes))
    ):
        raise EvaluationError("Checkpoint classifier metadata is malformed.")
    try:
        model = build_model(
            config,
            num_classes=num_classes,
            frontend=frontend,
            local_files_only=local_files_only,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise EvaluationError("Checkpoint state is incompatible with the A3 model.") from exc
    if any(parameter.requires_grad for parameter in model.encoder.frontend.parameters()):
        raise EvaluationError("A4 loaded a WavLM frontend that is not frozen.")
    return EvaluationModelBundle(
        model=model,
        config=config,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        num_classes=num_classes,
        speaker_to_index=dict(speaker_to_index),
        device=device,
    )


def validate_embedding_vector(vector: np.ndarray, embedding_dimension: int) -> np.ndarray:
    embedding = np.asarray(vector, dtype=np.float32)
    if embedding.shape != (embedding_dimension,) or not np.isfinite(embedding).all():
        raise EvaluationError("Embedding must be finite, 1-D, and fixed-dimensional.")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-4):
        raise EvaluationError("Evaluation embedding must already be L2-normalized.")
    return embedding


def extract_embeddings(
    bundle: EvaluationModelBundle,
    records: Sequence[TrainingRecord],
    *,
    dataset_root: str | Path,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract deterministic center-cropped embeddings; FP32 is the A4 default."""

    if not records:
        raise EvaluationError("Cannot extract embeddings for an empty manifest.")
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise EvaluationError(f"Dataset root does not exist: {root}")
    effective_batch_size = batch_size or bundle.config.evaluation.embedding_batch_size
    if effective_batch_size <= 0:
        raise EvaluationError("Embedding batch size must be positive.")
    ordered = sorted(records, key=lambda item: item.path)
    output: dict[str, np.ndarray] = {}
    bundle.model.eval()
    with torch.no_grad():
        for offset in range(0, len(ordered), effective_batch_size):
            batch_records = ordered[offset : offset + effective_batch_size]
            waveforms: list[Tensor] = []
            for record in batch_records:
                resolved = (root / Path(record.path)).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise EvaluationError("Manifest audio path escapes dataset root.") from exc
                try:
                    waveforms.append(
                        load_waveform(
                            resolved,
                            target_sample_rate=bundle.config.audio.target_sample_rate,
                        )
                    )
                except Exception as exc:
                    raise EvaluationError(f"Cannot load evaluation audio: {record.path}") from exc
            segments = [
                prepare_deterministic_segment(
                    waveform,
                    sample_rate=bundle.config.audio.target_sample_rate,
                    segment_seconds=bundle.config.audio.segment_seconds,
                )
                for waveform in waveforms
            ]
            waveform_batch = torch.stack(segments)
            attention_mask = torch.ones(
                waveform_batch.shape,
                dtype=torch.long,
            )
            waveform_batch = waveform_batch.to(bundle.device)
            attention_mask = attention_mask.to(bundle.device)
            with autocast_context(
                bundle.device, bundle.config.evaluation.mixed_precision
            ):
                embeddings = bundle.model.extract_embedding(
                    waveform_batch, attention_mask=attention_mask
                )
            embeddings = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
            if embeddings.shape != (
                len(batch_records),
                bundle.config.model.embedding_dimension,
            ):
                raise EvaluationError("Model returned an invalid embedding batch shape.")
            for record, vector in zip(batch_records, embeddings, strict=True):
                output[record.path] = validate_embedding_vector(
                    vector, bundle.config.model.embedding_dimension
                ).copy()
    if len(output) != len(records):
        raise EvaluationError("Embedding extraction lost or duplicated manifest paths.")
    return output


def _cache_metadata(
    *,
    bundle: EvaluationModelBundle,
    records: Sequence[TrainingRecord],
    split: str,
    dataset_root: str | Path,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "embedding_preprocessing": EMBEDDING_PREPROCESSING_VERSION,
        "split": split,
        "embedding_dimension": bundle.config.model.embedding_dimension,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "manifest_sha256": fingerprint_records(records),
        "config_sha256": fingerprint_evaluation_config(bundle.config),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "record_count": len(records),
    }


def write_embedding_cache(
    path: str | Path,
    embeddings: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered_paths = sorted(embeddings)
    dimension = int(metadata.get("embedding_dimension", 0))
    if not ordered_paths or dimension <= 0:
        raise EvaluationError("Embedding cache cannot be empty.")
    matrix = np.stack(
        [validate_embedding_vector(embeddings[path], dimension) for path in ordered_paths]
    ).astype(np.float32, copy=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                paths=np.asarray(ordered_paths, dtype=np.str_),
                embeddings=matrix,
                metadata_json=np.asarray(
                    json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")),
                    dtype=np.str_,
                ),
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_embedding_cache(
    path: str | Path, expected_metadata: Mapping[str, Any]
) -> EmbeddingCache:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise EvaluationError(f"Embedding cache does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as payload:
            paths = payload["paths"]
            matrix = payload["embeddings"]
            metadata = json.loads(str(payload["metadata_json"].item()))
    except Exception as exc:
        raise EvaluationError(f"Cannot read embedding cache: {source}") from exc
    if metadata != dict(expected_metadata):
        raise EvaluationError(
            "Embedding cache metadata is incompatible; use --recompute-embeddings."
        )
    if paths.ndim != 1 or matrix.ndim != 2 or len(paths) != len(matrix):
        raise EvaluationError("Embedding cache arrays are malformed.")
    dimension = int(metadata["embedding_dimension"])
    embeddings: dict[str, np.ndarray] = {}
    for raw_path, vector in zip(paths.tolist(), matrix, strict=True):
        normalized_path = str(raw_path)
        if normalized_path in embeddings:
            raise EvaluationError("Embedding cache contains duplicate paths.")
        embeddings[normalized_path] = validate_embedding_vector(vector, dimension).copy()
    if len(embeddings) != int(metadata["record_count"]):
        raise EvaluationError("Embedding cache record count is inconsistent.")
    return EmbeddingCache(embeddings=embeddings, metadata=dict(metadata))


def get_or_create_embedding_cache(
    path: str | Path,
    *,
    bundle: EvaluationModelBundle,
    records: Sequence[TrainingRecord],
    split: str,
    dataset_root: str | Path,
    recompute: bool = False,
    extractor: Callable[[], Mapping[str, np.ndarray]] | None = None,
) -> EmbeddingCache:
    metadata = _cache_metadata(
        bundle=bundle,
        records=records,
        split=split,
        dataset_root=dataset_root,
    )
    cache_path = Path(path).expanduser().resolve()
    if cache_path.exists() and not recompute:
        return load_embedding_cache(cache_path, metadata)
    produced = (
        extractor()
        if extractor is not None
        else extract_embeddings(bundle, records, dataset_root=dataset_root)
    )
    expected_paths = {record.path for record in records}
    if set(produced) != expected_paths:
        raise EvaluationError("Embedding extractor output does not match the manifest paths.")
    write_embedding_cache(cache_path, produced, metadata)
    return load_embedding_cache(cache_path, metadata)
