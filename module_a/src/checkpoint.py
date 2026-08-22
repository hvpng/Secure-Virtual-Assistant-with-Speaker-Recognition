"""Atomic A2/A3 checkpoint save/load contract."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer


CHECKPOINT_FIELDS = {
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "step",
    "config",
    "num_classes",
    "speaker_to_index",
    "scheduler_state_dict",
    "scaler_state_dict",
    "training_metadata",
}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete or incompatible."""


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    config: Mapping[str, Any],
    num_classes: int,
    speaker_to_index: Mapping[str, int],
    scheduler: Any | None = None,
    scaler: Any | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> Path:
    if epoch < 0 or step < 0 or num_classes <= 1:
        raise CheckpointError("Checkpoint epoch/step/classes are invalid.")
    if len(speaker_to_index) != num_classes or set(speaker_to_index.values()) != set(
        range(num_classes)
    ):
        raise CheckpointError("speaker_to_index must map exactly onto all class indices.")
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": dict(config),
        "num_classes": num_classes,
        "speaker_to_index": dict(speaker_to_index),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "training_metadata": dict(training_metadata or {}),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=checkpoint_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    expected_num_classes: int | None = None,
    expected_speaker_to_index: Mapping[str, int] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise CheckpointError(f"Cannot load checkpoint: {checkpoint_path}") from exc
    if not isinstance(payload, dict) or not CHECKPOINT_FIELDS.issubset(payload):
        raise CheckpointError("Checkpoint is missing required fields.")
    if expected_num_classes is not None and payload["num_classes"] != expected_num_classes:
        raise CheckpointError("Checkpoint num_classes is incompatible with this run.")
    if expected_speaker_to_index is not None and payload["speaker_to_index"] != dict(
        expected_speaker_to_index
    ):
        raise CheckpointError("Checkpoint speaker_to_index is incompatible with this run.")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None:
            state = payload.get("scheduler_state_dict")
            if state is None:
                raise CheckpointError("Checkpoint has no scheduler state to resume.")
            scheduler.load_state_dict(state)
        if scaler is not None:
            state = payload.get("scaler_state_dict")
            if state is None:
                raise CheckpointError("Checkpoint has no AMP scaler state to resume.")
            scaler.load_state_dict(state)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError("Checkpoint state is incompatible with the model.") from exc
    return payload
