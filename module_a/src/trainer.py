"""A3 mini-training loop with train-only monitoring and safe resume."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from module_a.src.checkpoint import load_checkpoint, save_checkpoint
from module_a.src.config import ModuleAConfig, config_to_dict
from module_a.src.device import autocast_context
from module_a.src.model_factory import build_optimizer, build_scheduler
from module_a.src.training_data import (
    SpeakerBalancedBatchSampler,
    SpeakerManifestDataset,
    TrainingRecord,
    WaveformBatchCollator,
    build_speaker_to_index,
    load_manifest,
    seed_dataloader_worker,
    select_train_records,
    split_train_monitor,
)


class TrainerError(RuntimeError):
    """Raised when A3 cannot train or resume without violating its contract."""


@dataclass(frozen=True)
class TrainingDataBundle:
    selected_speakers: tuple[str, ...]
    speaker_to_index: dict[str, int]
    fit_records: tuple[TrainingRecord, ...]
    monitor_records: tuple[TrainingRecord, ...]
    monitor_speakers: tuple[str, ...]
    fit_dataset: SpeakerManifestDataset
    monitor_dataset: SpeakerManifestDataset
    fit_sampler: SpeakerBalancedBatchSampler
    fit_loader: DataLoader
    monitor_loader: DataLoader


@dataclass(frozen=True)
class TrainingResult:
    status: str
    device: str
    epochs_completed: int
    global_step: int
    train_speakers: int
    classifier_num_classes: int
    fit_utterances: int
    monitor_utterances: int
    final_train_loss: float
    final_monitor_loss: float
    best_monitor_loss: float
    peak_cuda_memory_mb: float | None
    elapsed_seconds: float
    checkpoint_path: str
    resumed_from: str | None


def create_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    """Use the current torch.amp API; AMP is always disabled on CPU."""

    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_history(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")


def prepare_training_data(
    config: ModuleAConfig,
    *,
    train_manifest: str | Path,
    dataset_root: str | Path,
) -> TrainingDataBundle:
    """Prepare classifier classes and monitor data solely from the A1 train manifest."""

    all_records = load_manifest(train_manifest)
    selected, selected_speakers = select_train_records(
        all_records,
        max_speakers=config.training.max_train_speakers,
        seed=config.seed,
    )
    speaker_to_index = build_speaker_to_index(selected)
    holdout = split_train_monitor(
        selected,
        holdout_ratio=config.training.monitor_holdout_ratio,
        seed=config.seed,
        max_monitor_speakers=config.training.max_monitor_speakers,
    )
    fit_dataset = SpeakerManifestDataset(
        holdout.fit,
        dataset_root=dataset_root,
        speaker_to_index=speaker_to_index,
        sample_rate=config.audio.target_sample_rate,
    )
    monitor_dataset = SpeakerManifestDataset(
        holdout.monitor,
        dataset_root=dataset_root,
        speaker_to_index=speaker_to_index,
        sample_rate=config.audio.target_sample_rate,
    )
    sampler = SpeakerBalancedBatchSampler(
        holdout.fit,
        speakers_per_batch=config.training.speakers_per_batch,
        utterances_per_speaker=config.training.utterances_per_speaker,
        seed=config.seed,
    )
    generator = torch.Generator().manual_seed(config.seed)
    fit_loader = DataLoader(
        fit_dataset,
        batch_sampler=sampler,
        num_workers=config.training.num_workers,
        collate_fn=WaveformBatchCollator(
            sample_rate=config.audio.target_sample_rate,
            segment_seconds=config.audio.segment_seconds,
            training=True,
        ),
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
        persistent_workers=config.training.num_workers > 0,
    )
    monitor_loader = DataLoader(
        monitor_dataset,
        batch_size=(
            config.training.speakers_per_batch
            * config.training.utterances_per_speaker
        ),
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=WaveformBatchCollator(
            sample_rate=config.audio.target_sample_rate,
            segment_seconds=config.audio.segment_seconds,
            training=False,
        ),
        worker_init_fn=seed_dataloader_worker,
        generator=torch.Generator().manual_seed(config.seed + 1),
        persistent_workers=config.training.num_workers > 0,
    )
    if len(fit_loader) == 0 or len(monitor_loader) == 0:
        raise TrainerError("A3 fit and monitor DataLoaders must both be non-empty.")
    return TrainingDataBundle(
        selected_speakers=selected_speakers,
        speaker_to_index=speaker_to_index,
        fit_records=holdout.fit,
        monitor_records=holdout.monitor,
        monitor_speakers=holdout.monitor_speakers,
        fit_dataset=fit_dataset,
        monitor_dataset=monitor_dataset,
        fit_sampler=sampler,
        fit_loader=fit_loader,
        monitor_loader=monitor_loader,
    )


def _move_batch(batch: Mapping[str, object], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    try:
        waveforms = batch["waveforms"].to(device, non_blocking=True)  # type: ignore[union-attr]
        masks = batch["attention_mask"].to(device, non_blocking=True)  # type: ignore[union-attr]
        labels = batch["labels"].to(device, non_blocking=True)  # type: ignore[union-attr]
    except (AttributeError, KeyError) as exc:
        raise TrainerError("Training batch does not satisfy the tensor contract.") from exc
    return waveforms, masks, labels


def validate_monitor_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """Compute AAM loss only for held-out utterances of train classifier speakers."""

    was_training = model.training
    model.eval()
    loss_sum = 0.0
    item_count = 0
    with torch.no_grad():
        for batch in loader:
            waveforms, masks, labels = _move_batch(batch, device)
            with autocast_context(device, amp_enabled):
                output = model(waveforms, labels, attention_mask=masks)
            loss = float(output.loss.detach().cpu())
            if not math.isfinite(loss):
                raise TrainerError("Monitor loss is not finite.")
            count = int(labels.shape[0])
            loss_sum += loss * count
            item_count += count
    if was_training:
        model.train()
    if item_count == 0:
        raise TrainerError("Monitor DataLoader is empty.")
    return loss_sum / item_count


def _assert_finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise TrainerError(f"Non-finite gradient detected in: {name}")


def _checkpoint_metadata(
    *,
    next_epoch: int,
    next_batch: int,
    best_monitor_loss: float,
    final_train_loss: float,
    selected_speakers: tuple[str, ...],
    resume_signature: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "next_epoch": next_epoch,
        "next_batch": next_batch,
        "best_monitor_loss": best_monitor_loss,
        "final_train_loss": final_train_loss,
        "selected_speakers": list(selected_speakers),
        "resume_signature": dict(resume_signature),
    }


def _resume_signature(config: ModuleAConfig, total_steps: int) -> dict[str, Any]:
    serialized = config_to_dict(config)
    training = serialized["training"]
    return {
        "seed": config.seed,
        "model": serialized["model"],
        "audio": serialized["audio"],
        "loss": serialized["loss"],
        "scheduler": serialized["scheduler"],
        "training": {
            key: training[key]
            for key in (
                "epochs",
                "max_steps",
                "mixed_precision",
                "learning_rate",
                "weight_decay",
                "speakers_per_batch",
                "utterances_per_speaker",
                "num_workers",
                "gradient_accumulation_steps",
                "max_train_speakers",
                "max_monitor_speakers",
                "monitor_holdout_ratio",
            )
        },
        "total_steps": total_steps,
    }


def train_model(
    *,
    model: nn.Module,
    config: ModuleAConfig,
    data: TrainingDataBundle,
    device: torch.device,
    output_dir: str | Path,
    resume: str | Path | None = None,
    stop_after_steps: int | None = None,
) -> TrainingResult:
    """Run bounded A3 training and always save a resumable ``last.pt``."""

    output = Path(output_dir).expanduser().resolve()
    checkpoints = output / "checkpoints"
    history_path = output / "history.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    if resume is None:
        history_path.unlink(missing_ok=True)

    num_classes = len(data.speaker_to_index)
    if num_classes != len(data.selected_speakers) or num_classes < 2:
        raise TrainerError("Classifier mapping is empty or inconsistent.")
    if config.training.max_steps is None:
        total_steps = math.ceil(
            len(data.fit_loader)
            * config.training.epochs
            / config.training.gradient_accumulation_steps
        )
    else:
        total_steps = config.training.max_steps
    if total_steps <= 0:
        raise TrainerError("A3 total optimizer steps must be positive.")
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise TrainerError("stop_after_steps must be positive when provided.")
    invocation_stop_step = min(total_steps, stop_after_steps or total_steps)
    resume_signature = _resume_signature(config, total_steps)

    model.to(device)
    optimizer = build_optimizer(
        model,
        config.training.learning_rate,
        config.training.weight_decay,
    )
    frontend_ids = {id(parameter) for parameter in model.encoder.frontend.parameters()}  # type: ignore[attr-defined]
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if frontend_ids & optimizer_ids:
        raise TrainerError("Frozen WavLM parameters entered the optimizer.")
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=config.scheduler.type,
        total_steps=total_steps,
        warmup_steps=config.scheduler.warmup_steps,
    )
    scaler = create_grad_scaler(device, config.training.mixed_precision)
    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_monitor_loss = math.inf
    resumed_from: str | None = None
    if resume is not None:
        payload = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_num_classes=num_classes,
            expected_speaker_to_index=data.speaker_to_index,
            map_location=device,
        )
        metadata = payload.get("training_metadata") or {}
        if metadata.get("resume_signature") != resume_signature:
            raise TrainerError("Checkpoint training configuration is incompatible.")
        start_epoch = int(metadata.get("next_epoch", payload["epoch"]))
        start_batch = int(metadata.get("next_batch", 0))
        global_step = int(payload["step"])
        best_monitor_loss = float(metadata.get("best_monitor_loss", math.inf))
        resumed_train_loss = float(metadata.get("final_train_loss", math.nan))
        if metadata.get("selected_speakers") not in (
            None,
            list(data.selected_speakers),
        ):
            raise TrainerError("Checkpoint selected speaker subset is incompatible.")
        resumed_from = str(Path(resume).expanduser().resolve())
    else:
        resumed_train_loss = math.nan

    _atomic_json(output / "run_config.json", config_to_dict(config))
    _atomic_json(output / "speaker_to_index.json", data.speaker_to_index)
    _atomic_json(
        output / "train_monitor_split.json",
        {
            "strategy": "train-speaker-only deterministic utterance holdout",
            "seed": config.seed,
            "holdout_ratio": config.training.monitor_holdout_ratio,
            "selected_speakers": list(data.selected_speakers),
            "monitor_speakers": list(data.monitor_speakers),
            "fit_paths": [record.path for record in data.fit_records],
            "monitor_paths": [record.path for record in data.monitor_records],
        },
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_train_loss = resumed_train_loss
    final_monitor_loss = math.nan
    epochs_completed = start_epoch
    next_epoch = start_epoch
    next_batch = start_batch
    stopped = global_step >= invocation_stop_step
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, config.training.epochs):
        if stopped:
            break
        data.fit_sampler.set_epoch(epoch)
        model.train()
        accumulation_count = 0
        accumulation_loss_sum = 0.0
        for batch_index, batch in enumerate(data.fit_loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            waveforms, masks, labels = _move_batch(batch, device)
            with autocast_context(device, config.training.mixed_precision):
                output_batch = model(waveforms, labels, attention_mask=masks)
                loss = output_batch.loss
                scaled_loss = loss / config.training.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise TrainerError("Training loss is not finite.")
            scaler.scale(scaled_loss).backward()
            accumulation_count += 1
            accumulation_loss_sum += float(loss.detach().cpu())
            is_last_batch = batch_index + 1 == len(data.fit_loader)
            if (
                accumulation_count < config.training.gradient_accumulation_steps
                and not is_last_batch
            ):
                continue
            scaler.unscale_(optimizer)
            _assert_finite_gradients(model)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            final_train_loss = accumulation_loss_sum / accumulation_count
            accumulation_count = 0
            accumulation_loss_sum = 0.0
            global_step += 1
            next_epoch = epoch
            next_batch = batch_index + 1

            if global_step % config.training.log_every_steps == 0 or global_step == 1:
                event = {
                    "type": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": final_train_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                _append_history(history_path, event)
                print(json.dumps(event, sort_keys=True), flush=True)

            if (
                config.training.val_every_steps is not None
                and global_step % config.training.val_every_steps == 0
            ):
                final_monitor_loss = validate_monitor_loss(
                    model,
                    data.monitor_loader,
                    device=device,
                    amp_enabled=config.training.mixed_precision,
                )
                best_monitor_loss = min(best_monitor_loss, final_monitor_loss)
                _append_history(
                    history_path,
                    {
                        "type": "monitor",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": final_monitor_loss,
                    },
                )

            metadata = _checkpoint_metadata(
                next_epoch=next_epoch,
                next_batch=next_batch,
                best_monitor_loss=best_monitor_loss,
                final_train_loss=final_train_loss,
                selected_speakers=data.selected_speakers,
                resume_signature=resume_signature,
            )
            if (
                config.training.save_every_steps is not None
                and global_step % config.training.save_every_steps == 0
            ):
                save_checkpoint(
                    checkpoints / f"step_{global_step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=next_epoch,
                    step=global_step,
                    config=config_to_dict(config),
                    num_classes=num_classes,
                    speaker_to_index=data.speaker_to_index,
                    training_metadata=metadata,
                )
            if global_step >= invocation_stop_step:
                stopped = True
                break

        if not stopped:
            epochs_completed = epoch + 1
            next_epoch = epoch + 1
            next_batch = 0
        elif next_batch >= len(data.fit_loader):
            epochs_completed = epoch + 1
            next_epoch = epoch + 1
            next_batch = 0

    if math.isnan(final_train_loss) and global_step == 0:
        raise TrainerError("Training produced no optimizer steps.")
    final_monitor_loss = validate_monitor_loss(
        model,
        data.monitor_loader,
        device=device,
        amp_enabled=config.training.mixed_precision,
    )
    best_monitor_loss = min(best_monitor_loss, final_monitor_loss)
    _append_history(
        history_path,
        {
            "type": "monitor_final",
            "epoch": epochs_completed,
            "step": global_step,
            "loss": final_monitor_loss,
        },
    )
    metadata = _checkpoint_metadata(
        next_epoch=next_epoch,
        next_batch=next_batch,
        best_monitor_loss=best_monitor_loss,
        final_train_loss=final_train_loss,
        selected_speakers=data.selected_speakers,
        resume_signature=resume_signature,
    )
    last_checkpoint = save_checkpoint(
        checkpoints / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=next_epoch,
        step=global_step,
        config=config_to_dict(config),
        num_classes=num_classes,
        speaker_to_index=data.speaker_to_index,
        training_metadata=metadata,
    )
    elapsed = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    result = TrainingResult(
        status="ok",
        device=str(device),
        epochs_completed=epochs_completed,
        global_step=global_step,
        train_speakers=len(data.selected_speakers),
        classifier_num_classes=num_classes,
        fit_utterances=len(data.fit_records),
        monitor_utterances=len(data.monitor_records),
        final_train_loss=final_train_loss,
        final_monitor_loss=final_monitor_loss,
        best_monitor_loss=best_monitor_loss,
        peak_cuda_memory_mb=peak_memory,
        elapsed_seconds=elapsed,
        checkpoint_path=str(last_checkpoint),
        resumed_from=resumed_from,
    )
    _atomic_json(output / "training_summary.json", asdict(result))
    return result
