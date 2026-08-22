"""Phase-1 ECAPA training, validation-EER selection, and checkpoint utilities."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from module_a.src.config import save_json
from module_a.src.data import (
    AudioAugmenter,
    AudioRecord,
    SpeakerBalancedBatchSampler,
    VoxVietnamDataset,
    build_speaker_to_index,
    seed_worker,
)
from module_a.src.ecapa import AAMSoftmax, SpeakerEmbeddingModel, build_embedding_model
from module_a.src.evaluation import (
    build_sv_trials,
    compute_sv_metrics,
    evaluate_validation,
    extract_embeddings,
    score_sv_trials,
    sha256_file,
)


class TrainingError(RuntimeError):
    """Raised when Phase 1 cannot continue safely."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is unavailable.")
    if normalized not in {"cpu", "cuda"}:
        raise TrainingError("Device must be auto, cpu, or cuda.")
    return torch.device(normalized)


def _atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)
    return output


def save_checkpoint(
    path: str | Path,
    *,
    model: SpeakerEmbeddingModel,
    classifier: AAMSoftmax,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
    speaker_to_index: Mapping[str, int],
    epoch: int,
    global_step: int,
    best_validation_eer: float,
) -> Path:
    return _atomic_torch_save(
        {
            "checkpoint_version": 1,
            "architecture": "ecapa_tdnn",
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": dict(config),
            "speaker_to_index": dict(speaker_to_index),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_validation_eer": float(best_validation_eer),
        },
        path,
    )


def load_checkpoint_model(
    path: str | Path, *, device: torch.device
) -> tuple[SpeakerEmbeddingModel, dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingError(f"Cannot load checkpoint: {checkpoint_path}") from exc
    required = {
        "architecture", "model_state_dict", "config", "speaker_to_index",
        "epoch", "global_step", "best_validation_eer",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise TrainingError("Checkpoint is malformed or incomplete.")
    if payload["architecture"] != "ecapa_tdnn":
        raise TrainingError("Checkpoint architecture is not ECAPA-TDNN.")
    config = payload["config"]
    if not isinstance(config, dict):
        raise TrainingError("Checkpoint configuration is malformed.")
    model = build_embedding_model(config)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except Exception as exc:
        raise TrainingError("Checkpoint state is incompatible with ECAPA-TDNN.") from exc
    model.to(device).eval()
    return model, config, payload


def _validation_eer(
    model: SpeakerEmbeddingModel,
    validation_records: Sequence[AudioRecord],
    dataset_root: str | Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> float:
    embeddings = extract_embeddings(
        model, validation_records, dataset_root, config, device=device
    )
    trials = build_sv_trials(
        validation_records,
        seed=int(config["seed"]),
        max_positive_per_speaker=int(
            config["training"]["max_positive_trials_per_speaker"]
        ),
    )
    scores = score_sv_trials(trials, embeddings)
    return float(compute_sv_metrics(scores)["eer"])


def train_phase1(
    train_records: Sequence[AudioRecord],
    validation_records: Sequence[AudioRecord],
    dataset_root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: torch.device,
    resume: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train ECAPA and calibrate using VoxVietnam-T validation speakers only."""

    started = time.perf_counter()
    seed = int(config["seed"])
    seed_everything(seed)
    output = Path(output_dir).expanduser().resolve()
    speaker_to_index = build_speaker_to_index(train_records)
    augmenter = AudioAugmenter(config["augmentation"], int(config["audio"]["sample_rate"]))
    train_dataset = VoxVietnamDataset(
        train_records,
        dataset_root,
        sample_rate=int(config["audio"]["sample_rate"]),
        segment_seconds=float(config["audio"]["segment_seconds"]),
        training=True,
        speaker_to_index=speaker_to_index,
        augmenter=augmenter,
    )
    sampler = SpeakerBalancedBatchSampler(
        train_records,
        speakers_per_batch=int(config["training"]["speakers_per_batch"]),
        utterances_per_speaker=int(config["training"]["utterances_per_speaker"]),
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=int(config["training"]["num_workers"]),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if len(loader) == 0:
        raise TrainingError("Training DataLoader is empty.")

    model = build_embedding_model(config).to(device)
    classifier = AAMSoftmax(
        int(config["model"]["embedding_dim"]),
        len(speaker_to_index),
        margin=float(config["loss"]["margin"]),
        scale=float(config["loss"]["scale"]),
    ).to(device)
    parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    accumulation = int(config["training"]["gradient_accumulation"])
    total_updates = max(
        1,
        math.ceil(len(loader) / accumulation) * int(config["training"]["epochs"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_eer = math.inf
    if resume is not None:
        try:
            payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
            if payload["speaker_to_index"] != speaker_to_index:
                raise TrainingError("Resume speaker mapping does not match current train manifest.")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            classifier.load_state_dict(payload["classifier_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            scaler.load_state_dict(payload.get("scaler_state_dict", {}))
            start_epoch = int(payload["epoch"]) + 1
            global_step = int(payload["global_step"])
            best_eer = float(payload["best_validation_eer"])
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError("Cannot resume the supplied checkpoint safely.") from exc

    history_path = output / "history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    final_loss: float | None = None
    stopped = False
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        sampler.set_epoch(epoch)
        model.train()
        classifier.train()
        for batch_index, batch in enumerate(loader):
            waveforms = batch["waveform"].to(device)
            labels = batch["class_index"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                embeddings = model(waveforms, normalize=False)
                loss = classifier(embeddings, labels)
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise TrainingError("Non-finite training loss detected.")
            scaler.scale(scaled_loss).backward()
            update_boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if not update_boundary:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled and scaler.get_scale() < old_scale:
                continue
            scheduler.step()
            global_step += 1
            final_loss = float(loss.detach().cpu())
            event = {
                "type": "train",
                "epoch": epoch,
                "global_step": global_step,
                "loss": final_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            if max_steps is not None and global_step >= max_steps:
                stopped = True
                break

        should_validate = (
            (epoch + 1) % int(config["training"]["validate_every_epochs"]) == 0
            or stopped
            or epoch + 1 == int(config["training"]["epochs"])
        )
        if should_validate:
            validation_eer = _validation_eer(
                model, validation_records, dataset_root, config, device
            )
            if validation_eer < best_eer:
                best_eer = validation_eer
                save_checkpoint(
                    output / "checkpoints" / "best.pt",
                    model=model,
                    classifier=classifier,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    speaker_to_index=speaker_to_index,
                    epoch=epoch,
                    global_step=global_step,
                    best_validation_eer=best_eer,
                )
        save_checkpoint(
            output / "checkpoints" / "last.pt",
            model=model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            speaker_to_index=speaker_to_index,
            epoch=epoch,
            global_step=global_step,
            best_validation_eer=best_eer,
        )
        if stopped:
            break

    best_path = output / "checkpoints" / "best.pt"
    if not best_path.is_file():
        raise TrainingError("Training ended without a best validation checkpoint.")
    best_model, best_config, _ = load_checkpoint_model(best_path, device=device)
    checkpoint_hash = sha256_file(best_path)
    validation_embeddings = extract_embeddings(
        best_model, validation_records, dataset_root, best_config, device=device
    )
    validation_metrics = evaluate_validation(
        validation_records,
        validation_embeddings,
        best_config,
        checkpoint_sha256=checkpoint_hash,
        output_dir=output,
    )
    summary = {
        "status": "completed",
        "phase": 1,
        "dataset": "VoxVietnam-T",
        "model": "ECAPA-TDNN",
        "device": str(device),
        "amp_enabled": amp_enabled,
        "train_speakers": len(speaker_to_index),
        "validation_speakers": len({record.speaker_id for record in validation_records}),
        "train_utterances": len(train_records),
        "validation_utterances": len(validation_records),
        "global_step": global_step,
        "final_train_loss": final_loss,
        "best_validation_eer": best_eer,
        "augmentation_requested": dict(config["augmentation"]),
        "augmentation_enabled": augmenter.enabled,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": checkpoint_hash,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_metrics": validation_metrics,
    }
    save_json(output / "training_summary.json", summary)
    save_json(output / "speaker_to_index.json", speaker_to_index)
    return summary

