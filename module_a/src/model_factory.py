"""Single construction point for A2/A3 speaker models and optimizers."""

from __future__ import annotations

from torch import nn
import math

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from module_a.src.config import ModuleAConfig
from module_a.src.models.aam_softmax import AAMSoftmax
from module_a.src.models.campp import CAMPlusPlus
from module_a.src.models.speaker_model import SpeakerTrainingModel, WavLMCAMPlusPlus
from module_a.src.models.wavlm_frontend import HuggingFaceWavLMFrontend


def build_model(
    config: ModuleAConfig,
    *,
    num_classes: int,
    frontend: nn.Module | None = None,
    local_files_only: bool = False,
) -> SpeakerTrainingModel:
    """Build Tier-1 once; injected frontends keep default tests fully offline."""

    if frontend is None:
        frontend = HuggingFaceWavLMFrontend.from_pretrained(
            config.model.wavlm_model_name,
            expected_hidden_dimension=config.model.wavlm_hidden_dimension,
            local_files_only=local_files_only,
        )
    campp = CAMPlusPlus(
        feature_dim=config.model.adapter_dimension,
        embedding_dimension=config.model.embedding_dimension,
        growth_rate=config.model.campp_growth_rate,
        block_layers=config.model.campp_block_layers,
        init_channels=config.model.campp_init_channels,
        bottleneck_channels=config.model.campp_bottleneck_channels,
        fcm_channels=config.model.campp_fcm_channels,
        segment_frames=config.model.campp_segment_frames,
    )
    encoder = WavLMCAMPlusPlus(
        frontend,
        wavlm_hidden_dimension=config.model.wavlm_hidden_dimension,
        adapter_dimension=config.model.adapter_dimension,
        campp=campp,
        wavlm_frozen=config.model.wavlm_frozen,
    )
    aam_head = AAMSoftmax(
        config.model.embedding_dimension,
        num_classes,
        margin=config.loss.margin,
        scale=config.loss.scale,
    )
    return SpeakerTrainingModel(encoder, aam_head)


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> Optimizer:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Model has no trainable parameters.")
    return AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(
    optimizer: Optimizer,
    *,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int = 0,
) -> LRScheduler:
    """Build the A3 per-optimizer-step warmup/cosine schedule."""

    if scheduler_type != "cosine":
        raise ValueError("Only the A3 cosine scheduler is supported.")
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("Scheduler steps require 0 <= warmup_steps < total_steps.")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=multiplier)
