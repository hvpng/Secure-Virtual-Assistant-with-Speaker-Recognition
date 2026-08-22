"""Compact ECAPA-TDNN encoder and AAM-Softmax training head."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from module_a.src.features import LogMelFbank


class ModelError(RuntimeError):
    """Raised for invalid ECAPA inputs or non-finite outputs."""


class TDNNBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, *, dilation: int = 1
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.activation(self.conv(value)))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = self.layers(value.mean(dim=2, keepdim=True))
        return value * gate


class SERes2Block(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        scale: int,
        dilation: int,
        se_channels: int,
    ) -> None:
        super().__init__()
        if scale < 2 or channels % scale:
            raise ModelError("ECAPA channels must be divisible by scale >= 2.")
        width = channels // scale
        self.scale = scale
        self.pre = TDNNBlock(channels, channels, 1)
        self.res2 = nn.ModuleList(
            [TDNNBlock(width, width, 3, dilation=dilation) for _ in range(scale - 1)]
        )
        self.post = TDNNBlock(channels, channels, 1)
        self.se = SqueezeExcitation(channels, se_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        splits = self.pre(value).chunk(self.scale, dim=1)
        outputs = [splits[0]]
        for index, block in enumerate(self.res2, start=1):
            current = splits[index] if index == 1 else splits[index] + outputs[-1]
            outputs.append(block(current))
        return self.se(self.post(torch.cat(outputs, dim=1))) + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[2] == 0:
            raise ModelError("Statistics pooling expects [batch, channels, frames].")
        mean = value.mean(dim=2, keepdim=True)
        variance = (value - mean).square().mean(dim=2, keepdim=True)
        std = variance.clamp_min(1e-5).sqrt()
        context = torch.cat(
            [value, mean.expand_as(value), std.expand_as(value)], dim=1
        )
        weights = torch.softmax(self.attention(context), dim=2)
        weighted_mean = (weights * value).sum(dim=2)
        weighted_second = (weights * value.square()).sum(dim=2)
        weighted_std = (weighted_second - weighted_mean.square()).clamp_min(1e-5).sqrt()
        pooled = torch.cat([weighted_mean, weighted_std], dim=1)
        if not torch.isfinite(pooled).all():
            raise ModelError("Attentive statistics pooling produced non-finite values.")
        return pooled


class ECAPATDNN(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 80,
        channels: int = 512,
        scale: int = 8,
        se_channels: int = 128,
        embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input = TDNNBlock(input_dim, channels, 5)
        self.blocks = nn.ModuleList(
            [
                SERes2Block(
                    channels, scale=scale, dilation=dilation, se_channels=se_channels
                )
                for dilation in (2, 3, 4)
            ]
        )
        aggregate_channels = channels * len(self.blocks)
        self.aggregate = TDNNBlock(aggregate_channels, aggregate_channels, 1)
        self.pool = AttentiveStatisticsPooling(aggregate_channels)
        self.pool_norm = nn.BatchNorm1d(aggregate_channels * 2)
        self.embedding = nn.Linear(aggregate_channels * 2, embedding_dim)
        self.embedding_norm = nn.BatchNorm1d(embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != self.input.conv.in_channels:
            raise ModelError("ECAPA features must have shape [batch, 80, frames].")
        if not torch.isfinite(features).all():
            raise ModelError("ECAPA features contain NaN or Inf.")
        current = self.input(features)
        outputs: list[torch.Tensor] = []
        for block in self.blocks:
            current = block(current)
            outputs.append(current)
        aggregate = self.aggregate(torch.cat(outputs, dim=1))
        pooled = self.pool_norm(self.pool(aggregate))
        embedding = self.embedding_norm(self.embedding(pooled))
        if embedding.shape[1] != self.embedding_dim or not torch.isfinite(embedding).all():
            raise ModelError("ECAPA produced an invalid embedding.")
        return embedding


class SpeakerEmbeddingModel(nn.Module):
    """The single encoder shared by SV, SID, evaluation, and deployment."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        audio = config["audio"]
        features = config["features"]
        model = config["model"]
        self.config = {
            "audio": dict(audio),
            "features": dict(features),
            "model": dict(model),
        }
        self.features = LogMelFbank(
            sample_rate=int(audio["sample_rate"]),
            n_mels=int(features["fbank_dim"]),
            frame_length_ms=float(features["frame_length_ms"]),
            frame_shift_ms=float(features["frame_shift_ms"]),
            n_fft=int(features["n_fft"]),
        )
        self.encoder = ECAPATDNN(
            input_dim=int(features["fbank_dim"]),
            channels=int(model["channels"]),
            scale=int(model["scale"]),
            se_channels=int(model["se_channels"]),
            embedding_dim=int(model["embedding_dim"]),
        )

    def forward(self, waveforms: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
        embedding = self.encoder(self.features(waveforms))
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)
        if not torch.isfinite(embedding).all():
            raise ModelError("Speaker model produced non-finite embeddings.")
        return embedding

    def extract_embedding(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.forward(waveforms, normalize=True)


class AAMSoftmax(nn.Module):
    def __init__(
        self, embedding_dim: int, num_classes: int, *, margin: float = 0.2, scale: float = 30.0
    ) -> None:
        super().__init__()
        if num_classes < 2 or not 0 <= margin < math.pi / 2 or scale <= 0:
            raise ModelError("Invalid AAM-Softmax configuration.")
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.margin_correction = math.sin(math.pi - margin) * margin

    def logits(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
            raise ModelError("AAM inputs must be [batch, embedding] and [batch].")
        cosine = F.linear(
            F.normalize(embeddings, p=2, dim=1, eps=1e-12),
            F.normalize(self.weight, p=2, dim=1, eps=1e-12),
        ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = (1.0 - cosine.square()).clamp_min(1e-7).sqrt()
        margin_cosine = cosine * self.cos_m - sine * self.sin_m
        margin_cosine = torch.where(
            cosine > self.threshold,
            margin_cosine,
            cosine - self.margin_correction,
        )
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        logits = (one_hot * margin_cosine + (1.0 - one_hot) * cosine) * self.scale
        if not torch.isfinite(logits).all():
            raise ModelError("AAM-Softmax produced non-finite logits.")
        return logits

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(embeddings, labels), labels)


def build_embedding_model(config: Mapping[str, Any]) -> SpeakerEmbeddingModel:
    return SpeakerEmbeddingModel(config)

