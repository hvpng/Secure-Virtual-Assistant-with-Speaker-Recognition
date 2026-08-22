"""Numerically guarded Additive Angular Margin Softmax training head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AAMSoftmaxError(ValueError):
    """Raised when embeddings or class labels violate the AAM contract."""


class AAMSoftmax(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        num_classes: int,
        *,
        margin: float = 0.2,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        if embedding_dimension <= 0 or num_classes <= 1:
            raise AAMSoftmaxError("AAM requires a positive embedding dimension and >=2 classes.")
        if not 0 < margin < math.pi / 2 or scale <= 0:
            raise AAMSoftmaxError("AAM margin and scale are outside the supported range.")
        self.embedding_dimension = embedding_dimension
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dimension))
        nn.init.xavier_uniform_(self.weight)
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.monotonic_threshold = math.cos(math.pi - margin)
        self.margin_correction = math.sin(math.pi - margin) * margin

    def _validate_inputs(self, embeddings: Tensor, labels: Tensor) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dimension:
            raise AAMSoftmaxError(
                "AAM embeddings must have shape [batch, embedding_dimension]."
            )
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise AAMSoftmaxError("AAM labels must have shape [batch].")
        if labels.dtype not in (torch.int32, torch.int64):
            raise AAMSoftmaxError("AAM labels must be integer class indices.")
        if labels.numel() == 0 or labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise AAMSoftmaxError(
                f"AAM labels must be within [0, {self.num_classes - 1}]."
            )
        if not torch.isfinite(embeddings).all():
            raise AAMSoftmaxError("AAM embeddings contain NaN or infinity.")

    def _compute_tensors(
        self, embeddings: Tensor, labels: Tensor
    ) -> dict[str, Tensor]:
        """Compute the exact forward intermediates without changing loss semantics."""

        self._validate_inputs(embeddings, labels)

        normalized_embeddings = F.normalize(embeddings, dim=1, eps=1e-6)
        normalized_weights = F.normalize(self.weight, dim=1, eps=1e-6)
        cosine = F.linear(normalized_embeddings, normalized_weights).clamp(
            -1.0 + 1e-6, 1.0 - 1e-6
        )
        sine_squared = 1.0 - cosine.square()
        sine = torch.sqrt(sine_squared.clamp_min(1e-6))
        margin_cosine = cosine * self.cos_margin - sine * self.sin_margin
        margin_cosine = torch.where(
            cosine > self.monotonic_threshold,
            margin_cosine,
            cosine - self.margin_correction,
        )
        one_hot = F.one_hot(labels.to(torch.int64), num_classes=self.num_classes).to(
            cosine.dtype
        )
        logits = self.scale * (one_hot * margin_cosine + (1.0 - one_hot) * cosine)
        if not torch.isfinite(logits).all():
            raise AAMSoftmaxError("AAM produced non-finite logits.")
        return {
            "normalized_embedding": normalized_embeddings,
            "normalized_weight": normalized_weights,
            "cosine": cosine,
            "sine_squared_before_clamp": sine_squared,
            "sine": sine,
            "margin_cosine": margin_cosine,
            "logits": logits,
        }

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        return self._compute_tensors(embeddings, labels)["logits"]

    @torch.no_grad()
    def numerical_diagnostics(
        self, embeddings: Tensor, labels: Tensor
    ) -> dict[str, Tensor]:
        """Return detached first-step intermediates for opt-in A3 diagnostics."""

        return {
            name: tensor.detach()
            for name, tensor in self._compute_tensors(embeddings, labels).items()
        }

    def loss(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        loss = F.cross_entropy(self(embeddings, labels), labels.to(torch.int64))
        if not torch.isfinite(loss):
            raise AAMSoftmaxError("AAM loss is not finite.")
        return loss
