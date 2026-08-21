"""Training-side WavLM -> adapter -> CAM++ speaker model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from module_a.src.models.aam_softmax import AAMSoftmax
from module_a.src.models.campp import CAMPlusPlus


MIN_WAVEFORM_SAMPLES = 400


class SpeakerModelError(ValueError):
    """Raised when waveform, feature, or embedding contracts are violated."""


@dataclass(frozen=True)
class TrainingOutput:
    raw_embedding: Tensor
    logits: Tensor
    loss: Tensor


class WavLMCAMPlusPlus(nn.Module):
    """Frozen WavLM frontend with a trainable 768->80 adapter and CAM++ encoder."""

    def __init__(
        self,
        frontend: nn.Module,
        *,
        wavlm_hidden_dimension: int,
        adapter_dimension: int,
        campp: CAMPlusPlus,
        wavlm_frozen: bool = True,
    ) -> None:
        super().__init__()
        if not wavlm_frozen:
            raise SpeakerModelError("A2 only supports Stage-1 frozen WavLM.")
        frontend_dimension = getattr(frontend, "output_dimension", None)
        if frontend_dimension != wavlm_hidden_dimension:
            raise SpeakerModelError(
                f"WavLM frontend dimension mismatch: expected {wavlm_hidden_dimension}, "
                f"received {frontend_dimension}."
            )
        if adapter_dimension != 80 or campp.feature_dim != adapter_dimension:
            raise SpeakerModelError("Canonical adapter/CAM++ feature dimension must be 80.")
        self.frontend = frontend
        self.wavlm_hidden_dimension = wavlm_hidden_dimension
        self.adapter_dimension = adapter_dimension
        self.wavlm_frozen = True
        self.layer_norm = nn.LayerNorm(wavlm_hidden_dimension)
        self.projection = nn.Linear(wavlm_hidden_dimension, adapter_dimension)
        self.campp = campp
        freeze = getattr(self.frontend, "freeze", None)
        if callable(freeze):
            freeze()
        else:
            self.frontend.requires_grad_(False)
            self.frontend.eval()
        if any(parameter.requires_grad for parameter in self.frontend.parameters()):
            raise SpeakerModelError("Stage-1 WavLM parameters must all be frozen.")

    @property
    def embedding_dimension(self) -> int:
        return self.campp.embedding_dimension

    def train(self, mode: bool = True) -> "WavLMCAMPlusPlus":
        super().train(mode)
        if self.wavlm_frozen:
            self.frontend.eval()
        return self

    def _validate_waveforms(
        self, waveforms: Tensor, attention_mask: Tensor | None
    ) -> None:
        if waveforms.ndim != 2:
            raise SpeakerModelError("Waveforms must have shape [batch, num_samples].")
        if waveforms.shape[0] < 1 or waveforms.shape[1] < MIN_WAVEFORM_SAMPLES:
            raise SpeakerModelError(
                f"Waveforms require at least {MIN_WAVEFORM_SAMPLES} samples per item."
            )
        if not waveforms.is_floating_point():
            raise SpeakerModelError("Waveforms must use a floating-point dtype.")
        if not torch.isfinite(waveforms).all():
            raise SpeakerModelError("Waveforms contain NaN or infinity.")
        if attention_mask is not None:
            if attention_mask.shape != waveforms.shape:
                raise SpeakerModelError("attention_mask must match waveform shape.")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise SpeakerModelError("attention_mask values must be 0 or 1.")

    def extract_features(
        self, waveforms: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        self._validate_waveforms(waveforms, attention_mask)
        if self.wavlm_frozen:
            with torch.no_grad():
                hidden = self.frontend(waveforms, attention_mask=attention_mask)
        else:  # Reserved for a future Stage 2, unreachable in A2 construction.
            hidden = self.frontend(waveforms, attention_mask=attention_mask)
        if hidden.ndim != 3 or hidden.shape[-1] != self.wavlm_hidden_dimension:
            raise SpeakerModelError(
                "WavLM hidden state must have shape [batch, frames, wavlm_hidden_dimension]."
            )
        if hidden.shape[0] != waveforms.shape[0] or hidden.shape[1] < 1:
            raise SpeakerModelError("WavLM returned an invalid batch or frame dimension.")
        if not torch.isfinite(hidden).all():
            raise SpeakerModelError("WavLM returned NaN or infinity.")
        adapted = self.projection(self.layer_norm(hidden))
        if adapted.shape != (*hidden.shape[:2], self.adapter_dimension):
            raise SpeakerModelError("Adapter did not produce [batch, frames, 80].")
        return hidden, adapted

    def forward(self, waveforms: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _, adapted = self.extract_features(waveforms, attention_mask)
        # CAM++ consumes explicit [B, feature_dim=80, frames] orientation.
        raw_embedding = self.campp(adapted.transpose(1, 2).contiguous())
        if raw_embedding.shape != (waveforms.shape[0], self.embedding_dimension):
            raise SpeakerModelError("Speaker embedding has an unexpected shape.")
        if not torch.isfinite(raw_embedding).all():
            raise SpeakerModelError("Speaker embedding contains NaN or infinity.")
        return raw_embedding

    def extract_embedding(
        self, waveforms: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        raw_embedding = self(waveforms, attention_mask)
        norms = raw_embedding.norm(dim=1)
        if torch.any(norms <= 1e-8) or not torch.isfinite(norms).all():
            raise SpeakerModelError("Cannot normalize a zero or non-finite embedding.")
        normalized = F.normalize(raw_embedding, dim=1, eps=1e-8)
        if not torch.isfinite(normalized).all():
            raise SpeakerModelError("Normalized embedding contains NaN or infinity.")
        return normalized


class SpeakerTrainingModel(nn.Module):
    """Bundle the embedding encoder and class-count-dependent AAM training head."""

    def __init__(self, encoder: WavLMCAMPlusPlus, aam_head: AAMSoftmax) -> None:
        super().__init__()
        if encoder.embedding_dimension != aam_head.embedding_dimension:
            raise SpeakerModelError("Encoder and AAM embedding dimensions must match.")
        self.encoder = encoder
        self.aam_head = aam_head

    def forward(
        self,
        waveforms: Tensor,
        labels: Tensor,
        attention_mask: Tensor | None = None,
    ) -> TrainingOutput:
        raw_embedding = self.encoder(waveforms, attention_mask)
        logits = self.aam_head(raw_embedding, labels)
        loss = F.cross_entropy(logits, labels.to(torch.int64))
        if not torch.isfinite(loss):
            raise SpeakerModelError("Training loss is not finite.")
        return TrainingOutput(raw_embedding=raw_embedding, logits=logits, loss=loss)

    def extract_embedding(
        self, waveforms: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        return self.encoder.extract_embedding(waveforms, attention_mask)

