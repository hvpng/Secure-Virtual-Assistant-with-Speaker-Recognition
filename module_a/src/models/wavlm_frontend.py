"""Lazy Hugging Face WavLM wrapper plus an offline deterministic test double."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class WavLMFrontendError(RuntimeError):
    """Raised when WavLM cannot satisfy the frozen frontend contract."""


class HuggingFaceWavLMFrontend(nn.Module):
    """Thin wrapper returning the final WavLM hidden state as [B, frames, hidden]."""

    def __init__(self, model: nn.Module, *, expected_hidden_dimension: int = 768) -> None:
        super().__init__()
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size != expected_hidden_dimension:
            raise WavLMFrontendError(
                f"WavLM hidden dimension mismatch: expected {expected_hidden_dimension}, "
                f"received {hidden_size}."
            )
        self.model = model
        self.output_dimension = expected_hidden_dimension

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        expected_hidden_dimension: int = 768,
        local_files_only: bool = False,
    ) -> "HuggingFaceWavLMFrontend":
        """Load on explicit invocation only; importing Module A never downloads weights."""

        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise WavLMFrontendError(f"Cannot load Hugging Face WavLM model: {model_name}") from exc
        return cls(model, expected_hidden_dimension=expected_hidden_dimension)

    def freeze(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def forward(self, waveforms: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        outputs = self.model(
            input_values=waveforms,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise WavLMFrontendError("WavLM did not return a rank-3 last_hidden_state.")
        if hidden.shape[-1] != self.output_dimension:
            raise WavLMFrontendError(
                f"WavLM output dimension changed: expected {self.output_dimension}, "
                f"received {hidden.shape[-1]}."
            )
        return hidden


class DeterministicFakeWavLM(nn.Module):
    """Offline A2 test double; never selected implicitly by the model factory."""

    def __init__(self, hidden_dimension: int = 768, frame_count: int = 12) -> None:
        super().__init__()
        if hidden_dimension <= 0 or frame_count <= 0:
            raise ValueError("Fake WavLM dimensions must be positive.")
        self.output_dimension = hidden_dimension
        self.frame_count = frame_count
        self.projection = nn.Linear(1, hidden_dimension)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.linspace(-0.5, 0.5, hidden_dimension).unsqueeze(1)
            )
            self.projection.bias.copy_(torch.linspace(-0.1, 0.1, hidden_dimension))

    def freeze(self) -> None:
        self.requires_grad_(False)
        self.eval()

    def forward(self, waveforms: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if attention_mask is not None and attention_mask.shape != waveforms.shape:
            raise WavLMFrontendError("attention_mask must match waveform shape.")
        pooled = F.adaptive_avg_pool1d(waveforms.unsqueeze(1), self.frame_count)
        return self.projection(pooled.transpose(1, 2))

