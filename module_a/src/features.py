"""Shared 80-bin log-Mel frontend used by training, evaluation, and runtime."""

from __future__ import annotations

import math

import torch
from torch import nn


class FeatureError(RuntimeError):
    """Raised when waveform-to-feature conversion violates its contract."""


def _hz_to_mel(value: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


def mel_filterbank(
    *, sample_rate: int, n_fft: int, n_mels: int, f_min: float = 20.0
) -> torch.Tensor:
    if sample_rate <= 0 or n_fft <= 0 or n_mels <= 0:
        raise FeatureError("Mel filterbank dimensions must be positive.")
    frequencies = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    minimum = _hz_to_mel(torch.tensor(float(f_min)))
    maximum = _hz_to_mel(torch.tensor(float(sample_rate / 2)))
    mel_points = torch.linspace(minimum, maximum, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    filters = torch.zeros(n_mels, frequencies.numel())
    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]
        rising = (frequencies - left) / max(float(center - left), 1e-12)
        falling = (right - frequencies) / max(float(right - center), 1e-12)
        filters[index] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    if not torch.isfinite(filters).all() or not torch.all(filters.sum(dim=1) > 0):
        raise FeatureError("Mel filterbank construction failed.")
    return filters


class LogMelFbank(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        n_fft: int = 512,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = round(sample_rate * frame_length_ms / 1000)
        self.hop_length = round(sample_rate * frame_shift_ms / 1000)
        self.n_fft = n_fft
        if self.win_length > n_fft or self.win_length <= 0 or self.hop_length <= 0:
            raise FeatureError("Invalid STFT window configuration.")
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer(
            "mel_filters",
            mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels),
            persistent=True,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2 or waveforms.shape[1] < self.win_length:
            raise FeatureError("Waveforms must have shape [batch, samples] and one frame.")
        if not torch.isfinite(waveforms).all():
            raise FeatureError("Waveforms contain NaN or Inf.")
        waveform = waveforms.to(torch.float32)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spectrum.abs().square()
        mel = torch.einsum("mf,bft->bmt", self.mel_filters, power)
        log_mel = torch.log(mel.clamp_min(1e-10))
        normalized = log_mel - log_mel.mean(dim=2, keepdim=True)
        if normalized.shape[1] != self.n_mels or not torch.isfinite(normalized).all():
            raise FeatureError("Log-Mel frontend produced invalid features.")
        return normalized

