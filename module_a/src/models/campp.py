"""CAM++ reimplementation for 80-D WavLM adapter features.

This implementation is informed by Wang et al., "CAM++: A Fast and Efficient
Network for Speaker Verification Using Context-Aware Masking" (Interspeech
2023, arXiv:2303.00332) and the Apache-2.0 ModelScope/3D-Speaker reference.
It is a local reimplementation, not a verbatim copy or an official release.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CAMPlusPlusError(ValueError):
    """Raised when frame features violate the CAM++ tensor contract."""


class Residual2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == (1, 1)
            else nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        hidden = F.relu(self.bn1(self.conv1(inputs)), inplace=False)
        hidden = self.bn2(self.conv2(hidden))
        return F.relu(hidden + residual, inplace=False)


class FrequencyConvolutionModule(nn.Module):
    """Four residual blocks with an 8x frequency-only downsampling factor."""

    def __init__(self, feature_dim: int, channels: int) -> None:
        super().__init__()
        if feature_dim % 8 != 0:
            raise CAMPlusPlusError("CAM++ feature_dim must be divisible by 8.")
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            Residual2DBlock(channels, channels, (1, 1)),
            Residual2DBlock(channels, channels, (2, 1)),
            Residual2DBlock(channels, channels, (2, 1)),
            Residual2DBlock(channels, channels, (2, 1)),
        )
        self.output_channels = channels * (feature_dim // 8)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.blocks(self.stem(features.unsqueeze(1)))
        batch, channels, frequency, frames = hidden.shape
        return hidden.reshape(batch, channels * frequency, frames)


class ContextAwareMask(nn.Module):
    """Predict a ratio mask from global and fixed-segment mean context."""

    def __init__(
        self,
        context_channels: int,
        output_channels: int,
        segment_frames: int,
    ) -> None:
        super().__init__()
        self.segment_frames = segment_frames
        hidden_channels = max(output_channels // 2, 1)
        self.reduce = nn.Conv1d(context_channels, hidden_channels, kernel_size=1)
        self.expand = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)

    def _segment_context(self, inputs: Tensor) -> Tensor:
        context = torch.empty_like(inputs)
        frame_count = inputs.shape[-1]
        for start in range(0, frame_count, self.segment_frames):
            end = min(start + self.segment_frames, frame_count)
            context[..., start:end] = inputs[..., start:end].mean(dim=-1, keepdim=True)
        return context

    def forward(self, inputs: Tensor) -> Tensor:
        global_context = inputs.mean(dim=-1, keepdim=True).expand_as(inputs)
        context = global_context + self._segment_context(inputs)
        return torch.sigmoid(self.expand(F.relu(self.reduce(context), inplace=False)))


class CAMDenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bottleneck_channels: int,
        dilation: int,
        segment_frames: int,
    ) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
        )
        self.tdnn = nn.Sequential(
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv1d(
                bottleneck_channels,
                growth_rate,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                bias=False,
            ),
        )
        self.mask = ContextAwareMask(
            bottleneck_channels,
            growth_rate,
            segment_frames,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        bottleneck = self.bottleneck(inputs)
        local_features = self.tdnn(bottleneck)
        masked = local_features * self.mask(bottleneck)
        return torch.cat((inputs, masked), dim=1)


class CAMDenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        bottleneck_channels: int,
        dilation: int,
        segment_frames: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(
                CAMDenseLayer(
                    channels,
                    growth_rate,
                    bottleneck_channels,
                    dilation,
                    segment_frames,
                )
            )
            channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.output_channels = channels

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class StatisticsPooling(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=-1)
        variance = inputs.var(dim=-1, unbiased=False).clamp_min(1e-5)
        return torch.cat((mean, variance.sqrt()), dim=1)


class CAMPlusPlus(nn.Module):
    """CAM++ encoder accepting [batch, 80, frames] and returning embeddings."""

    def __init__(
        self,
        *,
        feature_dim: int = 80,
        embedding_dimension: int = 192,
        growth_rate: int = 32,
        block_layers: tuple[int, ...] = (12, 24, 16),
        init_channels: int = 128,
        bottleneck_channels: int = 128,
        fcm_channels: int = 32,
        segment_frames: int = 100,
    ) -> None:
        super().__init__()
        if feature_dim != 80:
            raise CAMPlusPlusError("Canonical CAM++ input feature dimension is 80.")
        if not block_layers or any(layer_count <= 0 for layer_count in block_layers):
            raise CAMPlusPlusError("CAM++ block_layers must contain positive integers.")
        self.feature_dim = feature_dim
        self.embedding_dimension = embedding_dimension
        self.fcm = FrequencyConvolutionModule(feature_dim, fcm_channels)
        self.input_tdnn = nn.Sequential(
            nn.BatchNorm1d(self.fcm.output_channels),
            nn.ReLU(),
            nn.Conv1d(
                self.fcm.output_channels,
                init_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
        )

        channels = init_channels
        blocks: list[nn.Module] = []
        dilations = (1, 2, 2)
        for index, layer_count in enumerate(block_layers):
            block = CAMDenseBlock(
                channels,
                layer_count,
                growth_rate,
                bottleneck_channels,
                dilations[min(index, len(dilations) - 1)],
                segment_frames,
            )
            blocks.append(block)
            channels = block.output_channels
            transition_channels = max(channels // 2, growth_rate)
            blocks.append(
                nn.Sequential(
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.Conv1d(channels, transition_channels, kernel_size=1, bias=False),
                )
            )
            channels = transition_channels
        self.backbone = nn.Sequential(*blocks)
        self.final_norm = nn.Sequential(nn.BatchNorm1d(channels), nn.ReLU())
        self.pool = StatisticsPooling()
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(channels * 2),
            nn.Linear(channels * 2, embedding_dimension),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise CAMPlusPlusError("CAM++ input must have shape [batch, feature_dim, frames].")
        if features.shape[1] != self.feature_dim:
            raise CAMPlusPlusError(
                f"CAM++ expected {self.feature_dim} features, received {features.shape[1]}."
            )
        if features.shape[2] < 1:
            raise CAMPlusPlusError("CAM++ requires at least one frame.")
        if not torch.isfinite(features).all():
            raise CAMPlusPlusError("CAM++ input contains NaN or infinity.")
        hidden = self.fcm(features)
        hidden = self.input_tdnn(hidden)
        hidden = self.final_norm(self.backbone(hidden))
        embedding = self.embedding(self.pool(hidden))
        if not torch.isfinite(embedding).all():
            raise CAMPlusPlusError("CAM++ produced a non-finite embedding.")
        return embedding
