from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

import pytest
import torch

from dataclasses import replace

from module_a.src.config import load_config
from module_a.src.model_factory import build_model
from module_a.src.models.wavlm_frontend import DeterministicFakeWavLM


@pytest.fixture
def write_wav():
    def _write(
        path: Path,
        *,
        sample_rate: int = 16_000,
        duration_sec: float = 0.1,
        channels: int = 1,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = round(sample_rate * duration_sec)
        mono = [
            int(2_000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(frame_count)
        ]
        samples = array("h", (value for value in mono for _ in range(channels)))
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(channels)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(samples.tobytes())
        return path

    return _write


@pytest.fixture
def small_model_config():
    config = load_config()
    return replace(
        config,
        model=replace(
            config.model,
            campp_growth_rate=4,
            campp_block_layers=(1, 1, 1),
            campp_init_channels=8,
            campp_bottleneck_channels=8,
            campp_fcm_channels=4,
            campp_segment_frames=4,
        ),
    )


@pytest.fixture
def training_model(small_model_config):
    torch.manual_seed(42)
    return build_model(
        small_model_config,
        num_classes=3,
        frontend=DeterministicFakeWavLM(
            small_model_config.model.wavlm_hidden_dimension,
            frame_count=12,
        ),
    )


@pytest.fixture
def waveform_batch():
    first = torch.linspace(-0.5, 0.5, steps=1_600)
    return torch.stack((first, first.flip(0)))
