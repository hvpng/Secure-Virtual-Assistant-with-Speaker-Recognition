from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from module_a.src.config import load_config


@pytest.fixture
def tiny_config():
    config = copy.deepcopy(load_config())
    config["audio"]["segment_seconds"] = 0.2
    config["model"].update({"channels": 16, "scale": 4, "se_channels": 4})
    config["training"].update(
        {
            "epochs": 1,
            "batch_size": 4,
            "speakers_per_batch": 2,
            "utterances_per_speaker": 2,
            "gradient_accumulation": 1,
            "amp": False,
            "num_workers": 0,
            "validate_every_epochs": 1,
            "max_positive_trials_per_speaker": 1,
        }
    )
    config["augmentation"].update(
        {"speed_perturb": False, "additive_noise": False, "reverb": False}
    )
    config["evaluation"]["batch_size"] = 4
    config["calibration"].update(
        {"sid_known_ratio": 0.5, "sid_max_enrollment": 1}
    )
    return config


@pytest.fixture
def write_wav():
    def _write(path: Path, *, seconds: float = 0.25, sample_rate: int = 16_000) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        count = round(seconds * sample_rate)
        time = np.arange(count, dtype=np.float32) / sample_rate
        waveform = 0.1 * np.sin(2 * np.pi * 220 * time)
        sf.write(path, waveform, sample_rate, subtype="PCM_16")
        return path

    return _write
