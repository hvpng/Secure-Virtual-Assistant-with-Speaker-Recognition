from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from module_a.src.audio_batch import (
    AudioBatchError,
    collate_fixed_waveforms,
    fit_waveform_to_segment,
    fit_waveform_to_segment_random,
    load_waveform,
)


def test_audio_loader_returns_mono_16k_float32(tmp_path, write_wav):
    path = write_wav(
        tmp_path / "stereo.wav",
        sample_rate=8_000,
        duration_sec=0.1,
        channels=2,
    )

    waveform = load_waveform(path, target_sample_rate=16_000)

    assert waveform.ndim == 1
    assert waveform.dtype == torch.float32
    assert waveform.numel() == pytest.approx(1_600, abs=2)
    assert torch.isfinite(waveform).all()


def test_fixed_collator_center_crops_and_repeat_pads_deterministically():
    short = torch.tensor([1.0, 2.0, 3.0])
    long = torch.arange(10, dtype=torch.float32)
    batch, mask = collate_fixed_waveforms(
        [short, long], sample_rate=4, segment_seconds=1.0
    )

    assert batch.tolist() == [[1.0, 2.0, 3.0, 1.0], [3.0, 4.0, 5.0, 6.0]]
    assert mask.shape == batch.shape
    assert torch.all(mask == 1)
    assert torch.equal(fit_waveform_to_segment(short, 4), batch[0])


def test_audio_batch_rejects_invalid_rank_and_nan():
    with pytest.raises(AudioBatchError, match="rank-1"):
        fit_waveform_to_segment(torch.randn(1, 10), 20)
    waveform = torch.randn(10)
    waveform[0] = float("nan")
    with pytest.raises(AudioBatchError, match="finite"):
        fit_waveform_to_segment(waveform, 20)


def test_train_random_crop_is_seeded_and_short_audio_still_repeat_pads():
    waveform = torch.arange(20, dtype=torch.float32)
    torch.manual_seed(42)
    first = fit_waveform_to_segment_random(waveform, 5)
    torch.manual_seed(42)
    second = fit_waveform_to_segment_random(waveform, 5)

    assert torch.equal(first, second)
    assert torch.equal(
        fit_waveform_to_segment_random(torch.tensor([1.0, 2.0]), 5),
        torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0]),
    )


def test_real_smoke_help_does_not_load_or_download_model():
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "module_a.scripts.smoke_model", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "--device" in result.stdout
    assert "--local-files-only" in result.stdout
