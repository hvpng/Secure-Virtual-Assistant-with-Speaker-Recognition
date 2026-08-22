from __future__ import annotations

import subprocess
import sys
import wave
from array import array
from pathlib import Path

import pytest
import torch

from module_a.src import audio_batch
from module_a.src.audio_batch import (
    AudioBatchError,
    collate_fixed_waveforms,
    fit_waveform_to_segment,
    fit_waveform_to_segment_random,
    load_waveform,
    prepare_deterministic_segment,
)


def _write_pcm24_wav(path: Path, samples: list[int], *, sample_rate: int = 16_000) -> Path:
    payload = bytearray()
    for sample in samples:
        encoded = sample if sample >= 0 else (1 << 24) + sample
        payload.extend(
            (encoded & 0xFF, (encoded >> 8) & 0xFF, (encoded >> 16) & 0xFF)
        )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(3)
        stream.setframerate(sample_rate)
        stream.writeframes(bytes(payload))
    return path


def test_audio_loader_uses_pcm16_fast_path(tmp_path, write_wav, monkeypatch):
    path = write_wav(tmp_path / "pcm16.wav", sample_rate=16_000, duration_sec=0.1)

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("PCM16 WAV unexpectedly used the torchaudio fallback")

    def unexpected_resample(*_args, **_kwargs):
        raise AssertionError("16 kHz WAV unexpectedly entered the resampler")

    monkeypatch.setattr(audio_batch.torchaudio, "load", unexpected_fallback)
    monkeypatch.setattr(
        audio_batch.torchaudio.functional, "resample", unexpected_resample
    )

    waveform = load_waveform(path, target_sample_rate=16_000)

    assert waveform.ndim == 1
    assert waveform.dtype == torch.float32
    assert waveform.numel() == 1_600
    assert torch.isfinite(waveform).all()


def test_audio_loader_falls_back_for_normalized_pcm24(tmp_path):
    samples = [0, 1 << 22, -(1 << 22), (1 << 23) - 1, -(1 << 23)]
    path = _write_pcm24_wav(tmp_path / "pcm24.wav", samples)

    waveform = load_waveform(path)

    expected = torch.tensor(samples, dtype=torch.float32) / float(1 << 23)
    assert waveform.ndim == 1
    assert waveform.dtype == torch.float32
    assert torch.isfinite(waveform).all()
    assert torch.allclose(waveform, expected, atol=1e-6, rtol=0)


def test_audio_loader_averages_stereo_channels_to_mono(tmp_path):
    path = tmp_path / "stereo.wav"
    stereo_samples = array("h", [16_384, -16_384, 8_192, 0])
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(stereo_samples.tobytes())

    waveform = load_waveform(path)

    assert torch.allclose(waveform, torch.tensor([0.0, 0.125]), atol=1e-7, rtol=0)


def test_audio_loader_resamples_only_when_needed(tmp_path, write_wav, monkeypatch):
    path = write_wav(tmp_path / "pcm16_8k.wav", sample_rate=8_000, duration_sec=0.1)
    calls: list[tuple[int, int]] = []
    resample = audio_batch.torchaudio.functional.resample

    def tracked_resample(waveform, *, orig_freq, new_freq):
        calls.append((orig_freq, new_freq))
        return resample(waveform, orig_freq=orig_freq, new_freq=new_freq)

    monkeypatch.setattr(audio_batch.torchaudio.functional, "resample", tracked_resample)

    waveform = load_waveform(path, target_sample_rate=16_000)

    assert calls == [(8_000, 16_000)]
    assert waveform.numel() == pytest.approx(1_600, abs=2)
    assert waveform.dtype == torch.float32
    assert torch.isfinite(waveform).all()


def test_audio_loader_reports_path_when_wav_is_corrupt(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not a wav")

    with pytest.raises(AudioBatchError) as error:
        load_waveform(path)

    assert "Cannot decode WAV file" in str(error.value)
    assert str(path) in str(error.value)


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


def test_deterministic_segment_repeat_pads_short_audio_without_zeros():
    waveform = torch.tensor([1.0, 2.0, 3.0])
    segment = prepare_deterministic_segment(
        waveform, sample_rate=4, segment_seconds=2.0
    )

    assert segment.shape == (8,)
    assert torch.equal(segment, torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]))
    assert not torch.any(segment == 0)


def test_deterministic_segment_preserves_exact_length_audio():
    waveform = torch.arange(8, dtype=torch.float32)

    segment = prepare_deterministic_segment(
        waveform, sample_rate=4, segment_seconds=2.0
    )

    assert torch.equal(segment, waveform)


def test_deterministic_segment_center_crops_long_audio():
    waveform = torch.arange(12, dtype=torch.float32)

    segment = prepare_deterministic_segment(
        waveform, sample_rate=4, segment_seconds=2.0
    )

    assert torch.equal(segment, waveform[2:10])


def test_deterministic_segment_is_repeatable():
    waveform = torch.arange(13, dtype=torch.float32)

    first = prepare_deterministic_segment(
        waveform, sample_rate=4, segment_seconds=2.0
    )
    second = prepare_deterministic_segment(
        waveform, sample_rate=4, segment_seconds=2.0
    )

    assert torch.equal(first, second)


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
