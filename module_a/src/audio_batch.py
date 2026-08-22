"""Shared A2/A3 waveform loading and fixed-segment collation."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio
from torch import Tensor


class AudioBatchError(ValueError):
    """Raised when an audio file or waveform cannot meet the model contract."""


DETERMINISTIC_SEGMENT_POLICY_VERSION = "deterministic_center_crop_repeat_pad_v1"


def _load_pcm16_wav(path: Path) -> tuple[Tensor, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getcomptype() != "NONE" or audio.getsampwidth() != 2:
                raise AudioBatchError(
                    "Strict WAV fast path supports only uncompressed PCM16."
                )
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            frames = audio.readframes(audio.getnframes())
    except (EOFError, wave.Error, OSError) as exc:
        raise AudioBatchError(f"Cannot decode PCM16 WAV: {path}") from exc
    if channels <= 0 or sample_rate <= 0 or not frames:
        raise AudioBatchError("WAV header or audio payload is empty.")
    samples = np.frombuffer(frames, dtype="<i2").copy()
    if samples.size % channels != 0:
        raise AudioBatchError("WAV payload does not align with its channel count.")
    waveform = torch.from_numpy(samples.reshape(-1, channels)).to(torch.float32)
    return waveform.mean(dim=1) / 32768.0, sample_rate


def _load_with_torchaudio(path: Path) -> tuple[Tensor, int]:
    """Decode normalized floating-point audio and average channels to mono."""

    try:
        decoded, sample_rate = torchaudio.load(str(path), normalize=True)
    except Exception as exc:
        raise AudioBatchError(f"torchaudio could not decode audio file: {path}") from exc
    if (
        decoded.ndim != 2
        or decoded.shape[0] < 1
        or decoded.shape[1] < 1
        or sample_rate <= 0
    ):
        raise AudioBatchError(
            f"Decoded audio must have shape [channels, samples] and a valid sample rate: {path}"
        )
    return decoded.to(torch.float32).mean(dim=0), int(sample_rate)


def load_waveform(audio_path: str | Path, *, target_sample_rate: int = 16_000) -> Tensor:
    """Load training audio as finite mono float32 at ``target_sample_rate``.

    Ordinary uncompressed PCM16 WAV files use the stdlib fast path, whose
    division by 32768 matches ``torchaudio.load(normalize=True)`` amplitude
    scaling. A WAV unsupported by that strict path (including valid PCM24) is
    decoded by torchaudio instead. This policy is scoped to Module A ingestion;
    it does not alter Track B's canonical PCM16 enrollment audio contract.
    """

    path = Path(audio_path)
    if not path.is_file():
        raise AudioBatchError(f"Audio file does not exist: {path}")
    if target_sample_rate <= 0:
        raise AudioBatchError("target_sample_rate must be positive.")
    if path.suffix.lower() == ".wav":
        try:
            waveform, sample_rate = _load_pcm16_wav(path)
        except AudioBatchError as strict_error:
            try:
                waveform, sample_rate = _load_with_torchaudio(path)
            except AudioBatchError as fallback_error:
                raise AudioBatchError(
                    f"Cannot decode WAV file {path}. "
                    f"Strict PCM16 decoder failed: {strict_error} "
                    f"Torchaudio fallback failed: {fallback_error}"
                ) from fallback_error
    else:
        waveform, sample_rate = _load_with_torchaudio(path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
    waveform = waveform.to(torch.float32)
    if (
        waveform.ndim != 1
        or waveform.numel() == 0
        or not torch.isfinite(waveform).all()
    ):
        raise AudioBatchError("Decoded waveform is empty or non-finite.")
    return waveform.contiguous()


def fit_waveform_to_segment(waveform: Tensor, segment_samples: int) -> Tensor:
    """Center-crop long audio and repeat-pad short audio to a fixed length.

    Repeat padding avoids sending silent padding frames into CAM++ statistics
    pooling. The operation is deterministic and introduces no A2 augmentation.
    """

    if waveform.ndim != 1:
        raise AudioBatchError("Each waveform must be a rank-1 mono tensor.")
    if waveform.numel() == 0 or segment_samples <= 0:
        raise AudioBatchError("Waveform and segment length must be non-empty.")
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise AudioBatchError("Waveform must be finite floating-point audio.")
    if waveform.numel() >= segment_samples:
        start = (waveform.numel() - segment_samples) // 2
        return waveform[start : start + segment_samples].contiguous()
    repeats = math.ceil(segment_samples / waveform.numel())
    return waveform.repeat(repeats)[:segment_samples].contiguous()


def prepare_deterministic_segment(
    waveform: Tensor,
    *,
    sample_rate: int,
    segment_seconds: float,
) -> Tensor:
    """Create the one fixed evaluation/monitor segment used by A3 and A4.

    The target length is ``round(sample_rate * segment_seconds)``. Long audio is
    center-cropped, exact-length audio is preserved, and short audio is repeated
    then truncated. No zero padding or random crop is used.
    """

    if sample_rate <= 0 or segment_seconds <= 0:
        raise AudioBatchError("sample_rate and segment_seconds must be positive.")
    segment_samples = round(sample_rate * segment_seconds)
    return fit_waveform_to_segment(waveform.to(torch.float32), segment_samples)


def fit_waveform_to_segment_random(waveform: Tensor, segment_samples: int) -> Tensor:
    """Random-crop long training audio and preserve A2 repeat-padding for short audio.

    ``torch.randint`` intentionally uses the current process/worker torch RNG. A3
    seeds the main process and each DataLoader worker explicitly.
    """

    if waveform.ndim != 1:
        raise AudioBatchError("Each waveform must be a rank-1 mono tensor.")
    if waveform.numel() == 0 or segment_samples <= 0:
        raise AudioBatchError("Waveform and segment length must be non-empty.")
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise AudioBatchError("Waveform must be finite floating-point audio.")
    if waveform.numel() <= segment_samples:
        return fit_waveform_to_segment(waveform, segment_samples)
    max_start = waveform.numel() - segment_samples
    start = int(torch.randint(max_start + 1, (1,)).item())
    return waveform[start : start + segment_samples].contiguous()


def collate_fixed_waveforms(
    waveforms: Iterable[Tensor],
    *,
    sample_rate: int = 16_000,
    segment_seconds: float = 3.0,
) -> tuple[Tensor, Tensor]:
    """Return [B, samples] float32 audio and an all-valid attention mask."""

    if sample_rate <= 0 or segment_seconds <= 0:
        raise AudioBatchError("sample_rate and segment_seconds must be positive.")
    items = list(waveforms)
    if not items:
        raise AudioBatchError("Cannot collate an empty waveform batch.")
    batch = torch.stack(
        [
            prepare_deterministic_segment(
                item,
                sample_rate=sample_rate,
                segment_seconds=segment_seconds,
            )
            for item in items
        ]
    )
    attention_mask = torch.ones(batch.shape, dtype=torch.long)
    return batch, attention_mask
