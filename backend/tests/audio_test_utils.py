from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(
    path: Path,
    duration_sec: float = 2.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    silent: bool = False,
    noise_amplitude: float = 0.0,
) -> Path:
    frame_count = int(duration_sec * sample_rate)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    if silent:
        mono = np.zeros(frame_count, dtype=np.int16)
    else:
        signal = (
            0.34 * np.sin(2 * np.pi * 190 * time)
            + 0.18 * np.sin(2 * np.pi * 380 * time)
            + 0.08 * np.sin(2 * np.pi * 570 * time)
        )
        envelope = np.where((time % 0.8) < 0.62, 1.0, 0.015)
        noise = np.random.default_rng(20260821).normal(0.0, 1.0, frame_count)
        combined = signal * envelope + noise_amplitude * noise
        mono = np.asarray(np.clip(combined, -0.98, 0.98) * 32767, dtype=np.int16)

    frames = mono if channels == 1 else np.repeat(mono[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.astype("<i2").tobytes())
    return path
