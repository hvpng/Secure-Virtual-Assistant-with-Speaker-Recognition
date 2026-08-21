from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

import pytest


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
