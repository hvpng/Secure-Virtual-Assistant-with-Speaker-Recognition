from __future__ import annotations

import wave
from pathlib import Path

from app.utils.audio_utils import normalize_audio
from tests.audio_test_utils import write_wav


def test_normalize_audio_creates_canonical_temp_wav(tmp_path: Path) -> None:
    source = write_wav(
        tmp_path / "stereo_8khz.wav", sample_rate=8_000, channels=2
    )
    source_before = source.read_bytes()

    normalized = Path(normalize_audio(str(source)))
    try:
        assert normalized != source
        assert normalized.is_file()
        with wave.open(str(normalized), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16_000
            assert wav_file.getsampwidth() == 2
            assert wav_file.getcomptype() == "NONE"
        assert source.read_bytes() == source_before
    finally:
        normalized.unlink(missing_ok=True)
