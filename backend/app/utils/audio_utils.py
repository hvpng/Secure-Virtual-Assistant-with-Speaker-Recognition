"""Shared audio normalization for speaker and later speech services."""

from __future__ import annotations

import re
import shutil
import tempfile
import wave
from pathlib import Path

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


CANONICAL_SAMPLE_RATE = 16_000
CANONICAL_CHANNELS = 1
CANONICAL_SAMPLE_WIDTH = 2


class AudioNormalizationError(ValueError):
    """Raised when an input cannot be converted to canonical PCM WAV."""


def _safe_temp_prefix(input_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", input_path.stem).strip("_")
    return f"{(stem or 'audio')[:48]}_"


def normalize_audio(input_path: str) -> str:
    """Create a temporary 16 kHz, mono, 16-bit PCM WAV copy.

    The source is never overwritten. The caller owns the returned path and must
    remove it in a ``finally`` block.
    """

    source = Path(input_path)
    if not source.is_file():
        raise AudioNormalizationError(f"Không tìm thấy file âm thanh: {source}")

    if source.suffix.lower() != ".wav" and shutil.which("ffmpeg") is None:
        raise AudioNormalizationError(
            "Không tìm thấy ffmpeg trên PATH để chuyển đổi âm thanh không phải WAV."
        )

    temporary = tempfile.NamedTemporaryFile(
        prefix=_safe_temp_prefix(source), suffix=".wav", delete=False
    )
    normalized_path = Path(temporary.name)
    temporary.close()

    try:
        audio = AudioSegment.from_file(source)
        audio = (
            audio.set_channels(CANONICAL_CHANNELS)
            .set_frame_rate(CANONICAL_SAMPLE_RATE)
            .set_sample_width(CANONICAL_SAMPLE_WIDTH)
        )

        with wave.open(str(normalized_path), "wb") as wav_file:
            wav_file.setnchannels(CANONICAL_CHANNELS)
            wav_file.setsampwidth(CANONICAL_SAMPLE_WIDTH)
            wav_file.setframerate(CANONICAL_SAMPLE_RATE)
            wav_file.writeframes(audio.raw_data)

        with wave.open(str(normalized_path), "rb") as wav_file:
            if (
                wav_file.getnchannels() != CANONICAL_CHANNELS
                or wav_file.getframerate() != CANONICAL_SAMPLE_RATE
                or wav_file.getsampwidth() != CANONICAL_SAMPLE_WIDTH
                or wav_file.getcomptype() != "NONE"
            ):
                raise AudioNormalizationError(
                    "Không thể tạo âm thanh WAV PCM mono 16 kHz, 16-bit."
                )
    except AudioNormalizationError:
        normalized_path.unlink(missing_ok=True)
        raise
    except FileNotFoundError as exc:
        normalized_path.unlink(missing_ok=True)
        raise AudioNormalizationError(
            "Không tìm thấy ffmpeg trên PATH để giải mã âm thanh."
        ) from exc
    except (CouldntDecodeError, wave.Error, OSError) as exc:
        normalized_path.unlink(missing_ok=True)
        raise AudioNormalizationError(
            f"File âm thanh không hợp lệ hoặc không thể giải mã: {source}"
        ) from exc
    except Exception:
        normalized_path.unlink(missing_ok=True)
        raise

    return str(normalized_path)
