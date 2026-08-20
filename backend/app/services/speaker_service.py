"""Stable speaker verification, identification, and enrollment service."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable, TypedDict

import jiwer
import numpy as np
import webrtcvad

from app.core.config import settings
from app.utils.audio_utils import CANONICAL_SAMPLE_RATE, normalize_audio


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SPEAKER_CONFIG_PATH = MODELS_DIR / "speaker_config.json"
ENROLLMENT_CONFIG_PATH = MODELS_DIR / "enrollment_config.json"
VOICE_PROFILES_DIR = Path(__file__).resolve().parents[2] / "data" / "voice_profiles"

EMPLOYEE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
VAD_FRAME_DURATION_MS = 30
VAD_AGGRESSIVENESS = 2
PCM16_MAX = float(np.iinfo(np.int16).max)
SNR_DB_LIMIT = 120.0


class SpeakerServiceError(RuntimeError):
    """Base error for controlled speaker-service failures."""


class SpeakerBackendUnavailableError(SpeakerServiceError):
    """Raised when the selected backend does not satisfy the stable ABI."""


class SpeakerConfigurationError(SpeakerServiceError):
    """Raised when a canonical threshold config is missing or invalid."""


class VoiceProfileError(SpeakerServiceError):
    """Raised when a persisted embedding is invalid."""


class QualityChecks(TypedDict):
    duration_ok: bool
    speech_ratio_ok: bool
    snr_ok: bool
    clipping_ok: bool
    content_match_ok: bool


class QualityMetrics(TypedDict):
    duration_sec: float
    speech_ratio: float
    snr_db: float
    clipping_ratio: float
    content_wer: float


TranscribeFn = Callable[[str], str]


@dataclass(frozen=True)
class EnrollmentThresholds:
    min_duration: float
    max_duration: float
    min_speech_ratio: float
    min_snr_db: float
    max_clipping_ratio: float
    max_content_wer: float


@dataclass(frozen=True)
class SpeakerThresholds:
    sv_threshold: float
    sid_threshold: float
    embedding_dimension: int | None


@dataclass(frozen=True)
class VadFrame:
    samples: np.ndarray
    is_speech: bool


def _read_config(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpeakerConfigurationError(f"Không đọc được cấu hình speaker: {path}") from exc
    if not isinstance(data, dict):
        raise SpeakerConfigurationError(f"Cấu hình phải là JSON object: {path}")
    if data.get("development_placeholder") is True and settings.app_mode.lower() in {
        "demo",
        "final",
        "production",
        "prod",
    }:
        raise SpeakerConfigurationError(
            f"Không được dùng development threshold trong APP_MODE={settings.app_mode}: {path.name}"
        )
    return data


def _number(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpeakerConfigurationError(f"Threshold '{key}' không hợp lệ.")
    return float(value)


def load_enrollment_thresholds() -> EnrollmentThresholds:
    data = _read_config(ENROLLMENT_CONFIG_PATH)
    thresholds = EnrollmentThresholds(
        min_duration=_number(data, "min_duration"),
        max_duration=_number(data, "max_duration"),
        min_speech_ratio=_number(data, "min_speech_ratio"),
        min_snr_db=_number(data, "min_snr_db"),
        max_clipping_ratio=_number(data, "max_clipping_ratio"),
        max_content_wer=_number(data, "max_content_wer"),
    )
    if thresholds.min_duration < 0 or thresholds.min_duration > thresholds.max_duration:
        raise SpeakerConfigurationError("Khoảng duration threshold không hợp lệ.")
    if not 0 <= thresholds.min_speech_ratio <= 1:
        raise SpeakerConfigurationError("min_speech_ratio phải nằm trong [0, 1].")
    if not 0 <= thresholds.max_clipping_ratio <= 1:
        raise SpeakerConfigurationError("max_clipping_ratio phải nằm trong [0, 1].")
    if thresholds.max_content_wer < 0:
        raise SpeakerConfigurationError("max_content_wer không được âm.")
    return thresholds


def load_speaker_thresholds() -> SpeakerThresholds:
    data = _read_config(SPEAKER_CONFIG_PATH)
    dimension_value = data.get("embedding_dimension")
    if dimension_value is not None and (
        isinstance(dimension_value, bool)
        or not isinstance(dimension_value, int)
        or dimension_value <= 0
    ):
        raise SpeakerConfigurationError("embedding_dimension không hợp lệ.")
    thresholds = SpeakerThresholds(
        sv_threshold=_number(data, "sv_threshold"),
        sid_threshold=_number(data, "sid_threshold"),
        embedding_dimension=dimension_value,
    )
    if not -1 <= thresholds.sv_threshold <= 1:
        raise SpeakerConfigurationError("sv_threshold phải nằm trong [-1, 1].")
    if not -1 <= thresholds.sid_threshold <= 1:
        raise SpeakerConfigurationError("sid_threshold phải nằm trong [-1, 1].")
    return thresholds


def _backend_name() -> str:
    name = os.getenv("SPEAKER_BACKEND", settings.speaker_backend).strip().lower()
    if name not in {"fake", "real"}:
        raise SpeakerConfigurationError("SPEAKER_BACKEND chỉ nhận 'fake' hoặc 'real'.")
    return name


@lru_cache(maxsize=4)
def _load_backend(name: str, device: str) -> tuple[ModuleType, object]:
    module_name = (
        "app.models.fake_speaker_model"
        if name == "fake"
        else "app.models.speaker_model"
    )
    try:
        backend = importlib.import_module(module_name)
    except Exception as exc:
        raise SpeakerBackendUnavailableError(
            f"Không thể import speaker backend '{name}' từ {module_name}."
        ) from exc

    load_model = getattr(backend, "load_model", None)
    backend_extract = getattr(backend, "extract_embedding", None)
    if not callable(load_model) or not callable(backend_extract):
        if name == "real":
            raise SpeakerBackendUnavailableError(
                "SPEAKER_BACKEND=real nhưng artifact Module A chưa sẵn sàng: "
                "app.models.speaker_model phải có load_model() và extract_embedding()."
            )
        raise SpeakerBackendUnavailableError(
            "Fake speaker backend không cung cấp stable model interface."
        )
    try:
        model = load_model(device=device)
    except Exception as exc:
        raise SpeakerBackendUnavailableError(
            f"Không load được speaker backend '{name}' trên device '{device}'."
        ) from exc
    return backend, model


def _normalize_embedding(embedding: object, expected_dimension: int | None = None) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise VoiceProfileError("Embedding phải là vector 1 chiều, hữu hạn và không rỗng.")
    if expected_dimension is not None and array.size != expected_dimension:
        raise VoiceProfileError(
            f"Embedding dimension không hợp lệ: cần {expected_dimension}, nhận {array.size}."
        )
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= np.finfo(np.float32).eps:
        raise VoiceProfileError("Embedding có norm bằng 0 hoặc không hợp lệ.")
    return array / norm


def extract_embedding(audio_path: str) -> np.ndarray:
    """Normalize audio, invoke the selected model ABI, and clean temp audio."""

    normalized_path: str | None = None
    try:
        normalized_path = normalize_audio(audio_path)
        name = _backend_name()
        backend, model = _load_backend(name, settings.speaker_device)
        embedding = backend.extract_embedding(model, normalized_path)
        return _normalize_embedding(
            embedding, load_speaker_thresholds().embedding_dimension
        )
    finally:
        if normalized_path is not None:
            Path(normalized_path).unlink(missing_ok=True)


def _read_pcm16(normalized_path: str) -> tuple[np.ndarray, bytes, int]:
    try:
        with wave.open(normalized_path, "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getsampwidth() != 2
                or wav_file.getframerate() != CANONICAL_SAMPLE_RATE
                or wav_file.getcomptype() != "NONE"
            ):
                raise SpeakerServiceError("Audio chưa đúng chuẩn PCM WAV mono 16 kHz, 16-bit.")
            frame_count = wav_file.getnframes()
            pcm_bytes = wav_file.readframes(frame_count)
            return np.frombuffer(pcm_bytes, dtype="<i2"), pcm_bytes, frame_count
    except wave.Error as exc:
        raise SpeakerServiceError("Không đọc được normalized WAV.") from exc


def _classify_vad_frames(pcm_bytes: bytes) -> list[VadFrame]:
    """Split PCM16 into valid frames and classify each frame exactly once."""

    bytes_per_frame = CANONICAL_SAMPLE_RATE * VAD_FRAME_DURATION_MS // 1000 * 2
    complete_length = len(pcm_bytes) - len(pcm_bytes) % bytes_per_frame
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frames: list[VadFrame] = []
    for offset in range(0, complete_length, bytes_per_frame):
        pcm_frame = pcm_bytes[offset : offset + bytes_per_frame]
        frames.append(
            VadFrame(
                samples=np.frombuffer(pcm_frame, dtype="<i2"),
                is_speech=vad.is_speech(pcm_frame, CANONICAL_SAMPLE_RATE),
            )
        )
    return frames


def _calculate_speech_ratio(frames: list[VadFrame]) -> float:
    if not frames:
        return 0.0
    speech_frames = sum(frame.is_speech for frame in frames)
    return speech_frames / len(frames)


def _estimate_snr_db(frames: list[VadFrame]) -> float:
    """Estimate VAD-grouped SNR using the Module A calibration definition.

    Canonical PCM16 is split into non-overlapping, complete 30 ms frames. The
    shared WebRTC VAD mode-2 classification assigns every frame to the speech or
    non-speech group. Signal/noise power is the arithmetic mean of squared PCM
    sample values over the corresponding group, and SNR is
    ``10 * log10(speech_power / max(noise_power, 1 PCM-unit²))``. A missing
    speech or non-speech group, or zero speech power, returns ``0.0``. The final
    value is clamped to [-120, 120] dB so silence and degenerate inputs never
    produce NaN or infinity. Module A calibration must use this exact definition.
    """

    speech_samples = [frame.samples for frame in frames if frame.is_speech]
    noise_samples = [frame.samples for frame in frames if not frame.is_speech]
    if not speech_samples or not noise_samples:
        return 0.0

    speech = np.concatenate(speech_samples).astype(np.float64)
    noise = np.concatenate(noise_samples).astype(np.float64)
    speech_power = float(np.mean(np.square(speech)))
    noise_power = max(float(np.mean(np.square(noise))), 1.0)
    if not math.isfinite(speech_power) or speech_power <= 0:
        return 0.0
    snr_db = 10.0 * math.log10(speech_power / noise_power)
    if not math.isfinite(snr_db):
        return 0.0
    return float(np.clip(snr_db, -SNR_DB_LIMIT, SNR_DB_LIMIT))


def _normalize_content(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).lower()
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(normalized.split())


def _content_wer(expected_text: str, transcribed_text: str) -> float:
    expected = _normalize_content(expected_text)
    actual = _normalize_content(transcribed_text)
    if not expected:
        return 0.0 if not actual else 1.0
    return float(jiwer.wer(expected, actual))


def check_audio_quality(
    audio_path: str,
    expected_text: str,
    transcribe_fn: TranscribeFn | None = None,
) -> dict[str, object]:
    thresholds = load_enrollment_thresholds()
    normalized_path: str | None = None
    try:
        normalized_path = normalize_audio(audio_path)
        samples, pcm_bytes, frame_count = _read_pcm16(normalized_path)
        duration_sec = frame_count / CANONICAL_SAMPLE_RATE
        vad_frames = _classify_vad_frames(pcm_bytes)
        speech_ratio = _calculate_speech_ratio(vad_frames)
        snr_db = _estimate_snr_db(vad_frames)
        clipping_ratio = (
            float(np.mean(np.abs(samples.astype(np.int32)) >= PCM16_MAX))
            if samples.size
            else 0.0
        )

        content_wer = 0.0
        content_failure_reason: str | None = None
        if expected_text:
            if transcribe_fn is None:
                content_wer = 1.0
                content_failure_reason = (
                    "Chưa có dịch vụ nhận dạng giọng nói để kiểm tra nội dung."
                )
            else:
                try:
                    content_wer = _content_wer(
                        expected_text, transcribe_fn(normalized_path)
                    )
                except Exception:
                    content_wer = 1.0
                    content_failure_reason = "Không thể kiểm tra nội dung đã đọc."

        checks: QualityChecks = {
            "duration_ok": thresholds.min_duration
            <= duration_sec
            <= thresholds.max_duration,
            "speech_ratio_ok": speech_ratio >= thresholds.min_speech_ratio,
            "snr_ok": snr_db >= thresholds.min_snr_db,
            "clipping_ok": clipping_ratio <= thresholds.max_clipping_ratio,
            "content_match_ok": content_failure_reason is None
            and content_wer <= thresholds.max_content_wer,
        }
        reasons: list[str] = []
        if duration_sec < thresholds.min_duration:
            reasons.append("Âm thanh quá ngắn.")
        elif duration_sec > thresholds.max_duration:
            reasons.append("Âm thanh quá dài.")
        if not checks["speech_ratio_ok"]:
            reasons.append("Không phát hiện đủ giọng nói.")
        if not checks["snr_ok"]:
            reasons.append("Âm thanh quá nhiễu.")
        if not checks["clipping_ok"]:
            reasons.append("Âm thanh bị vỡ do âm lượng quá lớn.")
        if not checks["content_match_ok"]:
            reasons.append(
                content_failure_reason
                or "Nội dung đọc không khớp câu yêu cầu."
            )

        metrics: QualityMetrics = {
            "duration_sec": round(duration_sec, 6),
            "speech_ratio": round(speech_ratio, 6),
            "snr_db": round(snr_db, 6),
            "clipping_ratio": round(clipping_ratio, 6),
            "content_wer": round(content_wer, 6),
        }
        return {
            "pass": all(checks.values()),
            "checks": checks,
            "metrics": metrics,
            "reasons": reasons,
        }
    finally:
        if normalized_path is not None:
            Path(normalized_path).unlink(missing_ok=True)


def _validate_employee_id(employee_id: str) -> str:
    if not isinstance(employee_id, str) or not EMPLOYEE_ID_PATTERN.fullmatch(employee_id):
        raise ValueError(
            "employee_id chỉ được chứa chữ ASCII, số, '_' hoặc '-', tối đa 64 ký tự."
        )
    return employee_id


def _profile_path(employee_id: str) -> Path:
    validated = _validate_employee_id(employee_id)
    return VOICE_PROFILES_DIR / f"{validated}.npy"


def _save_profile_atomic(employee_id: str, embedding: np.ndarray) -> None:
    profile_path = _profile_path(employee_id)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npy", prefix=f".{employee_id}.",
            dir=profile_path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.save(temporary, embedding.astype(np.float32), allow_pickle=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, profile_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_profile_path(path: Path, expected_dimension: int | None) -> np.ndarray:
    try:
        embedding = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise VoiceProfileError(f"Voice profile không hợp lệ: {path.name}") from exc
    return _normalize_embedding(embedding, expected_dimension)


def has_voice_profile(employee_id: str) -> bool:
    return _profile_path(employee_id).is_file()


def delete_voice_profile(employee_id: str) -> bool:
    profile_path = _profile_path(employee_id)
    if not profile_path.exists():
        return False
    profile_path.unlink()
    return True


def enroll_user(
    employee_id: str,
    audio_paths: list[str],
    expected_texts: list[str],
    transcribe_fn: TranscribeFn | None = None,
) -> dict[str, object]:
    _validate_employee_id(employee_id)
    if not audio_paths or len(audio_paths) != len(expected_texts):
        raise ValueError("audio_paths và expected_texts phải cùng độ dài và không rỗng.")

    quality_results = [
        check_audio_quality(audio_path, expected_text, transcribe_fn)
        for audio_path, expected_text in zip(audio_paths, expected_texts, strict=True)
    ]
    failed_items = [
        {
            "index": index,
            "checks": result["checks"],
            "reasons": result["reasons"],
        }
        for index, result in enumerate(quality_results)
        if not result["pass"]
    ]
    if failed_items:
        return {"success": False, "failed_items": failed_items}

    embeddings = [extract_embedding(path) for path in audio_paths]
    dimension = embeddings[0].size
    if any(embedding.size != dimension for embedding in embeddings):
        raise VoiceProfileError("Các enrollment embedding không cùng dimension.")
    profile = _normalize_embedding(np.mean(np.stack(embeddings), axis=0), dimension)
    _save_profile_atomic(employee_id, profile)
    return {"success": True, "failed_items": []}


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise VoiceProfileError("Query và profile embedding không cùng dimension.")
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def verify(audio_path: str, claimed_employee_id: str) -> dict[str, object]:
    profile_path = _profile_path(claimed_employee_id)
    if not profile_path.is_file():
        return {"is_match": False, "score": 0.0}
    thresholds = load_speaker_thresholds()
    try:
        profile = _load_profile_path(profile_path, thresholds.embedding_dimension)
        query = extract_embedding(audio_path)
        score = _cosine_similarity(query, profile)
    except VoiceProfileError:
        return {"is_match": False, "score": 0.0}
    return {"is_match": score >= thresholds.sv_threshold, "score": score}


def identify(audio_path: str) -> dict[str, object]:
    thresholds = load_speaker_thresholds()
    profile_directory = VOICE_PROFILES_DIR
    if not profile_directory.is_dir():
        return {"employee_id": None, "score": 0.0}

    profile_paths = sorted(profile_directory.glob("*.npy"))
    if not profile_paths:
        return {"employee_id": None, "score": 0.0}

    query = extract_embedding(audio_path)
    best_employee_id: str | None = None
    best_score = -1.0
    for profile_path in profile_paths:
        try:
            _validate_employee_id(profile_path.stem)
            profile = _load_profile_path(
                profile_path, thresholds.embedding_dimension
            )
            score = _cosine_similarity(query, profile)
        except (ValueError, VoiceProfileError):
            continue
        if score > best_score:
            best_score = score
            best_employee_id = profile_path.stem

    if best_employee_id is None:
        return {"employee_id": None, "score": 0.0}
    if best_score < thresholds.sid_threshold:
        return {"employee_id": None, "score": best_score}
    return {"employee_id": best_employee_id, "score": best_score}
