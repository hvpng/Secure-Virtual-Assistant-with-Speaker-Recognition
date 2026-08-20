"""Vietnamese batch ASR and TTS services for Module M2."""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

import torch
from gtts import gTTS

from app.core.config import settings
from app.utils.audio_utils import normalize_audio


logger = logging.getLogger(__name__)

GENERATED_AUDIO_DIR = Path(__file__).resolve().parents[2] / "data" / "generated_audio"
FALLBACK_MODEL_NAME = "base"


class ASRServiceError(RuntimeError):
    """Base controlled error for speech recognition."""


class ASRConfigurationError(ASRServiceError):
    """Raised for an unsupported ASR configuration."""


class ASRModelUnavailableError(ASRServiceError):
    """Raised when an ASR model cannot be loaded."""


class ASRTranscriptionError(ASRServiceError):
    """Raised when loaded ASR cannot produce a non-empty transcript."""


class TTSServiceError(RuntimeError):
    """Raised when gTTS cannot produce a valid MP3."""


class ASRPipeline(Protocol):
    def __call__(self, audio: str, **kwargs: object) -> object: ...


_primary_lock = threading.Lock()
_primary_pipeline: ASRPipeline | None = None
_primary_key: tuple[str, str] | None = None

_fallback_lock = threading.Lock()
_fallback_model: object | None = None
_fallback_key: tuple[str, str] | None = None


def resolve_asr_device(requested: str | None = None) -> str:
    """Resolve auto/cuda/cpu without assuming that CUDA is available."""

    configured = (requested or settings.asr_device).strip().lower()
    if configured not in {"auto", "cuda", "cpu"}:
        raise ASRConfigurationError("ASR_DEVICE chỉ nhận 'auto', 'cuda' hoặc 'cpu'.")
    if configured == "cpu":
        selected = "cpu"
    elif torch.cuda.is_available():
        selected = "cuda"
    else:
        if configured == "cuda":
            logger.warning("ASR_DEVICE=cuda nhưng CUDA không khả dụng; dùng CPU.")
        selected = "cpu"
    logger.info("ASR selected device: %s", selected)
    return selected


def _create_transformers_pipeline(model_name: str, device: str) -> ASRPipeline:
    """Create the HuggingFace PhoWhisper pipeline (unit tests replace this)."""

    try:
        transformers = importlib.import_module("transformers")
        pipeline_factory = getattr(transformers, "pipeline")
        return pipeline_factory(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )
    except Exception as exc:
        raise ASRModelUnavailableError(
            f"Không thể load PhoWhisper model '{model_name}' trên {device}."
        ) from exc


def _load_primary_asr(model_name: str, device: str) -> ASRPipeline:
    """Load exactly one primary pipeline per process and reuse it."""

    global _primary_key, _primary_pipeline
    key = (model_name, device)
    if _primary_pipeline is not None and _primary_key == key:
        return _primary_pipeline
    with _primary_lock:
        if _primary_pipeline is not None and _primary_key == key:
            return _primary_pipeline
        logger.info("Loading PhoWhisper model '%s' on %s.", model_name, device)
        pipeline_instance = _create_transformers_pipeline(model_name, device)
        _primary_pipeline = pipeline_instance
        _primary_key = key
        return pipeline_instance


def _clear_primary_asr_cache() -> None:
    global _primary_key, _primary_pipeline
    with _primary_lock:
        _primary_pipeline = None
        _primary_key = None


def _create_fallback_model(model_name: str, device: str) -> object:
    """Lazy-import openai-whisper only after fallback activation."""

    try:
        whisper = importlib.import_module("whisper")
        return whisper.load_model(model_name, device=device)
    except Exception as exc:
        raise ASRModelUnavailableError(
            f"Không thể load multilingual Whisper '{model_name}' trên {device}."
        ) from exc


def _load_fallback_asr(model_name: str, device: str) -> object:
    global _fallback_key, _fallback_model
    key = (model_name, device)
    if _fallback_model is not None and _fallback_key == key:
        return _fallback_model
    with _fallback_lock:
        if _fallback_model is not None and _fallback_key == key:
            return _fallback_model
        logger.info("Loading multilingual Whisper '%s' on %s.", model_name, device)
        model = _create_fallback_model(model_name, device)
        _fallback_model = model
        _fallback_key = key
        return model


def _clear_fallback_asr_cache() -> None:
    global _fallback_key, _fallback_model
    with _fallback_lock:
        _fallback_model = None
        _fallback_key = None


def _extract_transcript(result: object, provider: str) -> str:
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict) and isinstance(result.get("text"), str):
        text = result["text"]
    else:
        raise ASRTranscriptionError(
            f"{provider} trả kết quả transcription không hợp lệ."
        )
    transcript = text.strip()
    if not transcript:
        raise ASRTranscriptionError(f"{provider} không nhận dạng được nội dung giọng nói.")
    return transcript


def _transcribe_primary(normalized_path: str, device: str) -> str:
    pipeline_instance = _load_primary_asr(settings.asr_model, device)
    try:
        result = pipeline_instance(
            normalized_path,
            generate_kwargs={"language": "vi", "task": "transcribe"},
        )
    except ASRServiceError:
        raise
    except Exception as exc:
        raise ASRTranscriptionError("PhoWhisper transcription thất bại.") from exc
    return _extract_transcript(result, "PhoWhisper")


def _transcribe_fallback(normalized_path: str, device: str) -> str:
    model = _load_fallback_asr(FALLBACK_MODEL_NAME, device)
    transcribe = getattr(model, "transcribe", None)
    if not callable(transcribe):
        raise ASRModelUnavailableError(
            "Multilingual Whisper fallback không có transcribe()."
        )
    try:
        result = transcribe(normalized_path, language="vi", task="transcribe")
    except Exception as exc:
        raise ASRTranscriptionError(
            "Multilingual Whisper fallback transcription thất bại."
        ) from exc
    return _extract_transcript(result, "Multilingual Whisper fallback")


def _controlled_asr_error(exc: Exception) -> ASRServiceError:
    if isinstance(exc, ASRServiceError):
        return exc
    return ASRTranscriptionError("PhoWhisper transcription thất bại.")


def _release_failed_cuda_primary() -> None:
    _clear_primary_asr_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def speech_to_text(audio_path: str) -> str:
    """Normalize audio, transcribe Vietnamese speech, and clean temp WAV."""

    normalized_path: str | None = None
    try:
        normalized_path = normalize_audio(audio_path)
        selected_device = resolve_asr_device()
        primary_error: ASRServiceError | None = None

        try:
            return _transcribe_primary(normalized_path, selected_device)
        except Exception as exc:
            primary_error = _controlled_asr_error(exc)
            logger.warning("PhoWhisper failure on %s: %s", selected_device, primary_error)

        if selected_device == "cuda" and settings.asr_device.strip().lower() == "auto":
            logger.warning("PhoWhisper CUDA failed in auto mode; retrying PhoWhisper on CPU.")
            _release_failed_cuda_primary()
            try:
                return _transcribe_primary(normalized_path, "cpu")
            except Exception as exc:
                primary_error = _controlled_asr_error(exc)
                logger.warning("PhoWhisper CPU retry failed: %s", primary_error)

        if not settings.asr_fallback_enabled:
            assert primary_error is not None
            raise primary_error

        if selected_device == "cuda":
            _release_failed_cuda_primary()
        else:
            _clear_primary_asr_cache()
        fallback_device = "cpu" if selected_device == "cuda" else selected_device
        logger.warning("PhoWhisper unavailable; using multilingual Whisper fallback.")
        if fallback_device == "cpu":
            logger.warning("Multilingual Whisper fallback is running on CPU.")
        try:
            return _transcribe_fallback(normalized_path, fallback_device)
        except Exception as exc:
            fallback_error = (
                exc
                if isinstance(exc, ASRServiceError)
                else ASRTranscriptionError("Whisper fallback transcription thất bại.")
            )
            raise ASRTranscriptionError(
                "Cả PhoWhisper và multilingual Whisper fallback đều thất bại."
            ) from fallback_error
    finally:
        if normalized_path is not None:
            Path(normalized_path).unlink(missing_ok=True)


def text_to_speech(text: str) -> str:
    """Generate a unique Vietnamese MP3; M4/M7 own response-lifecycle cleanup."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Văn bản TTS không được để trống.")

    GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, output_name = tempfile.mkstemp(
        prefix="tts_", suffix=".mp3", dir=GENERATED_AUDIO_DIR
    )
    os.close(descriptor)
    output_path = Path(output_name)
    try:
        gTTS(text=text.strip(), lang="vi").save(str(output_path))
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise TTSServiceError("gTTS không tạo được file MP3 hợp lệ.")
        return str(output_path.resolve())
    except TTSServiceError:
        output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise TTSServiceError("Không thể tạo audio tiếng Việt bằng gTTS.") from exc
