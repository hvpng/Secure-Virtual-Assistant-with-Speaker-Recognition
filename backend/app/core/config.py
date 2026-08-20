import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    asr_model: str = os.getenv("ASR_MODEL", "vinai/PhoWhisper-small")
    asr_device: str = os.getenv("ASR_DEVICE", "auto")
    asr_fallback_enabled: bool = _env_bool("ASR_FALLBACK_ENABLED", True)
    speaker_device: str = os.getenv("SPEAKER_DEVICE", "auto")
    speaker_backend: str = os.getenv("SPEAKER_BACKEND", "fake")
    database_url: str = os.getenv("DATABASE_URL", "")
    app_mode: str = os.getenv("APP_MODE", "dev")


settings = Settings()
