import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    asr_device: str = os.getenv("ASR_DEVICE", "auto")
    speaker_device: str = os.getenv("SPEAKER_DEVICE", "auto")
    speaker_backend: str = os.getenv("SPEAKER_BACKEND", "fake")
    app_mode: str = os.getenv("APP_MODE", "dev")


settings = Settings()

