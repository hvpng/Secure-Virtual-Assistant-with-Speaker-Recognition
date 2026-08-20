"""Manual M2 smoke test; this script may download models and call gTTS."""

from __future__ import annotations

import argparse

from app.services.asr_tts_service import (
    resolve_asr_device,
    speech_to_text,
    text_to_speech,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual Vietnamese ASR/TTS smoke test")
    parser.add_argument("audio_path", nargs="?", help="Local Vietnamese WAV/WebM path")
    parser.add_argument(
        "--tts-text",
        help="Optional Vietnamese text for a live gTTS network smoke test",
    )
    args = parser.parse_args()

    print(f"Selected ASR device: {resolve_asr_device()}")
    if args.audio_path:
        print(f"Transcript: {speech_to_text(args.audio_path)}")
    if args.tts_text:
        print(f"Generated MP3: {text_to_speech(args.tts_text)}")
    if not args.audio_path and not args.tts_text:
        parser.error("Provide audio_path and/or --tts-text")


if __name__ == "__main__":
    main()
