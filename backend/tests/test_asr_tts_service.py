from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from app.services import asr_tts_service


@pytest.fixture(autouse=True)
def isolated_service_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = SimpleNamespace(
        asr_model="vinai/PhoWhisper-small",
        asr_device="cpu",
        asr_fallback_enabled=True,
    )
    monkeypatch.setattr(asr_tts_service, "settings", settings)
    monkeypatch.setattr(
        asr_tts_service, "GENERATED_AUDIO_DIR", tmp_path / "generated_audio"
    )
    asr_tts_service._clear_primary_asr_cache()
    asr_tts_service._clear_fallback_asr_cache()
    yield
    asr_tts_service._clear_primary_asr_cache()
    asr_tts_service._clear_fallback_asr_cache()


def _install_fake_normalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], list[Path]]:
    calls: list[str] = []
    normalized_paths: list[Path] = []

    def fake_normalize(input_path: str) -> str:
        calls.append(input_path)
        normalized = tmp_path / f"normalized_{len(calls)}.wav"
        normalized.write_bytes(b"canonical wav placeholder")
        normalized_paths.append(normalized)
        return str(normalized)

    monkeypatch.setattr(asr_tts_service, "normalize_audio", fake_normalize)
    return calls, normalized_paths


def _pipeline_with(result: object) -> Callable[..., object]:
    def pipeline(audio: str, **kwargs: object) -> object:
        assert Path(audio).is_file()
        assert kwargs["generate_kwargs"] == {
            "language": "vi",
            "task": "transcribe",
        }
        return result

    return pipeline


def test_speech_to_text_uses_shared_normalizer_and_cleans_temp_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, normalized_paths = _install_fake_normalizer(tmp_path, monkeypatch)
    monkeypatch.setattr(
        asr_tts_service,
        "_create_transformers_pipeline",
        lambda model, device: _pipeline_with({"text": "  Xin chào  "}),
    )

    transcript = asr_tts_service.speech_to_text("browser-recording.webm")

    assert transcript == "Xin chào"
    assert calls == ["browser-recording.webm"]
    assert not normalized_paths[0].exists()


def test_normalized_temp_is_cleaned_when_asr_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, normalized_paths = _install_fake_normalizer(tmp_path, monkeypatch)
    asr_tts_service.settings.asr_fallback_enabled = False
    monkeypatch.setattr(
        asr_tts_service,
        "_transcribe_primary",
        lambda *_: (_ for _ in ()).throw(
            asr_tts_service.ASRTranscriptionError("primary failed")
        ),
    )

    with pytest.raises(asr_tts_service.ASRTranscriptionError, match="primary failed"):
        asr_tts_service.speech_to_text("broken.wav")

    assert not normalized_paths[0].exists()


def test_primary_pipeline_loads_once_and_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_normalizer(tmp_path, monkeypatch)
    loads: list[tuple[str, str]] = []

    def factory(model_name: str, device: str) -> Callable[..., object]:
        loads.append((model_name, device))
        return _pipeline_with({"text": "xin chào"})

    monkeypatch.setattr(asr_tts_service, "_create_transformers_pipeline", factory)

    assert asr_tts_service.speech_to_text("one.wav") == "xin chào"
    assert asr_tts_service.speech_to_text("two.wav") == "xin chào"
    assert loads == [("vinai/PhoWhisper-small", "cpu")]


def test_empty_primary_transcription_raises_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_normalizer(tmp_path, monkeypatch)
    asr_tts_service.settings.asr_fallback_enabled = False
    monkeypatch.setattr(
        asr_tts_service,
        "_create_transformers_pipeline",
        lambda model, device: _pipeline_with({"text": "   "}),
    )

    with pytest.raises(
        asr_tts_service.ASRTranscriptionError,
        match="không nhận dạng được nội dung",
    ):
        asr_tts_service.speech_to_text("empty.wav")


def test_primary_failure_uses_lazy_fallback_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_normalizer(tmp_path, monkeypatch)
    fallback_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        asr_tts_service,
        "_transcribe_primary",
        lambda *_: (_ for _ in ()).throw(
            asr_tts_service.ASRModelUnavailableError("primary unavailable")
        ),
    )

    def fallback(path: str, device: str) -> str:
        fallback_calls.append((path, device))
        return "bản ghi dự phòng"

    monkeypatch.setattr(asr_tts_service, "_transcribe_fallback", fallback)

    transcript = asr_tts_service.speech_to_text("fallback.wav")

    assert transcript == "bản ghi dự phòng"
    assert fallback_calls and fallback_calls[0][1] == "cpu"
    assert "PhoWhisper unavailable; using multilingual Whisper fallback." in caplog.text


def test_primary_failure_without_fallback_raises_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_normalizer(tmp_path, monkeypatch)
    asr_tts_service.settings.asr_fallback_enabled = False
    monkeypatch.setattr(
        asr_tts_service,
        "_transcribe_primary",
        lambda *_: (_ for _ in ()).throw(RuntimeError("library error")),
    )
    fallback_called = False

    def forbidden_fallback(*_: object) -> str:
        nonlocal fallback_called
        fallback_called = True
        return "unexpected"

    monkeypatch.setattr(asr_tts_service, "_transcribe_fallback", forbidden_fallback)

    with pytest.raises(asr_tts_service.ASRTranscriptionError):
        asr_tts_service.speech_to_text("no-fallback.wav")

    assert fallback_called is False


def test_fallback_is_not_loaded_when_primary_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_normalizer(tmp_path, monkeypatch)
    monkeypatch.setattr(
        asr_tts_service, "_transcribe_primary", lambda *_: "primary result"
    )
    fallback_loads = 0

    def forbidden_loader(*_: object) -> object:
        nonlocal fallback_loads
        fallback_loads += 1
        raise AssertionError("fallback must remain lazy")

    monkeypatch.setattr(asr_tts_service, "_create_fallback_model", forbidden_loader)

    assert asr_tts_service.speech_to_text("primary.wav") == "primary result"
    assert fallback_loads == 0


@pytest.mark.parametrize(
    ("cuda_available", "expected"), [(True, "cuda"), (False, "cpu")]
)
def test_auto_device_resolution_is_deterministic(
    cuda_available: bool, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        asr_tts_service.torch.cuda, "is_available", lambda: cuda_available
    )

    assert asr_tts_service.resolve_asr_device("auto") == expected


def test_explicit_cuda_without_cuda_uses_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr_tts_service.torch.cuda, "is_available", lambda: False)

    assert asr_tts_service.resolve_asr_device("cuda") == "cpu"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_text_to_speech_rejects_empty_text(text: str) -> None:
    with pytest.raises(ValueError, match="không được để trống"):
        asr_tts_service.text_to_speech(text)


def test_text_to_speech_uses_vietnamese_gtts_and_creates_nonempty_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class FakeGTTS:
        def __init__(self, text: str, lang: str) -> None:
            calls.append((text, lang))

        def save(self, output_path: str) -> None:
            Path(output_path).write_bytes(b"fake mp3")

    monkeypatch.setattr(asr_tts_service, "gTTS", FakeGTTS)

    output = Path(asr_tts_service.text_to_speech("  Xin chào  "))

    assert calls == [("Xin chào", "vi")]
    assert output.is_absolute()
    assert output.parent == asr_tts_service.GENERATED_AUDIO_DIR.resolve()
    assert output.stat().st_size > 0


def test_text_to_speech_generates_unique_filenames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGTTS:
        def __init__(self, text: str, lang: str) -> None:
            pass

        def save(self, output_path: str) -> None:
            Path(output_path).write_bytes(b"mp3")

    monkeypatch.setattr(asr_tts_service, "gTTS", FakeGTTS)

    first = asr_tts_service.text_to_speech("Một")
    second = asr_tts_service.text_to_speech("Hai")

    assert first != second
    assert Path(first).exists()
    assert Path(second).exists()


def test_text_to_speech_removes_partial_file_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingGTTS:
        def __init__(self, text: str, lang: str) -> None:
            pass

        def save(self, output_path: str) -> None:
            Path(output_path).write_bytes(b"partial")
            raise OSError("network failed")

    monkeypatch.setattr(asr_tts_service, "gTTS", FailingGTTS)

    with pytest.raises(asr_tts_service.TTSServiceError):
        asr_tts_service.text_to_speech("Xin chào")

    assert list(asr_tts_service.GENERATED_AUDIO_DIR.glob("*.mp3")) == []
