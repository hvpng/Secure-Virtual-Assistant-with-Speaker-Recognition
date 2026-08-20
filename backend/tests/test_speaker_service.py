from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services import speaker_service
from tests.audio_test_utils import write_wav


PASSING_QUALITY = {
    "pass": True,
    "checks": {
        "duration_ok": True,
        "speech_ratio_ok": True,
        "snr_ok": True,
        "clipping_ok": True,
        "content_match_ok": True,
    },
    "metrics": {
        "duration_sec": 2.0,
        "speech_ratio": 1.0,
        "snr_db": 20.0,
        "clipping_ratio": 0.0,
        "content_wer": 0.0,
    },
    "reasons": [],
}


@pytest.fixture(autouse=True)
def isolated_fake_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    profiles = tmp_path / "profiles"
    monkeypatch.setattr(speaker_service, "VOICE_PROFILES_DIR", profiles)
    monkeypatch.setenv("SPEAKER_BACKEND", "fake")
    speaker_service._load_backend.cache_clear()
    return profiles


def _enroll_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, employee_id: str
) -> Path:
    monkeypatch.setattr(
        speaker_service, "check_audio_quality", lambda *args, **kwargs: PASSING_QUALITY
    )
    recording = write_wav(tmp_path / f"{employee_id}_enroll.wav")
    result = speaker_service.enroll_user(
        employee_id, [str(recording)], ["xin chào"]
    )
    assert result == {"success": True, "failed_items": []}
    return recording


def test_too_short_audio_fails_duration(tmp_path: Path) -> None:
    audio = write_wav(tmp_path / "short.wav", duration_sec=0.2)

    result = speaker_service.check_audio_quality(str(audio), "")

    assert result["checks"]["duration_ok"] is False
    assert result["pass"] is False
    assert "Âm thanh quá ngắn." in result["reasons"]


def test_silent_audio_fails_speech_ratio(tmp_path: Path) -> None:
    audio = write_wav(tmp_path / "silent.wav", silent=True)

    result = speaker_service.check_audio_quality(str(audio), "")

    assert result["checks"]["speech_ratio_ok"] is False
    assert result["pass"] is False
    assert result["metrics"]["snr_db"] == 0.0
    assert np.isfinite(result["metrics"]["snr_db"])


def test_expected_content_without_transcriber_fails_closed(tmp_path: Path) -> None:
    audio = write_wav(tmp_path / "missing_transcriber.wav")

    result = speaker_service.check_audio_quality(
        str(audio), "Tôi xác nhận giọng nói"
    )

    assert result["checks"]["content_match_ok"] is False
    assert result["pass"] is False
    assert (
        "Chưa có dịch vụ nhận dạng giọng nói để kiểm tra nội dung."
        in result["reasons"]
    )


def test_content_mismatch_fails_with_injected_transcriber(
    tmp_path: Path,
) -> None:
    audio = write_wav(tmp_path / "content.wav")

    result = speaker_service.check_audio_quality(
        str(audio), "Tôi xác nhận giọng nói", lambda _: "Nội dung hoàn toàn khác"
    )

    assert result["checks"]["content_match_ok"] is False
    assert result["metrics"]["content_wer"] > 0
    assert "Nội dung đọc không khớp câu yêu cầu." in result["reasons"]


def test_good_audio_passes_with_injected_transcriber(
    tmp_path: Path,
) -> None:
    audio = write_wav(tmp_path / "good.wav")

    result = speaker_service.check_audio_quality(
        str(audio),
        "Tôi xác nhận giọng nói của mình.",
        lambda normalized_path: "tôi xác nhận giọng nói của mình",
    )

    assert result["pass"] is True
    assert all(result["checks"].values())


def test_vad_snr_is_finite_and_clean_exceeds_noisy(tmp_path: Path) -> None:
    clean = write_wav(tmp_path / "clean.wav")
    noisy = write_wav(tmp_path / "noisy.wav", noise_amplitude=0.18)

    clean_result = speaker_service.check_audio_quality(str(clean), "")
    noisy_result = speaker_service.check_audio_quality(str(noisy), "")
    clean_snr = clean_result["metrics"]["snr_db"]
    noisy_snr = noisy_result["metrics"]["snr_db"]

    assert np.isfinite(clean_snr)
    assert np.isfinite(noisy_snr)
    assert clean_snr > noisy_snr


def test_enroll_success_creates_numpy_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enroll_identity(tmp_path, monkeypatch, "alice")

    profile_path = speaker_service.VOICE_PROFILES_DIR / "alice.npy"
    profile = np.load(profile_path, allow_pickle=False)
    assert profile.shape == (64,)
    assert profile.dtype == np.float32
    assert np.linalg.norm(profile) == pytest.approx(1.0)


def test_failed_enrollment_creates_no_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings = [
        write_wav(tmp_path / "alice_1.wav"),
        write_wav(tmp_path / "alice_2.wav"),
    ]

    def quality(path: str, *_: object, **__: object) -> dict[str, object]:
        if path.endswith("alice_2.wav"):
            return {
                **PASSING_QUALITY,
                "pass": False,
                "checks": {**PASSING_QUALITY["checks"], "duration_ok": False},
                "reasons": ["Âm thanh quá ngắn."],
            }
        return PASSING_QUALITY

    monkeypatch.setattr(speaker_service, "check_audio_quality", quality)
    result = speaker_service.enroll_user(
        "alice", [str(path) for path in recordings], ["một", "hai"]
    )

    assert result["success"] is False
    assert result["failed_items"][0]["index"] == 1
    assert not (speaker_service.VOICE_PROFILES_DIR / "alice.npy").exists()


def test_failed_reenrollment_preserves_existing_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enroll_identity(tmp_path, monkeypatch, "alice")
    profile_path = speaker_service.VOICE_PROFILES_DIR / "alice.npy"
    original = profile_path.read_bytes()
    monkeypatch.setattr(
        speaker_service,
        "check_audio_quality",
        lambda *args, **kwargs: {
            **PASSING_QUALITY,
            "pass": False,
            "checks": {**PASSING_QUALITY["checks"], "snr_ok": False},
            "reasons": ["Âm thanh quá nhiễu."],
        },
    )
    replacement = write_wav(tmp_path / "alice_replacement.wav")

    result = speaker_service.enroll_user(
        "alice", [str(replacement)], ["thay thế"]
    )

    assert result["success"] is False
    assert profile_path.read_bytes() == original


def test_verify_same_identity_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enroll_identity(tmp_path, monkeypatch, "alice")
    query = write_wav(tmp_path / "alice_query.wav")

    result = speaker_service.verify(str(query), "alice")

    assert result["is_match"] is True
    assert result["score"] == pytest.approx(1.0)


def test_verify_wrong_identity_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enroll_identity(tmp_path, monkeypatch, "alice")
    query = write_wav(tmp_path / "bob_query.wav")

    result = speaker_service.verify(str(query), "alice")

    assert result["is_match"] is False
    assert result["score"] < 0.8


def test_verify_missing_profile_fails_cleanly(tmp_path: Path) -> None:
    query = write_wav(tmp_path / "alice_query.wav")

    assert speaker_service.verify(str(query), "missing") == {
        "is_match": False,
        "score": 0.0,
    }


def test_identify_known_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enroll_identity(tmp_path, monkeypatch, "alice")
    _enroll_identity(tmp_path, monkeypatch, "bob")
    query = write_wav(tmp_path / "bob_query.wav")

    result = speaker_service.identify(str(query))

    assert result["employee_id"] == "bob"
    assert result["score"] == pytest.approx(1.0)


def test_identify_unknown_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enroll_identity(tmp_path, monkeypatch, "alice")
    query = write_wav(tmp_path / "unknown_query.wav")

    result = speaker_service.identify(str(query))

    assert result["employee_id"] is None
    assert result["score"] < 0.8


def test_identify_without_profiles_returns_none(tmp_path: Path) -> None:
    query = write_wav(tmp_path / "unknown_query.wav")

    assert speaker_service.identify(str(query)) == {
        "employee_id": None,
        "score": 0.0,
    }


@pytest.mark.parametrize(
    "employee_id", ["../alice", "alice/bob", "alice\\bob", "", ".hidden"]
)
def test_invalid_employee_id_is_rejected(employee_id: str) -> None:
    with pytest.raises(ValueError, match="employee_id"):
        speaker_service.has_voice_profile(employee_id)


def test_identify_skips_corrupt_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enroll_identity(tmp_path, monkeypatch, "alice")
    profiles = speaker_service.VOICE_PROFILES_DIR
    (profiles / "corrupt.npy").write_bytes(b"not a numpy file")
    query = write_wav(tmp_path / "alice_query.wav")

    result = speaker_service.identify(str(query))

    assert result["employee_id"] == "alice"


def test_real_backend_without_module_a_artifact_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = write_wav(tmp_path / "alice_query.wav")
    monkeypatch.setenv("SPEAKER_BACKEND", "real")
    speaker_service._load_backend.cache_clear()

    with pytest.raises(
        speaker_service.SpeakerBackendUnavailableError,
        match="artifact Module A chưa sẵn sàng",
    ):
        speaker_service.extract_embedding(str(audio))


def test_thresholds_are_loaded_from_canonical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "speaker_config.json"
    config_path.write_text(
        json.dumps(
            {
                "development_placeholder": True,
                "sv_threshold": 0.61,
                "sid_threshold": 0.72,
                "embedding_dimension": 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(speaker_service, "SPEAKER_CONFIG_PATH", config_path)

    thresholds = speaker_service.load_speaker_thresholds()

    assert thresholds.sv_threshold == 0.61
    assert thresholds.sid_threshold == 0.72
