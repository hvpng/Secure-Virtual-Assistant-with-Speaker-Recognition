from __future__ import annotations

from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api import routes
from app.core.enrollment_scripts import ENROLLMENT_TEXTS
from app.db.database import Base, Employee, get_db
from app.main import app
from app.services import asr_tts_service, nlu_service, speaker_service


class ApiContext(SimpleNamespace):
    client: TestClient
    session_factory: sessionmaker[Session]
    generated_dir: Path

    def employee(self, employee_id: str) -> Employee | None:
        with self.session_factory() as db:
            return db.get(Employee, employee_id)


def _employee(
    employee_id: str,
    name: str,
    leave_days: int,
    meetings: list[str],
    enrolled: bool = True,
) -> Employee:
    return Employee(
        id=employee_id,
        name=name,
        leave_days_left=leave_days,
        meetings_today=meetings,
        salary_mock=f"{employee_id}-salary",
        insurance_status=f"{employee_id}-insurance",
        password_hash_mock=f"{employee_id}-old-hash",
        voice_enrolled=enrolled,
    )


@pytest.fixture
def api_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ApiContext:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with testing_session() as db:
        db.add_all(
            [
                _employee("NV001", "Alice", 12, ["09:00 Alice meeting"]),
                _employee("NV002", "Bob", 3, ["15:00 Bob meeting"]),
                _employee("NV003", "Carol", 8, [], enrolled=False),
            ]
        )
        db.commit()

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    generated_dir = tmp_path / "generated_audio"
    generated_dir.mkdir()
    monkeypatch.setattr(asr_tts_service, "GENERATED_AUDIO_DIR", generated_dir)
    sequence = count(1)

    def fake_tts(text: str) -> str:
        output = generated_dir / f"tts_{next(sequence)}.mp3"
        output.write_bytes(f"mp3:{text}".encode())
        return str(output)

    monkeypatch.setattr(asr_tts_service, "text_to_speech", fake_tts)
    monkeypatch.setattr(asr_tts_service, "speech_to_text", lambda path: "mock transcript")
    monkeypatch.setattr(speaker_service, "has_voice_profile", lambda employee_id: True)

    context = ApiContext(
        client=TestClient(app),
        session_factory=testing_session,
        generated_dir=generated_dir,
    )
    yield context
    app.dependency_overrides.clear()
    engine.dispose()


def _chat(
    context: ApiContext,
    claimed_employee_id: str | None = None,
    content: bytes = b"fake wav",
) -> Any:
    data = {}
    if claimed_employee_id is not None:
        data["claimed_employee_id"] = claimed_employee_id
    return context.client.post(
        "/api/chat",
        data=data,
        files={"audio": ("command.wav", content, "audio/wav")},
    )


def _seven_audio_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("audio_files", (f"sample_{index}.wav", b"audio", "audio/wav"))
        for index in range(7)
    ]


def _intent(function_name: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
    return {"function_name": function_name, "arguments": arguments or {}}


@pytest.mark.parametrize("claimed", [None, "NV002"])
def test_general_faq_never_calls_speaker_auth(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
    claimed: str | None,
) -> None:
    monkeypatch.setattr(
        nlu_service, "parse_intent", lambda text: _intent("answer_faq", {"topic": "vpn"})
    )
    monkeypatch.setattr(
        speaker_service, "verify", lambda *args: pytest.fail("verify must not run")
    )
    monkeypatch.setattr(
        speaker_service, "identify", lambda *args: pytest.fail("identify must not run")
    )

    response = _chat(api_context, claimed)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["auth_type"] is None
    assert body["auth_passed"] is None
    assert body["employee_id"] is None
    assert body["speaker_score"] is None
    assert body["audio_reply_url"].startswith("/api/audio/")


def test_sv_missing_claim_denies_before_business(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("reset_password"))
    monkeypatch.setattr(
        routes,
        "execute_business_function",
        lambda *args: pytest.fail("business must not run"),
    )

    response = _chat(api_context)

    assert response.status_code == 200
    assert response.json()["auth_passed"] is False
    assert response.json()["employee_id"] is None


def test_sv_unknown_employee_denies(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("reset_password"))

    response = _chat(api_context, "NV999")

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert response.json()["auth_passed"] is False


@pytest.mark.parametrize(
    ("voice_enrolled", "profile_exists"), [(False, True), (True, False)]
)
def test_sv_requires_enrolled_db_state_and_profile(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
    voice_enrolled: bool,
    profile_exists: bool,
) -> None:
    with api_context.session_factory() as db:
        db.get(Employee, "NV001").voice_enrolled = voice_enrolled
        db.commit()
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("reset_password"))
    monkeypatch.setattr(speaker_service, "has_voice_profile", lambda employee_id: profile_exists)
    monkeypatch.setattr(
        speaker_service, "verify", lambda *args: pytest.fail("verify must not run")
    )

    response = _chat(api_context, "NV001")

    assert response.status_code == 200
    assert response.json()["auth_passed"] is False


def test_sv_verify_false_has_zero_business_side_effect(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("reset_password"))
    monkeypatch.setattr(
        speaker_service, "verify", lambda path, employee_id: {"is_match": False, "score": 0.21}
    )
    before = api_context.employee("NV001").password_hash_mock

    response = _chat(api_context, "NV001")

    assert response.json()["auth_passed"] is False
    assert response.json()["speaker_score"] == pytest.approx(0.21)
    assert api_context.employee("NV001").password_hash_mock == before


def test_sv_verify_true_binds_claimed_employee_and_executes_once(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("reset_password"))
    verify_calls: list[str] = []

    def verify(path: str, employee_id: str) -> dict[str, object]:
        verify_calls.append(employee_id)
        return {"is_match": True, "score": 0.93}

    monkeypatch.setattr(speaker_service, "verify", verify)

    response = _chat(api_context, "NV001")

    body = response.json()
    assert body["success"] is True
    assert body["auth_type"] == "SV"
    assert body["auth_passed"] is True
    assert body["employee_id"] == "NV001"
    assert verify_calls == ["NV001"]
    assert api_context.employee("NV001").password_hash_mock == "mock-reset-required-on-next-login"
    assert api_context.employee("NV002").password_hash_mock == "NV002-old-hash"


def test_malicious_llm_identity_argument_is_rejected_before_auth_or_business(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        nlu_service,
        "parse_intent",
        lambda text: _intent("reset_password", {"employee_id": "NV002"}),
    )
    monkeypatch.setattr(
        speaker_service, "verify", lambda *args: pytest.fail("auth must not run")
    )

    response = _chat(api_context, "NV001")

    assert response.status_code == 503
    assert api_context.employee("NV001").password_hash_mock == "NV001-old-hash"
    assert api_context.employee("NV002").password_hash_mock == "NV002-old-hash"


def test_salary_insurance_obeys_sv_and_uses_claimed_employee(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        nlu_service, "parse_intent", lambda text: _intent("check_salary_insurance")
    )
    monkeypatch.setattr(
        speaker_service, "verify", lambda path, employee_id: {"is_match": True, "score": 0.9}
    )

    response = _chat(api_context, "NV001")

    assert "NV001-salary" in response.json()["response_text"]
    assert "NV002-salary" not in response.json()["response_text"]


def test_sid_none_denies_without_business(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("check_leave_days"))
    monkeypatch.setattr(
        speaker_service, "identify", lambda path: {"employee_id": None, "score": 0.2}
    )

    response = _chat(api_context)

    body = response.json()
    assert body["success"] is False
    assert body["auth_passed"] is False
    assert body["employee_id"] is None
    assert "12" not in body["response_text"]


def test_sid_unknown_db_employee_denies_controlled(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("check_leave_days"))
    monkeypatch.setattr(
        speaker_service,
        "identify",
        lambda path: {"employee_id": "NV999", "score": 0.91},
    )

    response = _chat(api_context)

    assert response.status_code == 200
    assert response.json()["auth_passed"] is False
    assert response.json()["employee_id"] is None


@pytest.mark.parametrize(
    ("identified_id", "expected_days"), [("NV001", 12), ("NV002", 3)]
)
def test_sid_leave_days_uses_identified_employee(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
    identified_id: str,
    expected_days: int,
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("check_leave_days"))
    monkeypatch.setattr(
        speaker_service,
        "identify",
        lambda path: {"employee_id": identified_id, "score": 0.94},
    )

    response = _chat(api_context, claimed_employee_id="NV999")

    assert response.json()["employee_id"] == identified_id
    assert response.json()["auth_passed"] is True
    assert str(expected_days) in response.json()["response_text"]


def test_sid_meetings_are_personalized_to_identified_employee(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        nlu_service, "parse_intent", lambda text: _intent("check_today_meetings")
    )
    monkeypatch.setattr(
        speaker_service, "identify", lambda path: {"employee_id": "NV002", "score": 0.88}
    )

    response = _chat(api_context, claimed_employee_id="NV001")

    assert response.json()["employee_id"] == "NV002"
    assert "Bob meeting" in response.json()["response_text"]
    assert "Alice meeting" not in response.json()["response_text"]


@pytest.mark.parametrize("auth_kind", ["verify", "identify"])
def test_speaker_exception_denies_and_never_executes_business(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
    auth_kind: str,
) -> None:
    function_name = "reset_password" if auth_kind == "verify" else "check_leave_days"
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent(function_name))
    monkeypatch.setattr(
        speaker_service,
        auth_kind,
        lambda *args: (_ for _ in ()).throw(RuntimeError("speaker failure")),
    )
    before = api_context.employee("NV001").password_hash_mock

    response = _chat(api_context, "NV001")

    assert response.status_code == 200
    assert response.json()["auth_passed"] is False
    assert api_context.employee("NV001").password_hash_mock == before


def test_unknown_intent_fails_closed(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("unknown"))

    response = _chat(api_context)

    assert response.status_code == 503


def test_auth_policy_business_mismatch_fails_closed(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nlu_service, "parse_intent", lambda text: _intent("reset_password"))
    policy = dict(nlu_service.AUTH_REQUIREMENT)
    policy.pop("reset_password")
    monkeypatch.setattr(nlu_service, "AUTH_REQUIREMENT", policy)

    response = _chat(api_context, "NV001")

    assert response.status_code == 503
    assert api_context.employee("NV001").password_hash_mock == "NV001-old-hash"


def test_asr_failure_cleans_temp_and_skips_nlu(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def fail_asr(path: str) -> str:
        captured.append(path)
        assert Path(path).exists()
        raise asr_tts_service.ASRTranscriptionError("asr failed")

    monkeypatch.setattr(asr_tts_service, "speech_to_text", fail_asr)
    monkeypatch.setattr(
        nlu_service, "parse_intent", lambda text: pytest.fail("NLU must not run")
    )

    response = _chat(api_context)

    assert response.status_code == 503
    assert captured and not Path(captured[0]).exists()


def test_nlu_failure_skips_auth_and_business(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        nlu_service,
        "parse_intent",
        lambda text: (_ for _ in ()).throw(nlu_service.NLUResponseError("bad")),
    )
    monkeypatch.setattr(
        speaker_service, "verify", lambda *args: pytest.fail("speaker must not run")
    )

    response = _chat(api_context, "NV001")

    assert response.status_code == 503


def test_chat_temp_upload_cleanup_after_success(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def asr(path: str) -> str:
        captured.append(path)
        assert Path(path).is_file()
        return "vpn"

    monkeypatch.setattr(asr_tts_service, "speech_to_text", asr)
    monkeypatch.setattr(
        nlu_service, "parse_intent", lambda text: _intent("answer_faq", {"topic": "vpn"})
    )

    assert _chat(api_context).status_code == 200
    assert captured and not Path(captured[0]).exists()


def test_tts_failure_keeps_text_and_does_not_retry_business(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        nlu_service, "parse_intent", lambda text: _intent("answer_faq", {"topic": "vpn"})
    )
    monkeypatch.setattr(
        asr_tts_service,
        "text_to_speech",
        lambda text: (_ for _ in ()).throw(asr_tts_service.TTSServiceError("tts")),
    )
    calls = 0
    real_execute = routes.execute_business_function

    def counted(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(routes, "execute_business_function", counted)

    response = _chat(api_context)

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["response_text"]
    assert response.json()["audio_reply_url"] is None
    assert calls == 1


def test_enroll_success_uses_server_scripts_and_sets_voice_enrolled(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def enroll_user(
        employee_id: str,
        paths: list[str],
        expected_texts: list[str],
        transcribe_fn: object,
    ) -> dict[str, object]:
        captured.update(
            employee_id=employee_id,
            paths=list(paths),
            expected_texts=expected_texts,
            transcribe_fn=transcribe_fn,
        )
        assert all(Path(path).exists() for path in paths)
        return {"success": True, "failed_items": []}

    monkeypatch.setattr(speaker_service, "enroll_user", enroll_user)

    response = api_context.client.post(
        "/api/enroll",
        data={"employee_id": "NV100", "name": "New User", "expected_text": "malicious"},
        files=_seven_audio_files(),
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "failed_items": []}
    assert captured["expected_texts"] == list(ENROLLMENT_TEXTS)
    assert captured["transcribe_fn"] is asr_tts_service.speech_to_text
    assert all(not Path(path).exists() for path in captured["paths"])
    employee = api_context.employee("NV100")
    assert employee is not None and employee.voice_enrolled is True


def test_enroll_quality_failure_keeps_employee_state_false_and_cleans_temp(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_paths: list[str] = []
    failure = {
        "success": False,
        "failed_items": [
            {
                "index": 2,
                "checks": {
                    "duration_ok": True,
                    "speech_ratio_ok": False,
                    "snr_ok": False,
                    "clipping_ok": True,
                    "content_match_ok": True,
                },
                "reasons": ["Âm thanh quá nhiễu."],
            }
        ],
    }

    def fail_enroll(employee_id: str, paths: list[str], *args: object, **kwargs: object):
        captured_paths.extend(paths)
        return failure

    monkeypatch.setattr(speaker_service, "enroll_user", fail_enroll)

    response = api_context.client.post(
        "/api/enroll",
        data={"employee_id": "NV003", "name": "Carol"},
        files=_seven_audio_files(),
    )

    assert response.status_code == 400
    assert response.json() == failure
    assert api_context.employee("NV003").voice_enrolled is False
    assert all(not Path(path).exists() for path in captured_paths)


def test_enroll_wrong_file_count_rejects_before_speaker(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        speaker_service, "enroll_user", lambda *args: pytest.fail("enroll must not run")
    )

    response = api_context.client.post(
        "/api/enroll",
        data={"employee_id": "NV100", "name": "New User"},
        files=_seven_audio_files()[:1],
    )

    assert response.status_code == 400


def test_reenroll_failure_preserves_existing_state(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = {
        "success": False,
        "failed_items": [
            {
                "index": 0,
                "checks": {
                    "duration_ok": False,
                    "speech_ratio_ok": True,
                    "snr_ok": True,
                    "clipping_ok": True,
                    "content_match_ok": True,
                },
                "reasons": ["Âm thanh quá ngắn."],
            }
        ],
    }
    monkeypatch.setattr(speaker_service, "enroll_user", lambda *args, **kwargs: failure)

    response = api_context.client.post(
        "/api/employees/NV001/reenroll", files=_seven_audio_files()
    )

    assert response.status_code == 400
    assert api_context.employee("NV001").voice_enrolled is True


def test_reenroll_success_and_nonexistent_employee_policy(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        speaker_service,
        "enroll_user",
        lambda *args, **kwargs: {"success": True, "failed_items": []},
    )

    success = api_context.client.post(
        "/api/employees/NV001/reenroll", files=_seven_audio_files()
    )
    missing = api_context.client.post(
        "/api/employees/NV999/reenroll", files=_seven_audio_files()
    )

    assert success.status_code == 200
    assert success.json()["success"] is True
    assert missing.status_code == 404


def test_employee_list_exposes_only_safe_fields(api_context: ApiContext) -> None:
    response = api_context.client.get("/api/employees")

    assert response.status_code == 200
    assert response.json()["employees"]
    for item in response.json()["employees"]:
        assert set(item) == {"id", "name", "voice_enrolled"}


def test_delete_profile_keeps_employee_and_sets_flag_false(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    delete_calls: list[str] = []

    def delete(employee_id: str) -> bool:
        delete_calls.append(employee_id)
        return True

    monkeypatch.setattr(speaker_service, "delete_voice_profile", delete)

    response = api_context.client.delete("/api/employees/NV001/voice-profile")

    assert response.status_code == 200
    assert delete_calls == ["NV001"]
    assert api_context.employee("NV001") is not None
    assert api_context.employee("NV001").voice_enrolled is False


def test_delete_nonexistent_profile_repairs_flag_and_returns_404(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(speaker_service, "has_voice_profile", lambda employee_id: False)

    response = api_context.client.delete("/api/employees/NV001/voice-profile")

    assert response.status_code == 404
    assert api_context.employee("NV001").voice_enrolled is False


def test_generated_audio_serving_and_path_traversal_protection(
    api_context: ApiContext,
) -> None:
    audio = api_context.generated_dir / "safe.mp3"
    audio.write_bytes(b"mp3")
    outside = api_context.generated_dir.parent / "secret.mp3"
    outside.write_bytes(b"secret")

    safe_response = api_context.client.get("/api/audio/safe.mp3")
    traversal = api_context.client.get("/api/audio/..%2Fsecret.mp3")

    assert safe_response.status_code == 200
    assert safe_response.headers["content-type"].startswith("audio/mpeg")
    assert safe_response.content == b"mp3"
    assert traversal.status_code == 404
    assert traversal.content != b"secret"


def test_enrollment_scripts_contract(api_context: ApiContext) -> None:
    response = api_context.client.get("/api/enrollment-scripts")

    assert response.status_code == 200
    assert response.json() == {
        "scripts": [
            {"index": index, "text": text}
            for index, text in enumerate(ENROLLMENT_TEXTS)
        ]
    }


@pytest.mark.parametrize(
    ("filename", "content"), [("empty.wav", b""), ("payload.exe", b"bad")]
)
def test_invalid_upload_is_rejected(
    api_context: ApiContext, filename: str, content: bytes
) -> None:
    response = api_context.client.post(
        "/api/chat",
        files={"audio": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == 400
