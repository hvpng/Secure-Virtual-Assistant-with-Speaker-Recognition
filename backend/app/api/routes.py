"""M4 FastAPI routes and hardcoded authentication orchestration."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.schemas import (
    ChatResponse,
    EmployeeListItem,
    EmployeeListResponse,
    EnrollmentResponse,
    EnrollmentScriptsResponse,
    VoiceProfileResponse,
)
from app.core.enrollment_scripts import ENROLLMENT_TEXTS, enrollment_scripts_response
from app.db.database import Employee, get_db
from app.services import asr_tts_service, nlu_service, speaker_service
from app.services.business_service import (
    BUSINESS_FUNCTIONS,
    BusinessServiceError,
    execute_business_function,
)
from app.utils.audio_utils import AudioNormalizationError


logger = logging.getLogger(__name__)
router = APIRouter()

MAX_AUDIO_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".webm", ".mp3", ".ogg", ".m4a", ".mp4", ".flac"}
EMPLOYEE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
AUDIO_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.mp3$")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        await upload.close()
        raise HTTPException(status_code=400, detail="Định dạng audio không được hỗ trợ.")

    temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temporary_path = Path(temporary.name)
    size = 0
    try:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_AUDIO_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="File audio vượt quá giới hạn 20 MB.",
                )
            temporary.write(chunk)
        temporary.flush()
    except Exception:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.close()
        await upload.close()

    if size == 0:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File audio không được rỗng.")
    return str(temporary_path)


def _cleanup_temp_paths(paths: list[str]) -> None:
    for path in paths:
        Path(path).unlink(missing_ok=True)


def _validate_employee_id(employee_id: str) -> str:
    normalized = employee_id.strip()
    if not EMPLOYEE_ID_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="employee_id không hợp lệ.")
    return normalized


def _validate_name(name: str) -> str:
    normalized = " ".join(name.split())
    if not normalized or len(normalized) > 120:
        raise HTTPException(status_code=400, detail="Tên nhân viên không hợp lệ.")
    return normalized


def _safe_audio_url(filesystem_path: str) -> str | None:
    try:
        generated_dir = asr_tts_service.GENERATED_AUDIO_DIR.resolve()
        path = Path(filesystem_path).resolve(strict=True)
        if path.parent != generated_dir or not AUDIO_FILENAME_PATTERN.fullmatch(path.name):
            logger.error("TTS returned a path outside the generated-audio boundary.")
            return None
        if path.stat().st_size <= 0:
            return None
        return f"/api/audio/{path.name}"
    except (OSError, RuntimeError):
        return None


def _try_tts(response_text: str) -> str | None:
    try:
        return _safe_audio_url(asr_tts_service.text_to_speech(response_text))
    except Exception as exc:
        logger.warning("TTS response generation failed (%s).", type(exc).__name__)
        return None


def _chat_response(
    *,
    success: bool,
    text_asr: str,
    function_called: str | None,
    auth_type: str | None,
    auth_passed: bool | None,
    employee_id: str | None,
    speaker_score: float | None,
    response_text: str,
) -> ChatResponse:
    return ChatResponse(
        success=success,
        text_asr=text_asr,
        function_called=function_called,
        auth_type=auth_type,
        auth_passed=auth_passed,
        employee_id=employee_id,
        speaker_score=speaker_score,
        response_text=response_text,
        audio_reply_url=_try_tts(response_text),
    )


def _validated_intent(intent: object) -> tuple[str, dict[str, Any], str | None]:
    if not isinstance(intent, dict):
        raise HTTPException(status_code=503, detail="NLU trả response không hợp lệ.")
    function_name = intent.get("function_name")
    arguments = intent.get("arguments")
    if not isinstance(function_name, str) or not isinstance(arguments, dict):
        raise HTTPException(status_code=503, detail="NLU trả intent không hợp lệ.")
    if (
        function_name not in nlu_service.AUTH_REQUIREMENT
        or function_name not in BUSINESS_FUNCTIONS
    ):
        raise HTTPException(status_code=503, detail="Intent không nằm trong allowlist.")
    if function_name == "answer_faq":
        if set(arguments) != {"topic"} or not isinstance(arguments["topic"], str):
            raise HTTPException(status_code=503, detail="FAQ arguments không hợp lệ.")
    elif arguments:
        raise HTTPException(
            status_code=503,
            detail="Self-service intent không được chứa arguments.",
        )
    return function_name, arguments, nlu_service.AUTH_REQUIREMENT[function_name]


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    audio: UploadFile = File(...),
    claimed_employee_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> ChatResponse:
    temp_path: str | None = None
    try:
        temp_path = await _save_upload(audio)
        try:
            text_asr = asr_tts_service.speech_to_text(temp_path)
        except AudioNormalizationError as exc:
            raise HTTPException(status_code=400, detail="Audio không hợp lệ.") from exc
        except asr_tts_service.ASRServiceError as exc:
            raise HTTPException(status_code=503, detail="Dịch vụ ASR không khả dụng.") from exc

        try:
            intent = nlu_service.parse_intent(text_asr)
        except nlu_service.NLUServiceError as exc:
            raise HTTPException(status_code=503, detail="Dịch vụ NLU không khả dụng.") from exc

        function_name, arguments, auth_type = _validated_intent(intent)
        authenticated_employee_id: str | None = None
        speaker_score: float | None = None

        if auth_type is None:
            auth_passed: bool | None = None
        elif auth_type == "SV":
            if claimed_employee_id is None or not claimed_employee_id.strip():
                return _chat_response(
                    success=False,
                    text_asr=text_asr,
                    function_called=function_name,
                    auth_type="SV",
                    auth_passed=False,
                    employee_id=None,
                    speaker_score=None,
                    response_text="Vui lòng cung cấp mã nhân viên để xác thực giọng nói.",
                )
            claimed_id = _validate_employee_id(claimed_employee_id)
            employee = db.get(Employee, claimed_id)
            if employee is None:
                return _chat_response(
                    success=False,
                    text_asr=text_asr,
                    function_called=function_name,
                    auth_type="SV",
                    auth_passed=False,
                    employee_id=claimed_id,
                    speaker_score=None,
                    response_text="Không tìm thấy hồ sơ nhân viên để xác thực.",
                )
            try:
                profile_exists = speaker_service.has_voice_profile(claimed_id)
            except Exception:
                profile_exists = False
            if not employee.voice_enrolled or not profile_exists:
                return _chat_response(
                    success=False,
                    text_asr=text_asr,
                    function_called=function_name,
                    auth_type="SV",
                    auth_passed=False,
                    employee_id=claimed_id,
                    speaker_score=None,
                    response_text="Nhân viên chưa có hồ sơ giọng nói hợp lệ.",
                )
            try:
                verification = speaker_service.verify(temp_path, claimed_id)
                speaker_score = float(verification.get("score", 0.0))
                is_match = verification.get("is_match") is True
            except Exception as exc:
                logger.warning("Speaker verification failed (%s).", type(exc).__name__)
                is_match = False
            if not is_match:
                return _chat_response(
                    success=False,
                    text_asr=text_asr,
                    function_called=function_name,
                    auth_type="SV",
                    auth_passed=False,
                    employee_id=claimed_id,
                    speaker_score=speaker_score,
                    response_text="Xác thực giọng nói không thành công. Yêu cầu đã bị từ chối.",
                )
            authenticated_employee_id = claimed_id
            auth_passed = True
        elif auth_type == "SID":
            try:
                identification = speaker_service.identify(temp_path)
                identified_id = identification.get("employee_id")
                speaker_score = float(identification.get("score", 0.0))
            except Exception as exc:
                logger.warning("Speaker identification failed (%s).", type(exc).__name__)
                identified_id = None
            if not isinstance(identified_id, str):
                return _chat_response(
                    success=False,
                    text_asr=text_asr,
                    function_called=function_name,
                    auth_type="SID",
                    auth_passed=False,
                    employee_id=None,
                    speaker_score=speaker_score,
                    response_text="Không nhận diện được người dùng. Yêu cầu đã bị từ chối.",
                )
            employee = db.get(Employee, identified_id)
            if employee is None or not employee.voice_enrolled:
                return _chat_response(
                    success=False,
                    text_asr=text_asr,
                    function_called=function_name,
                    auth_type="SID",
                    auth_passed=False,
                    employee_id=None,
                    speaker_score=speaker_score,
                    response_text="Hồ sơ người dùng được nhận diện không hợp lệ.",
                )
            authenticated_employee_id = employee.id
            auth_passed = True
        else:
            raise HTTPException(status_code=503, detail="Auth policy không hợp lệ.")

        try:
            response_text = execute_business_function(
                db,
                function_name,
                authenticated_employee_id,
                arguments,
            )
            db.commit()
        except (BusinessServiceError, SQLAlchemyError) as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail="Business operation thất bại.") from exc

        return _chat_response(
            success=True,
            text_asr=text_asr,
            function_called=function_name,
            auth_type=auth_type,
            auth_passed=auth_passed,
            employee_id=authenticated_employee_id,
            speaker_score=speaker_score,
            response_text=response_text,
        )
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


@router.get("/api/enrollment-scripts", response_model=EnrollmentScriptsResponse)
def get_enrollment_scripts() -> EnrollmentScriptsResponse:
    return EnrollmentScriptsResponse(scripts=enrollment_scripts_response())


def _quality_failure_response(result: dict[str, object]) -> JSONResponse:
    return JSONResponse(status_code=400, content=result)


def _new_employee(employee_id: str, name: str) -> Employee:
    return Employee(
        id=employee_id,
        name=name,
        leave_days_left=0,
        meetings_today=[],
        salary_mock="Chưa cập nhật",
        insurance_status="Chưa cập nhật",
        password_hash_mock="mock-password-unset",
        voice_enrolled=True,
    )


@router.post("/api/enroll", response_model=EnrollmentResponse)
async def enroll(
    employee_id: str = Form(...),
    name: str = Form(...),
    audio_files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> EnrollmentResponse | JSONResponse:
    normalized_id = _validate_employee_id(employee_id)
    normalized_name = _validate_name(name)
    if len(audio_files) != len(ENROLLMENT_TEXTS):
        raise HTTPException(status_code=400, detail="Phải gửi đúng 7 file enrollment.")
    employee = db.get(Employee, normalized_id)
    if employee is not None and employee.name != normalized_name:
        raise HTTPException(status_code=409, detail="employee_id đã thuộc tên khác.")
    if employee is not None and employee.voice_enrolled:
        raise HTTPException(status_code=409, detail="Hãy dùng endpoint re-enroll.")

    temp_paths: list[str] = []
    profile_saved = False
    try:
        for upload in audio_files:
            temp_paths.append(await _save_upload(upload))
        try:
            result = speaker_service.enroll_user(
                normalized_id,
                temp_paths,
                list(ENROLLMENT_TEXTS),
                transcribe_fn=asr_tts_service.speech_to_text,
            )
        except (AudioNormalizationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Enrollment audio không hợp lệ.") from exc
        except speaker_service.SpeakerServiceError as exc:
            raise HTTPException(status_code=503, detail="Speaker service không khả dụng.") from exc
        if result.get("success") is not True:
            return _quality_failure_response(result)
        profile_saved = True

        if employee is None:
            employee = _new_employee(normalized_id, normalized_name)
            db.add(employee)
        else:
            employee.voice_enrolled = True
        try:
            db.commit()
        except SQLAlchemyError as exc:
            db.rollback()
            if profile_saved:
                try:
                    speaker_service.delete_voice_profile(normalized_id)
                except Exception:
                    logger.critical("Could not compensate profile after DB enrollment failure.")
            raise HTTPException(status_code=500, detail="Không lưu được trạng thái enrollment.") from exc
        return EnrollmentResponse(success=True, failed_items=[])
    finally:
        _cleanup_temp_paths(temp_paths)


@router.post("/api/employees/{employee_id}/reenroll", response_model=EnrollmentResponse)
async def reenroll(
    employee_id: str,
    audio_files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> EnrollmentResponse | JSONResponse:
    normalized_id = _validate_employee_id(employee_id)
    employee = db.get(Employee, normalized_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên.")
    if len(audio_files) != len(ENROLLMENT_TEXTS):
        raise HTTPException(status_code=400, detail="Phải gửi đúng 7 file enrollment.")
    try:
        has_profile = speaker_service.has_voice_profile(normalized_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Không kiểm tra được voice profile.") from exc
    if not employee.voice_enrolled or not has_profile:
        raise HTTPException(status_code=409, detail="Nhân viên chưa có profile để re-enroll.")

    temp_paths: list[str] = []
    try:
        for upload in audio_files:
            temp_paths.append(await _save_upload(upload))
        try:
            result = speaker_service.enroll_user(
                normalized_id,
                temp_paths,
                list(ENROLLMENT_TEXTS),
                transcribe_fn=asr_tts_service.speech_to_text,
            )
        except (AudioNormalizationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Re-enrollment audio không hợp lệ.") from exc
        except speaker_service.SpeakerServiceError as exc:
            raise HTTPException(status_code=503, detail="Speaker service không khả dụng.") from exc
        if result.get("success") is not True:
            return _quality_failure_response(result)
        return EnrollmentResponse(success=True, failed_items=[])
    finally:
        _cleanup_temp_paths(temp_paths)


@router.get("/api/employees", response_model=EmployeeListResponse)
def list_employees(db: Session = Depends(get_db)) -> EmployeeListResponse:
    employees = db.scalars(select(Employee).order_by(Employee.id)).all()
    return EmployeeListResponse(
        employees=[
            EmployeeListItem(
                id=employee.id,
                name=employee.name,
                voice_enrolled=employee.voice_enrolled,
            )
            for employee in employees
        ]
    )


@router.delete("/api/employees/{employee_id}/voice-profile", response_model=VoiceProfileResponse)
def remove_voice_profile(
    employee_id: str,
    db: Session = Depends(get_db),
) -> VoiceProfileResponse:
    normalized_id = _validate_employee_id(employee_id)
    employee = db.get(Employee, normalized_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên.")
    previous_state = employee.voice_enrolled
    try:
        profile_exists = speaker_service.has_voice_profile(normalized_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Không kiểm tra được voice profile.") from exc
    if not profile_exists:
        employee.voice_enrolled = False
        try:
            db.commit()
        except SQLAlchemyError as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail="Không đồng bộ được employee.") from exc
        raise HTTPException(status_code=404, detail="Voice profile không tồn tại.")

    employee.voice_enrolled = False
    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Không cập nhật được employee.") from exc
    try:
        deleted = speaker_service.delete_voice_profile(normalized_id)
        if not deleted:
            raise RuntimeError("Profile disappeared before deletion")
    except Exception as exc:
        employee.voice_enrolled = previous_state
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.critical("Could not compensate employee after profile deletion failure.")
        raise HTTPException(status_code=500, detail="Không xóa được voice profile.") from exc
    return VoiceProfileResponse(
        success=True,
        employee_id=normalized_id,
        voice_enrolled=False,
    )


@router.get("/api/audio/{filename}")
def get_generated_audio(filename: str) -> FileResponse:
    if not AUDIO_FILENAME_PATTERN.fullmatch(filename) or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Audio không tồn tại.")
    generated_dir = asr_tts_service.GENERATED_AUDIO_DIR.resolve()
    path = (generated_dir / filename).resolve()
    if path.parent != generated_dir or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio không tồn tại.")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)
