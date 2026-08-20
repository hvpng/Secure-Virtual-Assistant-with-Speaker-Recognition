"""Frozen M4 response contracts."""

from typing import Literal

from pydantic import BaseModel


class ChatResponse(BaseModel):
    success: bool
    text_asr: str
    function_called: str | None
    auth_type: Literal["SV", "SID"] | None
    auth_passed: bool | None
    employee_id: str | None
    speaker_score: float | None
    response_text: str
    audio_reply_url: str | None


class QualityChecksResponse(BaseModel):
    duration_ok: bool
    speech_ratio_ok: bool
    snr_ok: bool
    clipping_ok: bool
    content_match_ok: bool


class FailedEnrollmentItem(BaseModel):
    index: int
    checks: QualityChecksResponse
    reasons: list[str]


class EnrollmentResponse(BaseModel):
    success: bool
    failed_items: list[FailedEnrollmentItem]


class EmployeeListItem(BaseModel):
    id: str
    name: str
    voice_enrolled: bool


class EmployeeListResponse(BaseModel):
    employees: list[EmployeeListItem]


class EnrollmentScriptItem(BaseModel):
    index: int
    text: str


class EnrollmentScriptsResponse(BaseModel):
    scripts: list[EnrollmentScriptItem]


class VoiceProfileResponse(BaseModel):
    success: bool
    employee_id: str
    voice_enrolled: bool
