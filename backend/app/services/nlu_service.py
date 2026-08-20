"""Gemini function-calling intent parser with fail-closed validation."""

from __future__ import annotations

import logging
import threading
from typing import Any

from google import genai
from google.genai import types

from app.core.config import settings


logger = logging.getLogger(__name__)

AUTH_REQUIREMENT: dict[str, str | None] = {
    "answer_faq": None,
    "reset_password": "SV",
    "check_salary_insurance": "SV",
    "check_leave_days": "SID",
    "check_today_meetings": "SID",
}

SYSTEM_INSTRUCTION = """Bạn là bộ định tuyến intent cho IT Helpdesk tiếng Việt.
Chọn đúng một function được cung cấp và chỉ trích xuất semantic arguments theo schema.
Không thực thi action, không tạo identity và không quyết định authentication.
Luôn trả function call, không trả lời trực tiếp."""

_NO_ARGUMENTS_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

FUNCTION_DECLARATIONS: tuple[types.FunctionDeclaration, ...] = (
    types.FunctionDeclaration(
        name="answer_faq",
        description=(
            "Định tuyến câu hỏi FAQ IT/nội bộ chung như giờ làm việc, VPN, "
            "ngày nghỉ hoặc lịch nghỉ công ty."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Chủ đề FAQ ngắn gọn, ví dụ vpn hoặc giờ làm việc.",
                }
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    ),
    types.FunctionDeclaration(
        name="reset_password",
        description="Người dùng muốn reset hoặc đổi lại mật khẩu của chính họ.",
        parameters_json_schema=_NO_ARGUMENTS_SCHEMA,
    ),
    types.FunctionDeclaration(
        name="check_salary_insurance",
        description="Người dùng muốn xem lương hoặc thông tin bảo hiểm của chính họ.",
        parameters_json_schema=_NO_ARGUMENTS_SCHEMA,
    ),
    types.FunctionDeclaration(
        name="check_leave_days",
        description="Người dùng muốn xem số ngày phép còn lại của chính họ.",
        parameters_json_schema=_NO_ARGUMENTS_SCHEMA,
    ),
    types.FunctionDeclaration(
        name="check_today_meetings",
        description="Người dùng muốn xem lịch hoặc cuộc họp hôm nay của chính họ.",
        parameters_json_schema=_NO_ARGUMENTS_SCHEMA,
    ),
)

GEMINI_TOOL = types.Tool(function_declarations=list(FUNCTION_DECLARATIONS))

_client_lock = threading.Lock()
_client: genai.Client | None = None


class NLUServiceError(RuntimeError):
    """Base controlled error for M3 intent parsing."""


class NLUInputError(NLUServiceError):
    """Raised when user text is empty or invalid."""


class NLUConfigurationError(NLUServiceError):
    """Raised when Gemini configuration is unavailable."""


class NLURequestError(NLUServiceError):
    """Raised for controlled Gemini SDK/network failures."""


class NLUResponseError(NLUServiceError):
    """Raised when Gemini output violates the strict intent contract."""


def _get_client() -> genai.Client:
    """Create one reusable Gemini client lazily, never at import time."""

    global _client
    api_key = settings.gemini_api_key.strip()
    if not api_key:
        raise NLUConfigurationError("Thiếu GEMINI_API_KEY cho Gemini NLU.")
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            _client = genai.Client(api_key=api_key)
        except Exception as exc:
            raise NLUConfigurationError("Không thể khởi tạo Gemini client.") from exc
        return _client


def _clear_client_cache() -> None:
    global _client
    with _client_lock:
        _client = None


def _generation_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0,
        tools=[GEMINI_TOOL],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=list(AUTH_REQUIREMENT),
            )
        ),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def _extract_single_function_call(response: object) -> object:
    try:
        function_calls = getattr(response, "function_calls", None)
    except Exception as exc:
        raise NLUResponseError("Không đọc được function call từ Gemini.") from exc
    if not isinstance(function_calls, list) or not function_calls:
        raise NLUResponseError("Gemini không trả function call.")
    if len(function_calls) != 1:
        raise NLUResponseError("Gemini phải trả đúng một function call.")
    return function_calls[0]


def _validate_arguments(function_name: str, raw_arguments: Any) -> dict[str, object]:
    if not isinstance(raw_arguments, dict):
        raise NLUResponseError("Gemini trả arguments không phải JSON object.")

    if function_name == "answer_faq":
        if set(raw_arguments) != {"topic"}:
            raise NLUResponseError("answer_faq chỉ được nhận argument 'topic'.")
        topic = raw_arguments["topic"]
        if not isinstance(topic, str) or not topic.strip():
            raise NLUResponseError("answer_faq.topic phải là chuỗi không rỗng.")
        return {"topic": topic.strip()}

    if raw_arguments:
        raise NLUResponseError(f"{function_name} không được nhận arguments.")
    return {}


def _validate_function_call(function_call: object) -> dict[str, object]:
    function_name = getattr(function_call, "name", None)
    if not isinstance(function_name, str) or function_name not in AUTH_REQUIREMENT:
        raise NLUResponseError("Gemini trả function không nằm trong allowlist.")
    raw_arguments = getattr(function_call, "args", None)
    arguments = _validate_arguments(function_name, raw_arguments)
    return {"function_name": function_name, "arguments": arguments}


def parse_intent(user_text: str) -> dict[str, object]:
    """Parse one Vietnamese command into a validated function and arguments."""

    if not isinstance(user_text, str) or not user_text.strip():
        raise NLUInputError("Nội dung yêu cầu không được để trống.")
    model_name = settings.gemini_model.strip()
    if not model_name:
        raise NLUConfigurationError("Thiếu GEMINI_MODEL cho Gemini NLU.")

    client = _get_client()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_text.strip(),
            config=_generation_config(),
        )
    except NLUServiceError:
        raise
    except Exception as exc:
        logger.warning(
            "Gemini request failed for model %s (%s).",
            model_name,
            type(exc).__name__,
        )
        raise NLURequestError("Không thể gọi Gemini để phân tích intent.") from exc

    result = _validate_function_call(_extract_single_function_call(response))
    logger.info("Gemini intent selected function: %s", result["function_name"])
    return result
