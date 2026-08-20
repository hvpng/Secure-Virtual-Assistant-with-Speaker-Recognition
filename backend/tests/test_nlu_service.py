from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.services import nlu_service


class FakeModels:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.models = FakeModels(response=response, error=error)


def _response(*calls: tuple[str, object]) -> object:
    return SimpleNamespace(
        function_calls=[
            SimpleNamespace(name=function_name, args=arguments)
            for function_name, arguments in calls
        ]
    )


@pytest.fixture(autouse=True)
def isolated_nlu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nlu_service,
        "settings",
        SimpleNamespace(
            gemini_api_key="offline-test-key",
            gemini_model="gemini-3.6-flash",
        ),
    )
    nlu_service._clear_client_cache()
    yield
    nlu_service._clear_client_cache()


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    response: object = None,
    error: Exception | None = None,
) -> FakeClient:
    client = FakeClient(response=response, error=error)
    monkeypatch.setattr(nlu_service, "_get_client", lambda: client)
    return client


@pytest.mark.parametrize("user_text", ["", "   ", "\n\t"])
def test_empty_input_is_rejected(user_text: str) -> None:
    with pytest.raises(nlu_service.NLUInputError, match="không được để trống"):
        nlu_service.parse_intent(user_text)


@pytest.mark.parametrize(
    ("user_text", "function_name", "arguments"),
    [
        ("hướng dẫn tôi xin VPN", "answer_faq", {"topic": "vpn"}),
        ("tôi muốn reset mật khẩu", "reset_password", {}),
        (
            "cho tôi xem lương và bảo hiểm",
            "check_salary_insurance",
            {},
        ),
        ("tôi còn bao nhiêu ngày phép", "check_leave_days", {}),
        ("hôm nay tôi có lịch họp gì", "check_today_meetings", {}),
    ],
)
def test_valid_intents_return_canonical_contract(
    user_text: str,
    function_name: str,
    arguments: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_client(
        monkeypatch, response=_response((function_name, arguments))
    )

    result = nlu_service.parse_intent(f"  {user_text}  ")

    assert result == {"function_name": function_name, "arguments": arguments}
    assert client.models.calls[0]["contents"] == user_text
    assert client.models.calls[0]["model"] == "gemini-3.6-flash"


@pytest.mark.parametrize(
    "function_name",
    [
        "reset_password",
        "check_salary_insurance",
        "check_leave_days",
        "check_today_meetings",
    ],
)
def test_self_service_functions_reject_identity_argument(
    function_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_client(
        monkeypatch,
        response=_response((function_name, {"employee_id": "NV001"})),
    )

    with pytest.raises(nlu_service.NLUResponseError, match="không được nhận arguments"):
        nlu_service.parse_intent("yêu cầu self service")


def test_unknown_function_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, response=_response(("delete_everything", {})))

    with pytest.raises(nlu_service.NLUResponseError, match="allowlist"):
        nlu_service.parse_intent("unknown")


@pytest.mark.parametrize(
    ("function_name", "arguments"),
    [
        ("answer_faq", {}),
        ("answer_faq", {"topic": ""}),
        ("answer_faq", {"topic": "vpn", "extra": "drift"}),
        ("reset_password", ["not", "an", "object"]),
    ],
)
def test_malformed_arguments_fail_closed(
    function_name: str, arguments: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_client(
        monkeypatch, response=_response((function_name, arguments))
    )

    with pytest.raises(nlu_service.NLUResponseError):
        nlu_service.parse_intent("malformed")


@pytest.mark.parametrize(
    "function_calls",
    [None, [], "not a call list"],
)
def test_no_function_call_fails_closed(
    function_calls: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_client(
        monkeypatch, response=SimpleNamespace(function_calls=function_calls)
    )

    with pytest.raises(nlu_service.NLUResponseError, match="không trả function call"):
        nlu_service.parse_intent("không có tool")


def test_multiple_function_calls_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(
        monkeypatch,
        response=_response(
            ("reset_password", {}),
            ("check_leave_days", {}),
        ),
    )

    with pytest.raises(nlu_service.NLUResponseError, match="đúng một"):
        nlu_service.parse_intent("hai hành động")


def test_gemini_exception_becomes_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch, error=TimeoutError("secret transport details"))

    with pytest.raises(
        nlu_service.NLURequestError,
        match="Không thể gọi Gemini",
    ) as caught:
        nlu_service.parse_intent("reset mật khẩu")

    assert "secret transport details" not in str(caught.value)


def test_missing_api_key_is_controlled_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nlu_service.settings.gemini_api_key = "   "

    with pytest.raises(nlu_service.NLUConfigurationError, match="GEMINI_API_KEY"):
        nlu_service.parse_intent("reset mật khẩu")


def test_missing_model_is_controlled_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nlu_service.settings.gemini_model = ""

    with pytest.raises(nlu_service.NLUConfigurationError, match="GEMINI_MODEL"):
        nlu_service.parse_intent("reset mật khẩu")


def test_auth_requirement_is_exact_hardcoded_policy() -> None:
    assert nlu_service.AUTH_REQUIREMENT == {
        "answer_faq": None,
        "reset_password": "SV",
        "check_salary_insurance": "SV",
        "check_leave_days": "SID",
        "check_today_meetings": "SID",
    }


def test_tool_schemas_are_exact_and_contain_no_identity_fields() -> None:
    declarations = {
        declaration.name: declaration for declaration in nlu_service.FUNCTION_DECLARATIONS
    }
    assert set(declarations) == set(nlu_service.AUTH_REQUIREMENT)
    faq_schema = declarations["answer_faq"].parameters_json_schema
    assert faq_schema["required"] == ["topic"]
    assert set(faq_schema["properties"]) == {"topic"}
    for function_name in set(declarations) - {"answer_faq"}:
        schema = declarations[function_name].parameters_json_schema
        assert schema == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }


def test_generation_config_requires_one_allowed_call_and_disables_execution() -> None:
    config = nlu_service._generation_config()

    function_config = config.tool_config.function_calling_config
    assert function_config.mode.value == "ANY"
    assert function_config.allowed_function_names == list(
        nlu_service.AUTH_REQUIREMENT
    )
    assert config.automatic_function_calling.disable is True


def test_gemini_client_is_created_once_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    class FakeGenAIClient:
        def __init__(self, api_key: str) -> None:
            created.append(api_key)

    monkeypatch.setattr(nlu_service.genai, "Client", FakeGenAIClient)

    first = nlu_service._get_client()
    second = nlu_service._get_client()

    assert first is second
    assert created == ["offline-test-key"]
