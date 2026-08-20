# MODULE 3 — Gemini NLU Intent Parser

## Mục tiêu
Dùng Gemini **chỉ để hiểu intent + semantic args**. Không thực thi function thật và không quyết định authentication.

## Model
- SDK: `google-genai`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- default `gemini-3.6-flash`

## Tools

```text
answer_faq(topic: str)
reset_password()
check_salary_insurance()
check_leave_days()
check_today_meetings()
```

**Không có `employee_id` trong self-service tool args.**

## Hardcoded auth policy

```python
AUTH_REQUIREMENT = {
    "answer_faq": None,
    "reset_password": "SV",
    "check_salary_insurance": "SV",
    "check_leave_days": "SID",
    "check_today_meetings": "SID",
}
```

Gemini không được sửa policy này.

## API

```python
parse_intent(user_text: str) -> dict
```

Success:

```json
{"function_name":"reset_password","arguments":{}}
```

FAQ:

```json
{"function_name":"answer_faq","arguments":{"topic":"vpn"}}
```

## Gemini calling mode

Dùng function declarations/schema để nhận function call.

**Không pass actual Python business functions theo cách cho phép SDK automatic-execute.**

Sau response:
- function phải trong allowlist;
- args validate;
- malformed/no call -> controlled error;
- timeout/API error -> controlled error;
- không tự chọn fallback action.

Fail closed.

## Tests

Deterministic:
- mock Gemini valid;
- unknown tool;
- malformed args;
- no tool call;
- API exception.

Live optional:
- "hướng dẫn tôi xin VPN"
- "reset mật khẩu giúp tôi"
- "tôi muốn xem thông tin lương"
- "tôi còn bao nhiêu ngày phép"
- "hôm nay tôi có cuộc họp nào"

## Prompt dán cho Codex

Bạn đang làm **M3 בלבד**. Đọc AGENTS.md. Dùng Gemini API qua package `google-genai`, model mặc định `gemini-3.6-flash`. Implement đúng 5 function declarations. LLM chỉ parse intent, không chạy tool thật, không nhận employee_id cho self-service. Giữ `AUTH_REQUIREMENT` hardcoded. Validate allowlist và fail closed. Unit tests mock Gemini; live API chỉ smoke optional. Không code business logic/M4.

## Acceptance

```bash
cd backend
python -m compileall app
pytest -q tests/test_nlu_service.py
```

## Review
- [ ] không `anthropic`;
- [ ] không Claude ID/key;
- [ ] Gemini không auto-execute business functions;
- [ ] no employee_id self-service arg;
- [ ] unknown function fail closed;
- [ ] commit `feat(M3): Gemini NLU intent parser`.
