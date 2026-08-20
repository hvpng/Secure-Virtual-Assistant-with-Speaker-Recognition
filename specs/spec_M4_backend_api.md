# MODULE 4 — Backend API, DB & Hard Auth Gating

## Mục tiêu
Ghép M1 + M2 + M3. Đây là **security boundary**.

**Phải review source code thủ công; không chỉ đọc Codex summary.**

## Files
- `backend/app/api/routes.py`
- `backend/app/db/database.py`
- `backend/app/db/seed_data.py`
- `backend/app/core/enrollment_scripts.py`
- schemas/config liên quan
- tests M4

## Database

`employees`:
- `id: str` primary key
- `name: str`
- `leave_days_left: int`
- `meetings_today: JSON`
- `salary_mock: str`
- `insurance_status: str`
- `password_hash_mock: str`
- `voice_enrolled: bool`

Seed 3–5 records, có thể `voice_enrolled=false`.

## Business functions

```python
answer_faq(topic)
reset_password(employee_id)
check_salary_insurance(employee_id)
check_leave_days(employee_id)
check_today_meetings(employee_id)
```

Gemini không gọi trực tiếp.

## `/api/chat` exact flow

1. nhận multipart `audio` + optional `claimed_employee_id`;
2. save temp;
3. normalize -> canonical WAV;
4. ASR;
5. Gemini parse intent;
6. validate function trong `AUTH_REQUIREMENT`;
7. lookup auth type.

### General
Không speaker auth. `employee_id=null`.

### SV
- thiếu claimed ID -> deny;
- employee phải tồn tại và có voice profile;
- `verify(normalized, claimed_employee_id)`;
- fail -> **return trước business function**;
- pass -> `authenticated_employee_id = claimed_employee_id`;
- ignore identity từ LLM.

### SID
- `identify(normalized)`;
- None -> deny;
- bind `authenticated_employee_id = identified id`;
- employee phải tồn tại + voice_enrolled;
- ignore identity từ LLM.

8. business function chỉ nhận backend-authenticated identity;
9. TTS;
10. ChatResponse;
11. cleanup temp trong `finally`.

## ChatResponse

```json
{
  "success": true,
  "text_asr": "...",
  "function_called": "reset_password",
  "auth_type": "SV",
  "auth_passed": true,
  "employee_id": "NV001",
  "speaker_score": 0.73,
  "response_text": "...",
  "audio_reply_url": "/api/audio/<id>"
}
```

Auth denied:
- HTTP 200;
- `success=false`;
- `auth_passed=false`;
- zero business side effect;
- response denial + TTS nếu có.

General:
- auth fields null.

## Audio serving

Safe endpoint/static mount dưới `/api/audio/...`.

Chỉ serve từ `data/generated_audio`; không expose filesystem path tùy ý.

## Enrollment scripts

`GET /api/enrollment-scripts`:

```json
{"scripts":[{"index":0,"text":"..."}]}
```

Single source `app/core/enrollment_scripts.py`.

## `POST /api/enroll`

Multipart:
- `employee_id`
- `name`
- repeated `audio_files`, exactly 7.

Rules:
- existing employee đã enrolled -> 409, dùng re-enroll;
- existing ID name conflict -> 409;
- `enroll_user(..., transcribe_fn=speech_to_text)`;
- fail -> HTTP 400 structured `failed_items`;
- pass:
  - existing employee: set voice_enrolled true;
  - new employee: create only after successful voice enrollment.

DB/profile state phải rollback/không lệch nếu exception.

## Management

### `POST /api/employees/{id}/reenroll`
- 7 audio;
- all quality pass trước;
- chỉ replace profile cũ sau khi new profile hoàn thành;
- failed re-enroll giữ profile cũ.

### `DELETE /api/employees/{id}/voice-profile`
- xóa embedding;
- set `voice_enrolled=false`;
- giữ employee.

### `GET /api/employees`

```json
{
  "employees":[
    {"id":"NV001","name":"...","voice_enrolled":true}
  ]
}
```

Không trả salary/password hash.

## Security tests bắt buộc

1. General -> không speaker call.
2. SV pass -> sensitive action đúng 1 lần.
3. SV fail -> action 0 lần.
4. Missing claim -> action 0.
5. Identity-binding adversarial: Gemini payload cố có employee_id khác -> reject field hoặc backend vẫn chỉ dùng authenticated identity.
6. SID None -> không lộ personalized data.
7. Unknown function -> fail closed.
8. Re-enroll fail -> profile cũ nguyên.
9. Delete profile -> employee còn.
10. Path traversal audio endpoint không đọc file ngoài generated dir.

## Prompt dán cho Codex

Bạn đang làm **M4 בלבד**. Đọc AGENTS.md và M1/M2/M3 đã pass. Implement DB + exact API. Đây là security boundary: identity cho sensitive/personalized action bắt buộc do backend bind từ SV/SID, không từ Gemini. Unknown tool fail closed. Enrollment/re-enroll atomic; temp cleanup trong finally. Safe generated-audio serving. Viết deterministic integration tests bằng mock services; không phụ thuộc live API/model. Không code M5/M6.

## Acceptance

```bash
cd backend
python -m compileall app
pytest -q
uvicorn app.main:app --reload
```

Manual Swagger test toàn bộ endpoints.

## Manual review bắt buộc
- [ ] trace None/SV/SID branches;
- [ ] auth fail return trước side effect;
- [ ] business employee_id từ backend;
- [ ] chạy identity-confusion test thật;
- [ ] temp cleanup;
- [ ] exact response fields;
- [ ] commit `feat(M4): backend API and hard auth gating`.
