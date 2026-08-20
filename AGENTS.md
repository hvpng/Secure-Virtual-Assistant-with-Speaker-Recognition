# AGENTS.md — Codex Rules for Secure Voice Assistant

Đọc file này trước mọi task. Mỗi module là một phiên riêng; không dựa vào trí nhớ từ phiên trước.

## 1. Scope

Đây là **Module B** của đồ án. Module A chạy Kaggle và chịu trách nhiệm train/eval/export speaker model.

**Codex không train speaker model.**

## 2. Immutable rules

1. **Auth gating luôn là Python hardcoded ở backend.**
   - Gemini chỉ parse intent.
   - Unknown/malformed tool call fail closed.

2. **Identity không lấy từ LLM cho self-service.**
   - SV identity = `claimed_employee_id` sau verify.
   - SID identity = kết quả `identify()`.
   - Không dùng `employee_id` do Gemini sinh để reset/xem lương/ngày phép/lịch họp.

3. **Speaker model đã train sẵn ở Module A.**
   - Không retrain.
   - Không đổi WavLM/CAM++.
   - Không tự tải model khác để thay production artifact.
   - Chỉ import `backend/app/models/speaker_model.py`.
   - Fake backend chỉ dev/unit test; M7 final bắt buộc real backend.

4. **Enrollment luôn qua `check_audio_quality()` trước khi lưu.**
   - Bất kỳ sample fail -> không tạo/replace profile.
   - Re-enroll atomic.

5. **Không tự invent threshold.**
   - Speaker threshold từ `speaker_config.json`.
   - Quality threshold từ `enrollment_config.json`, đồng bộ với `enrollment_config.md`.
   - Placeholder/test config không được dùng M7 final.

6. **Canonical audio:** mono PCM WAV, 16 kHz.
   - `.webm` -> shared `normalize_audio()`.

7. **Không lưu raw voice lâu dài.**
   - temp cleanup trong `finally`.
   - chỉ giữ embedding `.npy`.

8. **Không đổi tech stack/model đã chốt.**

## 3. Tech stack

Backend: Python 3.11+, FastAPI, Uvicorn, SQLAlchemy, SQLite, Pydantic, python-multipart, python-dotenv, pydub/ffmpeg, numpy, webrtcvad, pytest/httpx.

AI:
- Speaker: Module A artifact; primary WavLM-Base+ + CAM++
- ASR: PhoWhisper-small qua `transformers`
- fallback ASR: Whisper multilingual `base`
- TTS: gTTS
- NLU: Gemini 3.6 Flash qua `google-genai`
- `GEMINI_API_KEY`
- `GEMINI_MODEL=gemini-3.6-flash`

Frontend: React + Vite + TypeScript + TailwindCSS + React Router + MediaRecorder; target Chrome/Edge.

## 4. Architecture

```text
browser audio -> normalize_audio()
                   |-> M2 ASR -> M3 Gemini -> function_name + args
                   |-> M1 speaker
                                     |
                              M4 hardcoded auth
                               |           |
                              SV          SID
                               \           /
                                identity
                                   |
                              business logic
                                   |
                                 SQLite
                                   |
                                  TTS
```

Dependency:

```text
M0
 |-- M1
 |-- M2
 |-- M3
M1 + M2 + M3 -> M4 -> M5/M6 -> M7
```

## 5. Repository structure

```text
project/
├── AGENTS.md
├── README.md
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes.py
│   │   ├── core/config.py
│   │   ├── core/enrollment_scripts.py
│   │   ├── db/database.py
│   │   ├── db/seed_data.py
│   │   ├── models/
│   │   │   ├── speaker_model.py
│   │   │   ├── fake_speaker_model.py
│   │   │   ├── speaker_config.json
│   │   │   ├── enrollment_config.json
│   │   │   └── weights/
│   │   ├── services/
│   │   │   ├── speaker_service.py
│   │   │   ├── asr_tts_service.py
│   │   │   └── nlu_service.py
│   │   └── utils/audio_utils.py
│   ├── data/
│   │   ├── voice_profiles/
│   │   ├── generated_audio/
│   │   └── employees.db
│   ├── tests/fixtures/
│   └── requirements.txt
├── frontend/src/
│   ├── api/client.ts
│   ├── components/
│   ├── pages/Enrollment.tsx
│   ├── pages/Chat.tsx
│   ├── App.tsx
│   └── main.tsx
├── docs/
│   ├── enrollment_config.md
│   └── enrollment_scripts.md
└── specs/
```

Root `README.md` là canonical. Không tạo `docs/README.md`. Không tạo `CLAUDE.md`.

## 6. Canonical API contracts

### ChatResponse

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
  "audio_reply_url": "/api/audio/..."
}
```

General: `auth_type/auth_passed/employee_id/speaker_score = null`.

### Enrollment failure

```json
{
  "success": false,
  "failed_items": [
    {
      "index": 0,
      "checks": {
        "duration_ok": false,
        "speech_ratio_ok": false,
        "snr_ok": false,
        "clipping_ok": true,
        "content_match_ok": true
      },
      "reasons": ["..."]
    }
  ]
}
```

Index 0-based.

## 7. Gemini tools

```text
answer_faq(topic)
reset_password()
check_salary_insurance()
check_leave_days()
check_today_meetings()
```

```python
AUTH_REQUIREMENT = {
    "answer_faq": None,
    "reset_password": "SV",
    "check_salary_insurance": "SV",
    "check_leave_days": "SID",
    "check_today_meetings": "SID",
}
```

Không thêm `employee_id` vào self-service tool args.

## 8. Test commands

Backend:

```bash
cd backend
python -m compileall app
pytest -q
```

Manual server:

```bash
cd backend
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

M7 final:
- full tests;
- real speaker backend;
- fake backend bị disable trong final/demo mode.

## 9. Git

Mỗi module code -> test -> manual review -> commit riêng -> mới mở module kế.

```text
chore(M0): scaffold backend and frontend
feat(M1): speaker service and enrollment quality
feat(M2): ASR and TTS services
feat(M3): Gemini NLU intent parser
feat(M4): backend API and hard auth gating
feat(M5): enrollment and voice profile management UI
feat(M6): voice chat UI
test(M7): real-model integration and E2E
```

## 10. Stop conditions

Nếu artifact thật chưa có:
- dùng fake/mock đúng spec;
- không tự đổi model/training;
- TODO rõ ràng;
- vẫn hoàn thành phần deterministic có thể test.

Nếu spec mâu thuẫn AGENTS.md, ưu tiên AGENTS.md và nêu mâu thuẫn trong summary.
