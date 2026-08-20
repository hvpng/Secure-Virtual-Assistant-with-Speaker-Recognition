# Workflow & Tech Stack — Secure IT Helpdesk Voice Assistant

> Đề 3 — Secure Virtual Assistant with Speaker Recognition  
> Team: 3 sinh viên  
> Module A chạy Kaggle; Module B chạy local/Codex trong VS Code.  
> Browser demo: Chrome/Edge desktop trên Windows 11.

## 1. Mục tiêu chấm điểm

**Yêu cầu 1 — 5 điểm:** train/fine-tune và đánh giá speaker recognition model, có dataset/split/training procedure/metric/results rõ ràng.

**Yêu cầu 2 — 5 điểm:** trợ lý ảo voice-based có enrollment + user/voice-profile management, general function, sensitive function gated bởi SV, personalized function dùng SID, và report kiến trúc/processing flow.

## 2. Use case

**IT Helpdesk nội bộ doanh nghiệp**

| Nhóm | Auth | Chức năng demo |
|---|---|---|
| General | Không | FAQ giờ làm việc, VPN, ngày nghỉ |
| Sensitive | SV | Reset mật khẩu; xem lương/bảo hiểm |
| Personalized | SID | Xem ngày phép; lịch họp hôm nay |

### Nguyên tắc bảo mật bất biến

LLM chỉ hiểu intent. LLM **không bao giờ** quyết định bỏ qua auth và **không bao giờ** quyết định identity dùng cho resource nhạy cảm.

- SV: backend bind identity = `claimed_employee_id` sau khi verify thành công.
- SID: backend bind identity = kết quả `identify()`.
- Các function self-service không nhận `employee_id` từ LLM.

## 3. Ranh giới Module A / Module B

```text
MODULE A — KAGGLE / RESEARCH
Vietnam-Celeb
   -> speaker-disjoint train/val/test
   -> WavLM-Base+ -> adapter/projection -> CAM++
   -> calibration trên validation
   -> final evaluation trên test
   -> export stable artifacts
        speaker_model.py
        weights/
        speaker_config.json
        enrollment_config.json
        enrollment_config.md
                |
                v
MODULE B — LOCAL / CODEX
M0 Scaffold
   |
   +--> M1 Speaker Service (fake backend trước, real adapter sau)
   +--> M2 ASR/TTS
   +--> M3 Gemini NLU
             |
             v
       M4 Backend API + DB + HARD AUTH GATING
          |                         |
          v                         v
       M5 Enrollment UI          M6 Voice Chat UI
          \                         /
           \                       /
                    M7
        Integration + real model + E2E
```

### Contract bắt buộc giữa A và B

Bất kể Module A dùng Tier 1/2/3, Module B chỉ thấy cùng interface:

```python
# backend/app/models/speaker_model.py
load_model(...)
extract_embedding(model, audio_path: str) -> np.ndarray
```

Cấu hình:
- `speaker_config.json`: architecture metadata + `sv_threshold` + `sid_threshold` + sample rate.
- `enrollment_config.json`: toàn bộ quality thresholds.
- `enrollment_config.md`: giải thích threshold và 7 câu enrollment.

Module B không retrain, không đổi architecture, không tự chọn threshold.

## 4. Dependency chính xác

```text
M0
 |---- M1
 |---- M2
 |---- M3
          \
M1 + M2 + M3 -> M4 -> M5 + M6 -> M7
```

M1 content-match cần ASR, nhưng để giữ thứ tự M1 trước M2:
- M1 nhận `transcribe_fn`.
- Unit test M1 dùng fake `transcribe_fn`.
- Sau M2 mới wire `speech_to_text` thật.

## 5. Audio contract chung

```text
browser .webm/.wav
   -> normalize_audio()
   -> mono PCM WAV, 16 kHz
      |-> M1 quality/SV/SID
      |-> M2 ASR
```

Canonical utility: `backend/app/utils/audio_utils.py`.

Raw audio chỉ là temp file và phải cleanup; lâu dài chỉ giữ voice embedding.

## 6. Tech stack đã chốt

### Backend
Python 3.11+, FastAPI, Uvicorn, Pydantic, SQLAlchemy + SQLite, python-multipart, python-dotenv, pydub + ffmpeg, numpy, webrtcvad, pytest + httpx.

### Speaker
- Primary: WavLM-Base+ + trainable adapter/projection + CAM++
- Fallback 2: Fbank + CAM++
- Fallback 3: ECAPA-TDNN fine-tuned
- App inference only
- `SPEAKER_DEVICE=auto|cuda|cpu`

### ASR/TTS
- PhoWhisper-small qua `transformers`
- fallback Whisper multilingual `base` qua `openai-whisper`
- gTTS `lang="vi"`
- `ASR_DEVICE=auto|cuda|cpu`

### NLU
- Gemini 3.6 Flash: `gemini-3.6-flash`
- SDK: `google-genai`
- `GEMINI_API_KEY`
- `GEMINI_MODEL=gemini-3.6-flash`
- function calling chỉ để parse intent; không automatic-execute business function.

### Frontend
React + Vite + TypeScript + TailwindCSS + React Router + MediaRecorder.

### Máy demo
Windows 11, RAM 16 GB, RTX 3050 6 GB. M7 phải đo latency trên máy này.

## 7. Data model

`employees`:
- `id`
- `name`
- `leave_days_left`
- `meetings_today` JSON
- `salary_mock`
- `insurance_status`
- `password_hash_mock`
- `voice_enrolled`

Embedding: `backend/data/voice_profiles/{employee_id}.npy`.

## 8. User lifecycle

- Seed 3–5 employee records có business data.
- `POST /api/enroll`: enroll người chưa có profile; nếu ID mới, chỉ tạo record sau khi audio pass.
- `POST /api/employees/{id}/reenroll`: chỉ replace profile sau khi toàn bộ sample mới pass.
- `DELETE /api/employees/{id}/voice-profile`: xóa profile, giữ employee.
- `GET /api/employees`: list UI/demo.

## 9. API contracts

### `POST /api/chat`

Multipart:
- `audio`
- `claimed_employee_id` optional

```json
{
  "success": true,
  "text_asr": "tôi muốn reset mật khẩu",
  "function_called": "reset_password",
  "auth_type": "SV",
  "auth_passed": true,
  "employee_id": "NV001",
  "speaker_score": 0.73,
  "response_text": "Mật khẩu đã được reset.",
  "audio_reply_url": "/api/audio/..."
}
```

General: `auth_type/auth_passed/employee_id/speaker_score = null`.

Auth denial: HTTP 200, `success=false`, có response/TTS, **không side effect**.

### `GET /api/enrollment-scripts`

```json
{"scripts":[{"index":0,"text":"..."}]}
```

Index luôn 0-based.

### `POST /api/enroll`

Fields:
- `employee_id`
- `name`
- repeated `audio_files`, đúng 7 file theo thứ tự scripts.

Failure HTTP 400:

```json
{
  "success": false,
  "failed_items": [
    {
      "index": 2,
      "checks": {
        "duration_ok": true,
        "speech_ratio_ok": false,
        "snr_ok": false,
        "clipping_ok": true,
        "content_match_ok": true
      },
      "reasons": ["Âm thanh quá nhiễu."]
    }
  ]
}
```

## 10. Gemini tools

```text
answer_faq(topic)
reset_password()
check_salary_insurance()
check_leave_days()
check_today_meetings()
```

Hardcoded:

```python
AUTH_REQUIREMENT = {
    "answer_faq": None,
    "reset_password": "SV",
    "check_salary_insurance": "SV",
    "check_leave_days": "SID",
    "check_today_meetings": "SID",
}
```

Unknown/malformed tool call phải fail closed.

## 11. Testing

1. Unit/contract deterministic với mock/fake.
2. Live smoke từng component.
3. M7 real E2E với real Module A speaker backend.

## 12. Enrollment reference

Paper chính:
- *Phoneme-Based Optimization of Enrollment Selection for Speaker Identification*
- DOI `10.1007/978-981-95-7075-1_34`

Correction:
- DOI `10.1007/978-981-95-7075-1_51`

Nếu không reproduce đầy đủ thuật toán paper, report phải nói rõ 7 câu là thiết kế của nhóm, không claim exact reproduction.
