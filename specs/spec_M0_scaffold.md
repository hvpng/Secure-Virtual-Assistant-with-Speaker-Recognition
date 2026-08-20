# MODULE 0 — Project Scaffold & Environment

## Mục tiêu
Dựng repo chuẩn để các module sau không tự tạo cấu trúc khác nhau.

## Scope
Chỉ scaffold/config/health check. Không code business logic.

## Cấu trúc bắt buộc
Dùng cây thư mục trong root `AGENTS.md`.

Phải có:
- `backend/app/utils/audio_utils.py` placeholder;
- `backend/app/models/fake_speaker_model.py` placeholder;
- `backend/app/core/config.py`;
- `backend/app/core/enrollment_scripts.py`;
- root `README.md`;
- không tạo `CLAUDE.md`.

## Backend dependencies

`backend/requirements.txt` tối thiểu:
- fastapi
- uvicorn
- python-multipart
- sqlalchemy
- pydantic
- python-dotenv
- numpy
- pydub
- webrtcvad
- torch
- transformers
- gTTS
- openai-whisper
- google-genai
- pytest
- httpx
- jiwer

README prerequisites:
- Python 3.11+
- Node.js/npm
- ffmpeg
- Chrome/Edge target

## Frontend
React + Vite + TypeScript + TailwindCSS + React Router.

M0 chỉ cần trang nhỏ hiển thị health status.

## Backend
- `GET /health`
- response exact `{"status":"ok"}`.
- CORS `http://localhost:5173` ở dev hoặc Vite proxy tương đương; chọn một và document.

## Environment

`.env.example`:

```text
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.6-flash
ASR_DEVICE=auto
SPEAKER_DEVICE=auto
SPEAKER_BACKEND=fake
APP_MODE=dev
```

Không commit `.env`.

## Prompt dán cho Codex

Bạn đang làm **M0 בלבד**. Đọc `AGENTS.md` trước. Hãy scaffold project đúng cây canonical; backend FastAPI + frontend React/Vite/TypeScript/TailwindCSS. Tạo `/health`, cấu hình dev CORS hoặc proxy để frontend gọi health được, tạo `requirements.txt`, `.env.example`, root `README.md`, placeholder files cho các module sau. Không triển khai speaker/ASR/Gemini/database business logic. Không tạo CLAUDE.md. Không code module tiếp theo.

## Acceptance

Backend:

```bash
cd backend
python -m compileall app
pytest -q
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
npm run dev
```

Manual:
- `/health` -> 200 + exact JSON;
- Chrome/Edge hiển thị backend status;
- no CORS error.

## Review
- [ ] cây đúng AGENTS;
- [ ] không business logic;
- [ ] `.env` gitignore;
- [ ] root README duy nhất;
- [ ] `google-genai`, không Anthropic;
- [ ] commit `chore(M0): scaffold backend and frontend`.
