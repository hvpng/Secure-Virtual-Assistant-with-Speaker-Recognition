# Secure Virtual Assistant with Speaker Recognition

This repository contains Module B, the local application for the Secure IT Helpdesk Voice Assistant. Module A trains an ECAPA-TDNN speaker encoder on VoxVietnam separately on Kaggle and exports one shared 192-D embedding model for SV and SID. M0 provides project structure, configuration, and a health-check connection between the frontend and backend only.

## Prerequisites

- Python 3.11 or newer
- Node.js and npm
- ffmpeg available on `PATH`
- Chrome or Edge for the target browser experience

On Windows 11, verify the prerequisites from PowerShell:

```powershell
python --version
node --version
npm --version
ffmpeg -version
```

## Backend setup

Create and activate a virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

Copy `.env.example` to `.env` and supply secrets locally when later modules require them. Never commit `.env`. M0 does not require a Gemini API key to run.

Run the backend:

```powershell
cd backend
uvicorn app.main:app --reload
```

The backend is available at <http://localhost:8000>. Test its health endpoint with:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

It returns `{"status":"ok"}`. Development CORS explicitly allows `http://localhost:5173`.

### M1 audio and speaker configuration

Speaker input is normalized through the shared audio utility to a temporary,
mono, 16 kHz, 16-bit PCM WAV. Browser `.webm` decoding requires `ffmpeg` on
`PATH`. Callers must delete the returned temporary WAV in a `finally` block;
the speaker service already follows this contract.

M1 uses `SPEAKER_BACKEND=fake` for deterministic local development and unit
tests. Set `SPEAKER_BACKEND=real` only after the ECAPA Module A export is integrated into
`backend/app/models/speaker_model.py` with the real artifact. There is no silent
fallback from real to fake. The checked-in speaker and enrollment JSON configs
contain development-only thresholds and must be replaced by Module A's
validation-calibrated exports before M7/final mode.

### M2 Vietnamese ASR and TTS

M2 uses `vinai/PhoWhisper-small` through HuggingFace `transformers` as primary
ASR. `ASR_DEVICE=auto` chooses CUDA when PyTorch reports it available and CPU
otherwise. The multilingual OpenAI Whisper `base` model is lazy-loaded only
after a logged PhoWhisper failure and only when `ASR_FALLBACK_ENABLED=true`.
The first real ASR run can download model files; deterministic unit tests mock
all model loaders and require no network.

Run a real WAV or browser WebM smoke test from `backend`:

```powershell
python -m scripts.smoke_asr C:\path\to\vietnamese-audio.wav
python -m scripts.smoke_asr C:\path\to\browser-recording.webm
```

Optionally test live gTTS (network required):

```powershell
python -m scripts.smoke_asr --tts-text "Xin chào, đây là thử nghiệm."
```

Generated MP3 files are written to `backend/data/generated_audio/`. M2 creates
unique names and removes partial failures; M4/M7 must define cleanup after a
response has been served.

### M3 Gemini intent routing

M3 uses native Gemini function calling through `google-genai`. Set
`GEMINI_API_KEY` locally in `.env`; `GEMINI_MODEL` defaults to
`gemini-3.6-flash`. Gemini only selects one of five allowlisted intents and
semantic arguments. Authentication requirements remain hardcoded in Python;
M3 does not verify speakers, choose identity, or execute business actions.

Run a live smoke test from `backend` (network and a valid API key required):

```powershell
python -m scripts.smoke_nlu "Tôi muốn reset mật khẩu."
```

### M4 backend API and hard auth gating

The backend initializes the SQLite schema and idempotently seeds four demo
employees when FastAPI starts. By default the database is stored at
`backend/data/employees.db`; set `DATABASE_URL` only when an alternate database
is needed for development or tests. API documentation is available at
<http://localhost:8000/docs>.

M4 exposes these routes:

- `POST /api/chat` — ASR, intent routing, Python-controlled SV/SID gating,
  business execution, and a generated TTS reply.
- `POST /api/enroll` — initial enrollment using the seven server-owned scripts.
- `POST /api/employees/{employee_id}/reenroll` — atomic profile replacement.
- `DELETE /api/employees/{employee_id}/voice-profile` — remove only the voice
  profile while retaining the employee record.
- `GET /api/employees` — demo-safe employee selector data.
- `GET /api/enrollment-scripts` — the canonical enrollment prompts.
- `GET /api/audio/{filename}` — generated MP3 replies restricted to
  `backend/data/generated_audio/`.

Local development and deterministic tests may use `SPEAKER_BACKEND=fake`.
Final/demo mode must set `SPEAKER_BACKEND=real` and provide the calibrated
Module A artifact and configs; the application never silently falls back to
the fake backend.

Run backend checks from `backend`:

```powershell
python -m compileall app
pytest -q
```

## Frontend setup

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The scaffold calls `http://localhost:8000/health` and displays the backend status. If the backend is unavailable, it displays a simple connection error. Placeholder routes are available at `/enroll` and `/chat`; their full interfaces belong to M5 and M6.

Run frontend checks from `frontend`:

```powershell
npm run lint
npm run build
```

## Module boundaries

M0 contains no speaker inference, ASR/TTS, Gemini function calling, business database schema, enrollment flow, or voice chat implementation. Speaker artifacts are produced by Module A and integrated in later modules.

The Kaggle ECAPA/VoxVietnam training, validation calibration, final VoxVietnam-O
evaluation, and export workflow is documented in `module_a/README.md`.
