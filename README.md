# Secure Virtual Assistant with Speaker Recognition

This repository contains Module B, the local application for the Secure IT Helpdesk Voice Assistant. Module A trains and exports the speaker model separately on Kaggle. M0 provides project structure, configuration, and a health-check connection between the frontend and backend only.

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
tests. Set `SPEAKER_BACKEND=real` only after Module A replaces
`backend/app/models/speaker_model.py` with the real artifact. There is no silent
fallback from real to fake. The checked-in speaker and enrollment JSON configs
contain development-only thresholds and must be replaced by Module A's
validation-calibrated exports before M7/final mode.

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
