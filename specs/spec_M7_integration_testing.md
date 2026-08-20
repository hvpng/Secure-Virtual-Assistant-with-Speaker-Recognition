# MODULE 7 — Real Integration, E2E, Report Readiness

## Mục tiêu
Thay fake speaker bằng artifact thật từ Module A, chạy real E2E và khóa bản demo/nộp.

## 1. Preflight

Module A cung cấp:
- `speaker_model.py`
- `weights/`
- `speaker_config.json`
- `enrollment_config.json`
- `enrollment_config.md`
- enrollment scripts/results.

Set:

```text
SPEAKER_BACKEND=real
GEMINI_MODEL=gemini-3.6-flash
APP_MODE=final
```

Final/demo mode phải fail explicit nếu `SPEAKER_BACKEND=fake`.

## 2. Config validation

- không threshold placeholder;
- sample rate 16000;
- model load;
- embedding finite;
- `sv_threshold`/`sid_threshold` load;
- enrollment config load.

## 3. Deterministic suite

Backend:

```bash
cd backend
python -m compileall app
pytest -q
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

Unit/integration mock tests không phụ thuộc network.

## 4. Live component smoke
- real speaker extract;
- PhoWhisper;
- Gemini;
- gTTS;
- DB.

Ghi rõ dependency internet.

## 5. Real E2E

1. General FAQ, no claim, no auth gate.
2. SV success: Alice claim Alice, sensitive action thành công.
3. SV impostor: Bob claim Alice, action không chạy.
4. Identity-binding adversarial: câu lệnh cố nhắc target người khác; không được dùng arbitrary identity từ Gemini.
5. SID known Alice -> đúng personalized data.
6. SID second user Bob -> đúng Bob data.
7. SID unknown -> không lộ data.
8. Enrollment 7 good -> profile -> verify/identify smoke.
9. Enrollment quality fail -> failed index exact, no partial.
10. Wrong content -> content_match false.
11. Re-enroll bad -> profile cũ giữ nguyên; good -> replace.
12. Remove profile -> employee còn, voice_enrolled false, profile không dùng được.

## 6. Latency benchmark

Máy demo:
- Windows 11
- RAM 16 GB
- RTX 3050 6 GB

Đo ít nhất 5 chat requests:
- normalize;
- ASR;
- Gemini;
- speaker step nếu có;
- TTS;
- total.

Report median/rough range.

Nếu CUDA OOM:
- đổi device bằng env;
- không đổi architecture/model;
- document device cuối.

## 7. Browser test

Chrome/Edge:
- mic;
- webm upload;
- enrollment;
- chat;
- audio playback;
- no console errors.

## 8. README/report

Root README:
- architecture;
- prerequisites;
- install;
- ffmpeg;
- env vars;
- Module A artifact placement;
- demo;
- limitations.

Report:
- Requirement 1 data/split/model/train/eval/results;
- calibration validation;
- Requirement 2 architecture;
- enrollment procedure/storage;
- general/SV/SID;
- E2E;
- limitations:
  - replay/deepfake chưa giải quyết;
  - Gemini/gTTS network dependency;
  - phạm vi phoneme enrollment.

Paper:
- original DOI `_34`;
- correction DOI `_51`;
- không claim exact reproduction nếu không làm.

## Prompt dán cho Codex

Bạn đang làm **M7 בלבד**. Đọc AGENTS.md. Cài real Module A artifacts và set SPEAKER_BACKEND=real; không training model. Chạy deterministic tests, live smoke và real E2E. Thêm guard final/demo không chạy fake speaker. Đo latency trên máy local. Hoàn thiện root README/test docs. Khi gặp lỗi, sửa integration tối thiểu, không đổi tech stack/model.

## Final review
- [ ] env không expose secret;
- [ ] final dùng real speaker;
- [ ] impostor test;
- [ ] unknown SID;
- [ ] re-enroll failure atomic;
- [ ] inspect M4 auth lần cuối;
- [ ] report numbers khớp artifacts;
- [ ] commit `test(M7): real-model integration and E2E`.
