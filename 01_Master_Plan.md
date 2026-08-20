# Master Plan — Secure IT Helpdesk Voice Assistant

## 1. Mapping yêu cầu đề bài

| Yêu cầu | Điểm | Module | Deliverable |
|---|---:|---|---|
| Train/eval speaker model | 5 | Module A | Kaggle notebook, manifests, eval, weights, `speaker_model.py`, configs |
| Voice assistant + enrollment + SV + SID | 5 | M0–M7 | FastAPI + React app, tests, report, demo |

## 2. Dependency

```text
Module A (Kaggle) chạy song song
               |
               | stable artifact
               v
M0 -> M1
   -> M2
   -> M3
M1 + M2 + M3 -> M4 -> M5 + M6 -> M7
```

M1 code trước M2 bằng fake `transcribe_fn`; sau M2 mới wire ASR thật.

## 3. Phân công

| Người | Track | Trách nhiệm |
|---|---|---|
| A | Research | Module A: data/model/train/eval/calibration/export |
| B | Backend | M0–M4 |
| C | Frontend | M5–M6 |
| Cả nhóm | Integration | M7/report/demo |

## 4. Timeline khuyến nghị

### Ngày 1
A: split + smoke WavLM -> adapter -> CAM++ trên subset; chuẩn bị fallback nếu forward/backward không ổn.  
B: M0; M1 với deterministic fake backend + quality contract.

### Ngày 2
A: train Tier đang hoạt động, theo dõi val EER sớm.  
B: M2 PhoWhisper/gTTS; M3 Gemini NLU.

### Ngày 3
A: calibration trên **validation**: SV/SID/SNR/duration; freeze config; final test; export.  
B: M4 API/DB/auth; review security thủ công.

### Ngày 4
C: M5 + M6.  
B/C: swap real Module A artifacts, smoke local RTX 3050.

### Ngày 5 / buffer
M7 E2E, latency, report, video, package.

Nếu buộc 3 ngày: cắt animation, TTS nâng cấp, optional WavLM Stage 2, optional extra experiments. Không cắt auth, enrollment quality, real model final, SV/SID demo.

## 5. Gate từng module

1. Chỉ giao đúng một spec cho Codex.
2. Code.
3. Test/build.
4. Review thủ công.
5. Fix.
6. Commit riêng.
7. Mới mở module kế tiếp.

## 6. Requirement 1 acceptance

- [ ] speaker-disjoint train/val/test.
- [ ] verification protocol reproducible.
- [ ] architecture thực tế ghi rõ.
- [ ] EER; minDCF nếu kịp.
- [ ] SID top-1 + unknown rejection.
- [ ] calibration chỉ validation.
- [ ] `sv_threshold`, `sid_threshold` freeze trước test.
- [ ] SNR/duration experiments trên validation.
- [ ] final test sau freeze.
- [ ] stable `speaker_model.py`.
- [ ] configs export đầy đủ.

## 7. Requirement 2 acceptance

- [ ] General FAQ không auth.
- [ ] SV đúng -> sensitive action thành công.
- [ ] SV sai -> action không chạy.
- [ ] SID đúng registered speaker.
- [ ] Unknown SID -> không lộ data.
- [ ] Enrollment tốt -> profile.
- [ ] Enrollment kém/sai nội dung -> reject đúng item.
- [ ] Re-enroll atomic.
- [ ] Remove voice profile giữ employee.
- [ ] LLM không điều khiển identity.
- [ ] Unknown Gemini tool fail closed.
- [ ] TTS audio URL phát Chrome/Edge.
- [ ] Final E2E dùng real speaker backend.
- [ ] Latency đo trên máy demo.

## 8. Risk log

| Risk | Mức | Mitigation |
|---|---|---|
| WavLM+CAM++ khó | Cao | timebox; Tier 2 Fbank+CAM++; Tier 3 ECAPA |
| A/B artifact mismatch | Cao | stable `speaker_model.py` ABI |
| Test leakage | Cao | calibration val; test final only |
| Browser audio | Cao | shared `normalize_audio()` |
| Auth bypass | Cao | hardcoded policy + identity binding + tests |
| Gemini/network | Trung bình | mock unit tests + live smoke |
| Local OOM/latency | Trung bình | device env + benchmark sớm |
| Raw biometric storage | Trung bình | temp cleanup; chỉ giữ embedding |
