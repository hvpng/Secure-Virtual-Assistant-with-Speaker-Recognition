# MODULE 1 — Speaker Service + Enrollment Quality

## Mục tiêu
Tạo wrapper ổn định cho SV/SID và enrollment quality. M1 phải chạy/test được trước khi Module A real artifact sẵn sàng.

## Rules
- Không train model.
- Không tự tải ECAPA/WavLM khác cho production.
- Dev/unit test dùng deterministic fake backend.
- Real backend chỉ import `app.models.speaker_model`.
- Không hardcode threshold `0.5`.

## Files
- `backend/app/services/speaker_service.py`
- `backend/app/models/fake_speaker_model.py`
- `backend/app/utils/audio_utils.py`
- tests M1

## Audio utility

```python
normalize_audio(input_path: str) -> str
```

Output:
- 16 kHz;
- mono;
- PCM WAV;
- temp path.

Caller cleanup.

## Backend selection
`SPEAKER_BACKEND=fake|real`.

Fake backend:
- deterministic;
- chỉ dev/unit test;
- không cần speaker accuracy thật;
- final M7 không cho phép fake.

## Service contract

```python
extract_embedding(audio_path: str) -> np.ndarray
```

```python
check_audio_quality(
    audio_path: str,
    expected_text: str,
    transcribe_fn=None
) -> dict
```

Result:

```json
{
  "pass": false,
  "checks": {
    "duration_ok": true,
    "speech_ratio_ok": false,
    "snr_ok": false,
    "clipping_ok": true,
    "content_match_ok": true
  },
  "metrics": {
    "duration_sec": 5.1,
    "speech_ratio": 0.42,
    "snr_db": 6.7,
    "clipping_ratio": 0.0,
    "content_wer": 0.0
  },
  "reasons": ["Âm thanh quá nhiễu."]
}
```

Content normalization:
- Unicode normalization;
- lowercase;
- bỏ punctuation;
- normalize whitespace;
- word-level WER bằng `jiwer`;
- giữ dấu tiếng Việt.

Threshold đọc `app/models/enrollment_config.json`.

```python
enroll_user(
    employee_id: str,
    audio_paths: list[str],
    expected_texts: list[str],
    transcribe_fn=None
) -> dict
```

Rules:
- validate list length;
- quality check tất cả;
- có fail -> không ghi profile;
- all pass -> embedding từng sample -> mean profile -> atomic write `{employee_id}.npy`.

Failure:

```json
{
  "success": false,
  "failed_items": [
    {"index": 2, "checks": {}, "reasons": []}
  ]
}
```

```python
verify(audio_path: str, claimed_employee_id: str) -> dict
identify(audio_path: str) -> dict
```

Threshold lấy `speaker_config.json`.

Output verify:
`{"is_match": bool, "score": float}`.

Output identify:
`{"employee_id": "NV001"|null, "score": float}`.

Management helpers:

```python
delete_voice_profile(employee_id: str) -> bool
has_voice_profile(employee_id: str) -> bool
```

Re-enroll atomic có thể do service hỗ trợ bằng temp profile path/replace helper.

## Placeholder config
Nếu Module A chưa bàn giao, chỉ dùng test config trong fixtures; không commit threshold giả như final.

## Prompt dán cho Codex

Bạn đang làm **M1 בלבד**. Đọc AGENTS.md. Implement shared audio normalization + speaker service theo contract. Dùng deterministic fake backend cho development/tests. Không tải pretrained speaker model. Không hardcode similarity threshold. `check_audio_quality` đọc config và nhận `transcribe_fn` injection để M1 chưa phụ thuộc M2. Viết deterministic unit tests. Không code routes/API/M2/M3.

## Acceptance
- normalize fixture thành 16 kHz mono WAV;
- silent/too-short fail;
- good fixture pass với injected transcript;
- wrong transcript fail content;
- enrollment fail -> không `.npy`;
- enrollment pass -> `.npy`;
- missing profile fail cleanly;
- fake verify/identify deterministic;
- threshold đọc file.

Commands:

```bash
cd backend
python -m compileall app
pytest -q
```

## Manual review
- [ ] không model download;
- [ ] không threshold literal;
- [ ] shared normalization;
- [ ] no partial profile;
- [ ] commit `feat(M1): speaker service and enrollment quality`.
