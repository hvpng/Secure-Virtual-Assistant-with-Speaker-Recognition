# MODULE 2 — ASR & TTS Service

## Mục tiêu
PhoWhisper tiếng Việt + gTTS với interface đơn giản và dùng canonical normalized audio.

## Files
- `backend/app/services/asr_tts_service.py`
- tests M2
- cập nhật config/README nếu cần

## ASR

```python
speech_to_text(audio_path: str) -> str
```

Rules:
1. input normalize qua `audio_utils.normalize_audio()` nếu chưa là canonical WAV;
2. primary: PhoWhisper-small qua HuggingFace `transformers`;
3. fallback: Whisper multilingual `base` qua `openai-whisper`;
4. không cố load PhoWhisper checkpoint bằng `openai-whisper`;
5. model load một lần/lazy singleton;
6. `ASR_DEVICE=auto|cuda|cpu`;
7. nếu fallback CPU phải log rõ;
8. output string `.strip()`.

## TTS

Canonical:

```python
text_to_speech(text: str) -> str
```

- gTTS `lang="vi"`;
- tự sinh unique filename trong `backend/data/generated_audio/`;
- return local path;
- M4 chịu trách nhiệm serve URL;
- caller không truyền `output_path`.

## M1 integration
Sau M2, composition/M4 truyền `speech_to_text` vào M1 `check_audio_quality/enroll_user`.

Không tạo import cycle M1 <-> M2.

## Tests

Unit:
- mock model để test flow;
- TTS output path unique;
- normalization được dùng.

Live smoke:
- marker riêng, không chạy mặc định;
- fixture tiếng Việt;
- assert transcript không rỗng.

## Prompt dán cho Codex

Bạn đang làm **M2 בלבד**. Đọc AGENTS.md. Implement `asr_tts_service.py` với PhoWhisper-small qua `transformers`, Whisper multilingual `base` chỉ là fallback qua `openai-whisper`, và gTTS. Dùng shared `audio_utils.normalize_audio()` từ M1, không viết converter riêng. Canonical TTS signature chỉ nhận text và tự tạo output path. Load model một lần. Tách deterministic unit test khỏi live smoke. Không code API/M3/M4.

## Acceptance

```bash
cd backend
python -m compileall app
pytest -q
```

Manual live smoke:
- transcript tiếng Việt không rỗng;
- mp3 nghe được.

## Review
- [ ] không duplicate conversion;
- [ ] PhoWhisper dùng transformers;
- [ ] fallback đúng;
- [ ] không load model mỗi call;
- [ ] commit `feat(M2): ASR and TTS services`.
