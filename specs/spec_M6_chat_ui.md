# MODULE 6 — Voice Chat UI

## Mục tiêu
Trang demo chính: voice interaction + auth visibility.

## Backend contracts
- `GET /api/employees`
- `POST /api/chat`
- `audio_reply_url`

## Layout

Header:
- dropdown `Tôi là:` cho **claim SV**;
- employee list từ backend;
- chỉ nên claim employee đã `voice_enrolled=true`.

Chat:
- user bubble;
- assistant bubble;
- có thể hiện ASR transcript nhỏ để demo.

Mic:
- MediaRecorder;
- Chrome/Edge;
- start/stop;
- gửi multipart `audio`;
- gửi `claimed_employee_id` nếu dropdown có chọn.

## Status

Backend non-streaming nên dùng UI states:
- Đang nghe...
- Đang gửi...
- Đang xử lý...
- Hoàn tất / lỗi

Không giả vờ mô tả chính xác internal backend stage nếu backend chưa stream.

## Auth badge

Từ exact ChatResponse.

Pass:
- `auth_passed=true`
- `employee_id` có giá trị
- badge `✅ Đã xác thực/nhận diện: <name>`.

Fail:
- `auth_passed=false`
- badge đỏ.

General:
- `auth_passed=null`
- không badge.

Map ID -> name từ employee list.

## TTS

Khi có `audio_reply_url`:
- attempt autoplay sau user gesture;
- nếu browser chặn autoplay, hiện play button.

## Empty voice state

Employee DB có thể có seed records nhưng chưa enroll.

Do đó dùng:

```ts
employees.filter(e => e.voice_enrolled).length === 0
```

Không dùng `employees.length === 0`.

## Prompt dán cho Codex

Bạn đang làm **M6 בלבד**. Đọc AGENTS.md và exact ChatResponse. Implement voice chat cho Chrome/Edge với MediaRecorder. Claim dropdown chỉ phục vụ SV; SID do backend identify. Render auth badge từ response, không suy luận auth ở frontend. Empty state dựa trên voice_enrolled. TTS autoplay có fallback play button. Không sửa security logic backend.

## Acceptance

```bash
cd frontend
npm run lint
npm run build
npm run dev
```

Manual:
- FAQ no claim -> no badge;
- SV claim field đúng;
- SID không cần claim;
- auth failure badge;
- TTS play;
- mic denied;
- no console error.

## Review
- [ ] frontend không gate action;
- [ ] no identity inferred từ transcript/LLM;
- [ ] no fake auth status;
- [ ] response fields exact;
- [ ] commit `feat(M6): voice chat UI`.
