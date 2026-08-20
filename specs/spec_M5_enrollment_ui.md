# MODULE 5 — Enrollment & Voice Profile Management UI

## Mục tiêu
UI đăng ký 7 câu + feedback quality + list/re-enroll/remove voice profile.

## Backend contracts
- `GET /api/enrollment-scripts`
- `GET /api/employees`
- `POST /api/enroll`
- `POST /api/employees/{id}/reenroll`
- `DELETE /api/employees/{id}/voice-profile`

Không hardcode 7 câu ở frontend.

## Enrollment UX

Form:
- Name
- Employee ID

Flow:
1. fetch scripts;
2. hiển thị `Câu x/7`;
3. record qua MediaRecorder;
4. target Chrome/Edge;
5. khoảng 5 giây, có stop thủ công;
6. playback;
7. re-record;
8. next chỉ khi có blob;
9. submit exactly 7 repeated `audio_files`.

Frontend không tự quyết định quality pass; backend là authority.

## Failure UX

Backend trả:

```json
{
  "success": false,
  "failed_items": [
    {
      "index": 2,
      "checks": {},
      "reasons": []
    }
  ]
}
```

Behavior:
- giữ recordings không fail;
- chỉ reset/re-record failed indices;
- show backend `reasons`;
- có thể map check key thành label thân thiện nhưng không thay mất reason gốc.

## Management section

List:
- ID
- name
- badge `Đã enroll` / `Chưa enroll`

Actions:
- `Re-enroll`: dùng lại recorder 7 câu.
- `Xóa voice profile`: confirm -> DELETE; employee vẫn tồn tại.

Không delete employee.

## Shared recorder
Nên tách `frontend/src/components/AudioRecorder.tsx` nếu giúp tái sử dụng.

Không thêm recorder library nặng.

## Prompt dán cho Codex

Bạn đang làm **M5 בלבד**. Đọc AGENTS.md và exact M4 API. Implement Enrollment page + voice-profile management. Scripts fetch từ backend. Dùng MediaRecorder built-in cho Chrome/Edge. Khi backend trả failed_items 0-based, chỉ record lại đúng item fail. Thêm re-enroll và remove voice-profile, không delete employee. Không sửa backend contract trừ khi phát hiện bug thật; nếu có, nêu rõ trong summary. Không code Chat page M6.

## Acceptance

```bash
cd frontend
npm run lint
npm run build
npm run dev
```

Manual:
- mic denied -> message rõ;
- record/play 7 audio;
- failed item UX;
- success refresh state;
- re-enroll;
- remove profile;
- no console errors.

## Review
- [ ] no hardcoded scripts;
- [ ] index 0-based;
- [ ] giữ pass recordings;
- [ ] remove không xóa employee;
- [ ] FormData đúng;
- [ ] commit `feat(M5): enrollment and voice profile management UI`.
