import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  buildEnrollmentFormData,
  buildVoiceCommandFormData,
  enrollEmployee,
  getEmployees,
  getEnrollmentScripts,
  reenrollEmployee,
  removeVoiceProfile,
  resolveApiUrl,
  sendVoiceCommand,
} from './client'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('M5 API client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('fetches employees and sorts enrollment scripts by their server index', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse({ employees: [{ id: 'NV001', name: 'An', voice_enrolled: true }] }))
      .mockResolvedValueOnce(jsonResponse({ scripts: [{ index: 2, text: 'Ba' }, { index: 0, text: 'Một' }] }))

    await expect(getEmployees()).resolves.toEqual([
      { id: 'NV001', name: 'An', voice_enrolled: true },
    ])
    await expect(getEnrollmentScripts()).resolves.toEqual([
      { index: 0, text: 'Một' },
      { index: 2, text: 'Ba' },
    ])
  })

  it('builds exactly repeated audio_files in index order without expected_text', () => {
    const formData = buildEnrollmentFormData(
      [6, 2, 0, 5, 1, 4, 3].map((index) => ({
        index,
        blob: new Blob([String(index)], { type: 'audio/webm' }),
      })),
      { id: ' NV001 ', name: ' Nguyễn An ' },
    )

    expect(formData.get('employee_id')).toBe('NV001')
    expect(formData.get('name')).toBe('Nguyễn An')
    expect(formData.getAll('audio_files')).toHaveLength(7)
    expect(formData.getAll('audio_files').map((value) => (value as File).name)).toEqual([
      'enroll_0.webm',
      'enroll_1.webm',
      'enroll_2.webm',
      'enroll_3.webm',
      'enroll_4.webm',
      'enroll_5.webm',
      'enroll_6.webm',
    ])
    expect([...formData.keys()]).not.toContain('expected_text')
  })

  it('uses the initial enrollment endpoint without manually setting multipart headers', async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse({ success: true, failed_items: [] }))

    await enrollEmployee(
      { id: 'NV001', name: 'Nguyễn An' },
      [{ index: 0, blob: new Blob(['audio']) }],
    )

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/enroll',
      expect.objectContaining({ method: 'POST', body: expect.any(FormData) }),
    )
    expect(vi.mocked(fetch).mock.calls[0][1]).not.toHaveProperty('headers')
  })

  it('uses the re-enrollment endpoint and sends only audio_files', async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse({ success: true, failed_items: [] }))

    await reenrollEmployee('NV 01', [{ index: 0, blob: new Blob(['audio']) }])

    const [url, init] = vi.mocked(fetch).mock.calls[0]
    const formData = init?.body as FormData
    expect(url).toBe('http://127.0.0.1:8000/api/employees/NV%2001/reenroll')
    expect([...formData.keys()]).toEqual(['audio_files'])
  })

  it('uses DELETE for voice profile removal', async () => {
    vi.mocked(fetch).mockResolvedValue(
      jsonResponse({ success: true, employee_id: 'NV001', voice_enrolled: false }),
    )

    await removeVoiceProfile('NV001')

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/employees/NV001/voice-profile',
      { method: 'DELETE' },
    )
  })

  it('preserves a structured quality failure even when HTTP status is 400', async () => {
    const failure = {
      success: false,
      failed_items: [
        {
          index: 2,
          checks: {
            duration_ok: true,
            speech_ratio_ok: true,
            snr_ok: true,
            clipping_ok: true,
            content_match_ok: false,
          },
          reasons: ['Nội dung không khớp.'],
        },
      ],
    }
    vi.mocked(fetch).mockResolvedValue(jsonResponse(failure, 400))

    await expect(
      enrollEmployee({ id: 'NV001', name: 'An' }, [{ index: 0, blob: new Blob(['x']) }]),
    ).resolves.toEqual(failure)
  })

  it('maps FastAPI and network failures to controlled errors', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse({ detail: 'employee_id không hợp lệ.' }, 400))
      .mockRejectedValueOnce(new TypeError('network down'))

    await expect(getEmployees()).rejects.toEqual(
      expect.objectContaining({ message: 'employee_id không hợp lệ.', status: 400 }),
    )
    await expect(getEmployees()).rejects.toEqual(
      expect.objectContaining({ message: 'Không thể kết nối đến máy chủ.', status: null }),
    )
  })

  it('uses a safe fallback for non-JSON server responses', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('<html>error</html>', { status: 503 }))

    await expect(getEmployees()).rejects.toThrow(
      'Không thể hoàn thành yêu cầu. Vui lòng thử lại.',
    )
  })

  it('builds the voice command with only audio when no claim is selected', () => {
    const formData = buildVoiceCommandFormData(
      new Blob(['voice'], { type: 'audio/webm;codecs=opus' }),
    )

    expect([...formData.keys()]).toEqual(['audio'])
    expect((formData.get('audio') as File).name).toBe('voice-command.webm')
    expect(formData.get('claimed_employee_id')).toBeNull()
  })

  it('appends an optional trimmed claim without frontend intent or auth fields', () => {
    const formData = buildVoiceCommandFormData(new Blob(['voice']), ' NV001 ')

    expect([...formData.keys()]).toEqual(['audio', 'claimed_employee_id'])
    expect(formData.get('claimed_employee_id')).toBe('NV001')
    expect(formData.get('function_name')).toBeNull()
    expect(formData.get('auth_type')).toBeNull()
    expect(formData.get('auth_passed')).toBeNull()
    expect(formData.get('employee_id')).toBeNull()
    expect(formData.get('transcript')).toBeNull()
  })

  it('returns an HTTP 200 auth denial as a valid chat result', async () => {
    const denial = {
      success: false,
      text_asr: 'Tôi muốn reset mật khẩu',
      function_called: 'reset_password',
      auth_type: 'SV',
      auth_passed: false,
      employee_id: null,
      speaker_score: null,
      response_text: 'Vui lòng cung cấp mã nhân viên.',
      audio_reply_url: null,
    }
    vi.mocked(fetch).mockResolvedValue(jsonResponse(denial))

    await expect(sendVoiceCommand(new Blob(['voice']))).resolves.toEqual(denial)
    expect(vi.mocked(fetch).mock.calls[0][1]).not.toHaveProperty('headers')
  })

  it('rejects empty audio, HTTP errors, and malformed successful chat responses', async () => {
    await expect(sendVoiceCommand(new Blob([]))).rejects.toThrow('Bản ghi âm không được rỗng.')

    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse({ detail: 'File quá lớn.' }, 413))
      .mockResolvedValueOnce(jsonResponse({ success: true }))
      .mockResolvedValueOnce(new Response('<html>unavailable</html>', { status: 503 }))

    await expect(sendVoiceCommand(new Blob(['voice']))).rejects.toEqual(
      expect.objectContaining({ message: 'File quá lớn.', status: 413 }),
    )
    await expect(sendVoiceCommand(new Blob(['voice']))).rejects.toThrow(
      'Không thể xử lý yêu cầu. Vui lòng thử lại.',
    )
    await expect(sendVoiceCommand(new Blob(['voice']))).rejects.toThrow(
      'Không thể hoàn thành yêu cầu. Vui lòng thử lại.',
    )
  })

  it('resolves backend TTS paths without allowing a different origin', () => {
    expect(resolveApiUrl('/api/audio/reply.mp3')).toBe(
      'http://127.0.0.1:8000/api/audio/reply.mp3',
    )
    expect(() => resolveApiUrl('https://attacker.example/reply.mp3')).toThrow(
      'Đường dẫn phản hồi âm thanh không hợp lệ.',
    )
  })
})
