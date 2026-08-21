import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ApiError,
  type ChatResponse,
  getEmployees,
  resolveApiUrl,
  sendVoiceCommand,
} from '../api/client'
import { Chat } from './Chat'

vi.mock('../api/client', async (importOriginal) => {
  const original = await importOriginal<typeof import('../api/client')>()
  return {
    ...original,
    getEmployees: vi.fn(),
    resolveApiUrl: vi.fn(),
    sendVoiceCommand: vi.fn(),
  }
})

vi.mock('../components/AudioRecorder', () => ({
  AudioRecorder: ({
    disabled,
    hasRecording,
    onStarted,
    onRecorded,
    startLabel,
    rerecordLabel,
  }: {
    disabled: boolean
    hasRecording: boolean
    onStarted: (index: number) => void
    onRecorded: (index: number, blob: Blob, mimeType: string) => void
    startLabel: string
    rerecordLabel: string
  }) => (
    <button
      type="button"
      aria-label="record-command"
      disabled={disabled}
      onClick={() => {
        onStarted(0)
        onRecorded(0, new Blob(['voice command'], { type: 'audio/webm' }), 'audio/webm')
      }}
    >
      {hasRecording ? rerecordLabel : startLabel}
    </button>
  ),
}))

const employees = [
  { id: 'NV001', name: 'Nguyễn An', voice_enrolled: true },
  { id: 'NV002', name: 'Trần Bình', voice_enrolled: true },
]

const generalResponse: ChatResponse = {
  success: true,
  text_asr: 'Hướng dẫn tôi cách xin VPN',
  function_called: 'answer_faq',
  auth_type: null,
  auth_passed: null,
  employee_id: null,
  speaker_score: null,
  response_text: 'Hãy tạo yêu cầu trên cổng IT.',
  audio_reply_url: '/api/audio/general.mp3',
}

async function renderPage() {
  const user = userEvent.setup()
  const result = render(<Chat />)
  await screen.findByRole('option', { name: 'NV001 - Nguyễn An' })
  return { user, ...result }
}

async function recordCommand(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: 'record-command' }))
  await screen.findByText('Đã ghi · audio/webm')
}

describe('Chat page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(getEmployees).mockResolvedValue(employees)
    vi.mocked(sendVoiceCommand).mockResolvedValue(generalResponse)
    vi.mocked(resolveApiUrl).mockImplementation(
      (path) => `http://127.0.0.1:8000${path}`,
    )
  })

  it('loads the optional claim selector and starts with a disabled send button', async () => {
    await renderPage()

    expect(screen.getByRole('button', { name: 'record-command' })).toHaveTextContent(
      'Bắt đầu nói',
    )
    expect(screen.getByRole('button', { name: 'Gửi yêu cầu' })).toBeDisabled()
    expect(screen.getByLabelText('Xác nhận danh tính khi cần')).toHaveValue('')
    expect(screen.getByText(/Đây chỉ là thông tin khai báo/)).toBeInTheDocument()
  })

  it('renders playback, replaces the old command, and revokes URLs on re-record/unmount', async () => {
    const { user, unmount } = await renderPage()
    await recordCommand(user)

    expect(document.querySelectorAll('audio')).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'record-command' })).toHaveTextContent(
      'Ghi lại',
    )
    await user.click(screen.getByRole('button', { name: 'record-command' }))
    expect(URL.revokeObjectURL).toHaveBeenCalled()

    const callsBeforeUnmount = vi.mocked(URL.revokeObjectURL).mock.calls.length
    unmount()
    expect(vi.mocked(URL.revokeObjectURL).mock.calls.length).toBeGreaterThan(
      callsBeforeUnmount,
    )
  })

  it('prevents double submit while pending and renders a General result with TTS', async () => {
    const { user } = await renderPage()
    let completeRequest: (response: ChatResponse) => void = () => undefined
    vi.mocked(sendVoiceCommand).mockImplementation(
      () => new Promise((resolve) => { completeRequest = resolve }),
    )
    await recordCommand(user)

    await user.click(screen.getByRole('button', { name: 'Gửi yêu cầu' }))
    expect(screen.getByRole('button', { name: 'Đang gửi…' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'record-command' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Đang gửi…' }))
    expect(sendVoiceCommand).toHaveBeenCalledOnce()

    completeRequest(generalResponse)
    const result = await screen.findByTestId('chat-result')
    expect(within(result).getByText('Hướng dẫn tôi cách xin VPN')).toBeInTheDocument()
    expect(within(result).getByText('Hãy tạo yêu cầu trên cổng IT.')).toBeInTheDocument()
    expect(within(result).getByText('Không yêu cầu xác thực')).toBeInTheDocument()
    expect(within(result).getByText(/answer_faq/)).toBeInTheDocument()
    expect(within(result).getAllByText('Không có')).toHaveLength(2)
    expect(result.querySelector('audio')).toHaveAttribute(
      'src',
      'http://127.0.0.1:8000/api/audio/general.mp3',
    )
  })

  it('sends the selected employee only as a claim and renders SV success details', async () => {
    const { user } = await renderPage()
    const response: ChatResponse = {
      ...generalResponse,
      function_called: 'reset_password',
      auth_type: 'SV',
      auth_passed: true,
      employee_id: 'NV001',
      speaker_score: 0.91,
      response_text: 'Mật khẩu đã được đặt lại.',
      audio_reply_url: null,
    }
    vi.mocked(sendVoiceCommand).mockResolvedValue(response)
    await user.selectOptions(screen.getByLabelText('Xác nhận danh tính khi cần'), 'NV001')
    await recordCommand(user)
    await user.click(screen.getByRole('button', { name: 'Gửi yêu cầu' }))

    const result = await screen.findByTestId('chat-result')
    expect(sendVoiceCommand).toHaveBeenCalledWith(expect.any(Blob), 'NV001')
    expect(within(result).getByText('Xác minh người nói: Thành công')).toBeInTheDocument()
    expect(within(result).getByText('NV001 - Nguyễn An')).toBeInTheDocument()
    expect(within(result).getByText('0.910')).toBeInTheDocument()
    expect(within(result).getByText('Không tạo được phản hồi âm thanh.')).toBeInTheDocument()
  })

  it('renders an HTTP 200 SV denial as a business result instead of an API error', async () => {
    const { user } = await renderPage()
    vi.mocked(sendVoiceCommand).mockResolvedValue({
      ...generalResponse,
      success: false,
      function_called: 'reset_password',
      auth_type: 'SV',
      auth_passed: false,
      response_text: 'Vui lòng cung cấp mã nhân viên để xác thực giọng nói.',
      audio_reply_url: null,
    })
    await recordCommand(user)
    await user.click(screen.getByRole('button', { name: 'Gửi yêu cầu' }))

    const result = await screen.findByTestId('chat-result')
    expect(within(result).getByText('Xác minh người nói: Thất bại')).toBeInTheDocument()
    expect(
      within(result).getByText('Vui lòng cung cấp mã nhân viên để xác thực giọng nói.'),
    ).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('uses SID identity from the response even when a different claim is selected', async () => {
    const { user } = await renderPage()
    vi.mocked(sendVoiceCommand).mockResolvedValue({
      ...generalResponse,
      function_called: 'check_leave_days',
      auth_type: 'SID',
      auth_passed: true,
      employee_id: 'NV002',
      speaker_score: 0.842,
      response_text: 'Bạn còn 12 ngày phép.',
      audio_reply_url: null,
    })
    await user.selectOptions(screen.getByLabelText('Xác nhận danh tính khi cần'), 'NV001')
    await recordCommand(user)
    await user.click(screen.getByRole('button', { name: 'Gửi yêu cầu' }))

    const result = await screen.findByTestId('chat-result')
    expect(within(result).getByText('Nhận diện người nói: Thành công')).toBeInTheDocument()
    expect(within(result).getByText('NV002 - Trần Bình')).toBeInTheDocument()
    expect(within(result).queryByText('NV001 - Nguyễn An')).not.toBeInTheDocument()
    expect(within(result).getByText('0.842')).toBeInTheDocument()
  })

  it('does not present the selected claim as identity after SID failure', async () => {
    const { user } = await renderPage()
    vi.mocked(sendVoiceCommand).mockResolvedValue({
      ...generalResponse,
      success: false,
      function_called: 'check_leave_days',
      auth_type: 'SID',
      auth_passed: false,
      employee_id: null,
      speaker_score: 0.31,
      response_text: 'Không nhận diện được người dùng.',
      audio_reply_url: null,
    })
    await user.selectOptions(screen.getByLabelText('Xác nhận danh tính khi cần'), 'NV001')
    await recordCommand(user)
    await user.click(screen.getByRole('button', { name: 'Gửi yêu cầu' }))

    const result = await screen.findByTestId('chat-result')
    expect(within(result).getByText('Nhận diện người nói: Thất bại')).toBeInTheDocument()
    expect(within(result).queryByText('NV001 - Nguyễn An')).not.toBeInTheDocument()
    expect(within(result).getByText('Không có')).toBeInTheDocument()
  })

  it('shows controlled API errors while retaining the command for retry', async () => {
    const { user } = await renderPage()
    vi.mocked(sendVoiceCommand).mockRejectedValue(
      new ApiError('Dịch vụ ASR không khả dụng.', 503),
    )
    await recordCommand(user)
    await user.click(screen.getByRole('button', { name: 'Gửi yêu cầu' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Dịch vụ ASR không khả dụng.')
    expect(screen.getByRole('button', { name: 'Gửi yêu cầu' })).toBeEnabled()
    expect(document.querySelector('audio')).toBeInTheDocument()
  })
})
