import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ApiError,
  enrollEmployee,
  getEmployees,
  getEnrollmentScripts,
  reenrollEmployee,
  removeVoiceProfile,
} from '../api/client'
import { Enrollment } from './Enrollment'

vi.mock('../api/client', async (importOriginal) => {
  const original = await importOriginal<typeof import('../api/client')>()
  return {
    ...original,
    enrollEmployee: vi.fn(),
    getEmployees: vi.fn(),
    getEnrollmentScripts: vi.fn(),
    reenrollEmployee: vi.fn(),
    removeVoiceProfile: vi.fn(),
  }
})

vi.mock('../components/AudioRecorder', () => ({
  AudioRecorder: ({
    index,
    disabled,
    hasRecording,
    onStarted,
    onRecorded,
  }: {
    index: number
    disabled: boolean
    hasRecording: boolean
    onStarted: (index: number) => void
    onRecorded: (index: number, blob: Blob, mimeType: string) => void
  }) => (
    <button
      type="button"
      aria-label={`record-${index}`}
      disabled={disabled}
      onClick={() => {
        onStarted(index)
        onRecorded(index, new Blob([`voice-${index}`], { type: 'audio/webm' }), 'audio/webm')
      }}
    >
      {hasRecording ? 'Ghi lại' : 'Bắt đầu ghi'}
    </button>
  ),
}))

const scripts = Array.from({ length: 7 }, (_, index) => ({
  index,
  text: `Nội dung câu ${index + 1}`,
}))

const employees = [
  { id: 'NV001', name: 'Nguyễn An', voice_enrolled: true },
  { id: 'NV002', name: 'Trần Bình', voice_enrolled: false },
]

const passingChecks = {
  duration_ok: true,
  speech_ratio_ok: true,
  snr_ok: true,
  clipping_ok: true,
  content_match_ok: true,
}

async function renderPage() {
  const user = userEvent.setup()
  const result = render(<Enrollment />)
  await screen.findByText('Nội dung câu 1')
  await screen.findByText('Nguyễn An')
  return { user, ...result }
}

async function fillIdentityAndRecordAll(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText('Mã nhân viên'), 'NV900')
  await user.type(screen.getByLabelText('Họ và tên'), 'Lê Minh')
  for (let index = 0; index < 7; index += 1) {
    await user.click(screen.getByRole('button', { name: `record-${index}` }))
  }
}

describe('Enrollment page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(getEnrollmentScripts).mockResolvedValue(scripts)
    vi.mocked(getEmployees).mockResolvedValue(employees)
    vi.mocked(enrollEmployee).mockResolvedValue({ success: true, failed_items: [] })
    vi.mocked(reenrollEmployee).mockResolvedValue({ success: true, failed_items: [] })
    vi.mocked(removeVoiceProfile).mockResolvedValue(undefined)
    vi.mocked(URL.createObjectURL).mockImplementation(
      (blob) => `blob:preview-${blob instanceof Blob ? blob.size : 'media'}`,
    )
    vi.spyOn(window, 'confirm').mockReturnValue(true)
  })

  it('fetches and renders all scripts in index order and demo-safe employees', async () => {
    await renderPage()

    expect(getEnrollmentScripts).toHaveBeenCalledOnce()
    expect(getEmployees).toHaveBeenCalledOnce()
    const cards = screen.getAllByText(/Câu \d\/7/)
    expect(cards.map((card) => card.textContent)).toEqual(
      scripts.map((script) => `Câu ${script.index + 1}/7`),
    )
    expect(screen.getByText('NV001')).toBeInTheDocument()
    expect(screen.getByText('Đã đăng ký')).toBeInTheDocument()
    expect(screen.getByText('Chưa đăng ký')).toBeInTheDocument()
  })

  it('keeps submit disabled when identity fields or recordings are missing', async () => {
    const { user } = await renderPage()
    const submit = screen.getByRole('button', { name: 'Đăng ký giọng nói' })
    expect(submit).toBeDisabled()
    expect(screen.getByText('Vui lòng nhập đầy đủ mã nhân viên và họ tên.')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Mã nhân viên'), 'NV900')
    await user.type(screen.getByLabelText('Họ và tên'), 'Lê Minh')

    expect(submit).toBeDisabled()
    expect(screen.getByText('Vui lòng ghi âm đầy đủ 7 câu trước khi đăng ký.')).toBeInTheDocument()
    expect(enrollEmployee).not.toHaveBeenCalled()
  })

  it('maps structured quality failures to the correct zero-based card and keeps blobs', async () => {
    const { user } = await renderPage()
    vi.mocked(enrollEmployee).mockResolvedValue({
      success: false,
      failed_items: [
        {
          index: 2,
          checks: { ...passingChecks, speech_ratio_ok: false },
          reasons: ['Tỷ lệ giọng nói quá thấp.'],
        },
      ],
    })
    await fillIdentityAndRecordAll(user)

    await user.click(screen.getByRole('button', { name: 'Đăng ký giọng nói' }))

    const failedCard = screen.getByText('Câu 3/7').closest('article')
    expect(failedCard).not.toBeNull()
    expect(within(failedCard!).getByText('Tỷ lệ giọng nói quá thấp.')).toBeInTheDocument()
    expect(within(failedCard!).getByText('✗ Tỷ lệ giọng nói')).toBeInTheDocument()
    expect(document.querySelectorAll('audio')).toHaveLength(7)
    expect(screen.getAllByRole('button', { name: /record-/ })).toHaveLength(7)
  })

  it('lets a failed recording be replaced and revokes its previous object URL', async () => {
    const { user } = await renderPage()
    await user.click(screen.getByRole('button', { name: 'record-2' }))
    await user.click(screen.getByRole('button', { name: 'record-2' }))

    expect(URL.revokeObjectURL).toHaveBeenCalled()
  })

  it('clears recordings and refreshes employees after successful enrollment', async () => {
    const { user } = await renderPage()
    vi.mocked(getEmployees).mockResolvedValueOnce([
      ...employees,
      { id: 'NV900', name: 'Lê Minh', voice_enrolled: true },
    ])
    await fillIdentityAndRecordAll(user)

    await user.click(screen.getByRole('button', { name: 'Đăng ký giọng nói' }))

    expect(await screen.findByText('Đăng ký giọng nói thành công.')).toBeInTheDocument()
    expect(getEmployees).toHaveBeenCalledTimes(2)
    expect(screen.getByText('Đã ghi 0/7 câu')).toBeInTheDocument()
    expect(screen.getByText('NV900')).toBeInTheDocument()
  })

  it('prefills explicit re-enrollment mode, calls the correct API, and can cancel', async () => {
    const { user } = await renderPage()
    await user.click(screen.getByRole('button', { name: 'Đăng ký lại' }))

    expect(screen.getByRole('heading', { name: 'Đăng ký lại hồ sơ' })).toBeInTheDocument()
    expect(screen.getByLabelText('Mã nhân viên')).toHaveValue('NV001')
    expect(screen.getByLabelText('Mã nhân viên')).toHaveAttribute('readonly')
    expect(screen.getByLabelText('Họ và tên')).toHaveValue('Nguyễn An')

    for (let index = 0; index < 7; index += 1) {
      await user.click(screen.getByRole('button', { name: `record-${index}` }))
    }
    await user.click(screen.getByRole('button', { name: 'Gửi đăng ký lại' }))
    await waitFor(() => expect(reenrollEmployee).toHaveBeenCalledWith('NV001', expect.any(Array)))
    expect(enrollEmployee).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Đăng ký lại' }))
    await user.click(screen.getByRole('button', { name: 'Hủy đăng ký lại' }))
    expect(screen.getByRole('heading', { name: 'Đăng ký hồ sơ mới' })).toBeInTheDocument()
    expect(screen.getByLabelText('Mã nhân viên')).toHaveValue('')
    expect(screen.getByLabelText('Mã nhân viên')).not.toHaveAttribute('readonly')
  })

  it('refreshes after profile removal but does not fake-update on delete failure', async () => {
    const { user } = await renderPage()
    vi.mocked(getEmployees).mockResolvedValueOnce([
      { ...employees[0], voice_enrolled: false },
      employees[1],
    ])

    await user.click(screen.getByRole('button', { name: 'Xóa hồ sơ giọng nói' }))
    expect(removeVoiceProfile).toHaveBeenCalledWith('NV001')
    expect(await screen.findByText('Đã xóa hồ sơ giọng nói. Hồ sơ nhân viên vẫn được giữ lại.')).toBeInTheDocument()
    expect(screen.getAllByText('Chưa đăng ký')).toHaveLength(2)

    vi.mocked(getEmployees).mockResolvedValue([
      { ...employees[0], voice_enrolled: true },
      employees[1],
    ])
    await user.click(screen.getAllByRole('button', { name: 'Làm mới' })[0])
    await screen.findByRole('button', { name: 'Xóa hồ sơ giọng nói' })
    vi.mocked(removeVoiceProfile).mockRejectedValue(new ApiError('Không xóa được profile.', 500))
    await user.click(screen.getByRole('button', { name: 'Xóa hồ sơ giọng nói' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Không xóa được profile.')
    expect(screen.getByText('Đã đăng ký')).toBeInTheDocument()
  })

  it('renders controlled API failure messages and revokes object URLs on unmount', async () => {
    const { user, unmount } = await renderPage()
    await user.type(screen.getByLabelText('Mã nhân viên'), 'NV900')
    await user.type(screen.getByLabelText('Họ và tên'), 'Lê Minh')
    await user.click(screen.getByRole('button', { name: 'record-0' }))
    vi.mocked(enrollEmployee).mockRejectedValue(new ApiError('Máy chủ đang bận.', 503))
    for (let index = 1; index < 7; index += 1) {
      await user.click(screen.getByRole('button', { name: `record-${index}` }))
    }

    await user.click(screen.getByRole('button', { name: 'Đăng ký giọng nói' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Máy chủ đang bận.')

    const revokeCallsBeforeUnmount = vi.mocked(URL.revokeObjectURL).mock.calls.length
    unmount()
    expect(vi.mocked(URL.revokeObjectURL).mock.calls.length).toBeGreaterThan(
      revokeCallsBeforeUnmount,
    )
  })
})
