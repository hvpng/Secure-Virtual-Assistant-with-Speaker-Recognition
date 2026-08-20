import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { StrictMode, useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { selectSupportedAudioMimeType } from '../utils/mediaRecorder'
import { AudioRecorder } from './AudioRecorder'

class ControlledMediaRecorder extends EventTarget {
  static instances: ControlledMediaRecorder[] = []
  static isTypeSupported = vi.fn((mimeType: string) => mimeType === 'audio/webm;codecs=opus')

  state: RecordingState = 'inactive'
  mimeType: string
  stop = vi.fn(() => {
    this.state = 'inactive'
  })

  constructor(_stream: MediaStream, options?: MediaRecorderOptions) {
    super()
    this.mimeType = options?.mimeType ?? 'audio/webm'
    ControlledMediaRecorder.instances.push(this)
  }

  start() {
    this.state = 'recording'
  }

  emitData(contents = 'voice') {
    const event = new Event('dataavailable') as BlobEvent
    Object.defineProperty(event, 'data', {
      value: new Blob([contents], { type: this.mimeType }),
    })
    this.dispatchEvent(event)
  }

  emitStop() {
    this.dispatchEvent(new Event('stop'))
  }

  emitError() {
    this.dispatchEvent(new Event('error'))
  }
}

type HarnessProps = {
  onRecorded?: (index: number, blob: Blob, mimeType: string) => void
  onError?: (index: number, message: string) => void
  events?: string[]
}

function Harness({ onRecorded = vi.fn(), onError = vi.fn(), events = [] }: HarnessProps) {
  const [active, setActive] = useState<number | null>(null)
  return (
    <>
      {[0, 1].map((index) => (
        <AudioRecorder
          key={index}
          index={index}
          isRecording={active === index}
          disabled={active !== null}
          hasRecording={false}
          onStarted={setActive}
          onRecorded={(recordingIndex, blob, mimeType) => {
            events.push('parent-completed')
            setActive(null)
            onRecorded(recordingIndex, blob, mimeType)
          }}
          onError={(recordingIndex, message) => {
            events.push('parent-error')
            setActive(null)
            onError(recordingIndex, message)
          }}
        />
      ))}
    </>
  )
}

async function startFirstRecording(props: HarnessProps = {}) {
  const result = render(
    <StrictMode>
      <Harness {...props} />
    </StrictMode>,
  )
  fireEvent.click(screen.getAllByRole('button', { name: 'Bắt đầu ghi' })[0])
  await screen.findByRole('button', { name: 'Dừng ghi' })
  await waitFor(() => expect(ControlledMediaRecorder.instances).toHaveLength(1))
  return { ...result, recorder: ControlledMediaRecorder.instances[0] }
}

describe('AudioRecorder', () => {
  let trackStop: ReturnType<typeof vi.fn>
  let lifecycleEvents: string[]

  beforeEach(() => {
    ControlledMediaRecorder.instances = []
    ControlledMediaRecorder.isTypeSupported.mockClear()
    lifecycleEvents = []
    trackStop = vi.fn(() => lifecycleEvents.push('track-stopped'))
    vi.stubGlobal('MediaRecorder', ControlledMediaRecorder)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: trackStop }],
        }),
      },
    })
  })

  it('prefers Opus WebM only after checking browser support', () => {
    expect(selectSupportedAudioMimeType()).toBe('audio/webm;codecs=opus')
    expect(ControlledMediaRecorder.isTypeSupported).toHaveBeenCalledWith(
      'audio/webm;codecs=opus',
    )
  })

  it('uses the same recorder for start and stop, retains final data, then exits recording', async () => {
    const onRecorded = vi.fn()
    const { recorder } = await startFirstRecording({
      onRecorded,
      events: lifecycleEvents,
    })

    expect(recorder.state).toBe('recording')
    expect(screen.getByRole('button', { name: 'Bắt đầu ghi' })).toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: 'Dừng ghi' }))

    expect(recorder.stop).toHaveBeenCalledOnce()
    expect(trackStop).not.toHaveBeenCalled()
    expect(onRecorded).not.toHaveBeenCalled()

    // A second click before onstop must not call stop() twice or clear the active instance.
    fireEvent.click(screen.getByRole('button', { name: 'Dừng ghi' }))
    expect(recorder.stop).toHaveBeenCalledOnce()

    recorder.emitData('final voice chunk')
    expect(trackStop).not.toHaveBeenCalled()
    recorder.emitStop()

    await waitFor(() => expect(onRecorded).toHaveBeenCalledOnce())
    const completedBlob = onRecorded.mock.calls[0][1] as Blob
    expect(completedBlob.size).toBeGreaterThan(0)
    expect(completedBlob.type).toBe('audio/webm;codecs=opus')
    expect(onRecorded).toHaveBeenCalledWith(
      0,
      expect.any(Blob),
      'audio/webm;codecs=opus',
    )
    expect(trackStop).toHaveBeenCalledOnce()
    expect(lifecycleEvents).toEqual(['parent-completed', 'track-stopped'])
    expect(screen.queryByRole('button', { name: 'Dừng ghi' })).not.toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Bắt đầu ghi' })).toHaveLength(2)
  })

  it('stops the active recorder when its state is paused', async () => {
    const { recorder } = await startFirstRecording()
    recorder.state = 'paused'

    fireEvent.click(screen.getByRole('button', { name: 'Dừng ghi' }))

    expect(recorder.stop).toHaveBeenCalledOnce()
  })

  it('waits for finalization before stopping tracks when unmounted', async () => {
    const { recorder, unmount } = await startFirstRecording()

    unmount()

    expect(recorder.stop).toHaveBeenCalledOnce()
    expect(trackStop).not.toHaveBeenCalled()
    recorder.emitData('final unmount chunk')
    recorder.emitStop()
    expect(trackStop).toHaveBeenCalledOnce()
  })

  it('handles recorder errors with track cleanup and releases the UI state', async () => {
    const onError = vi.fn()
    const { recorder } = await startFirstRecording({ onError, events: lifecycleEvents })

    recorder.emitError()

    await waitFor(() =>
      expect(onError).toHaveBeenCalledWith(0, 'Không thể ghi âm. Vui lòng thử lại.'),
    )
    expect(trackStop).toHaveBeenCalledOnce()
    expect(lifecycleEvents).toEqual(['parent-error', 'track-stopped'])
    expect(screen.queryByRole('button', { name: 'Dừng ghi' })).not.toBeInTheDocument()
  })

  it('rejects an empty finalized Blob and still cleans up the stream', async () => {
    const onError = vi.fn()
    const { recorder } = await startFirstRecording({ onError })
    fireEvent.click(screen.getByRole('button', { name: 'Dừng ghi' }))

    recorder.emitStop()

    await waitFor(() =>
      expect(onError).toHaveBeenCalledWith(
        0,
        'Không thu được dữ liệu âm thanh. Vui lòng ghi lại.',
      ),
    )
    expect(trackStop).toHaveBeenCalledOnce()
  })

  it('shows a controlled callback error when MediaRecorder is unavailable', async () => {
    const onError = vi.fn()
    vi.stubGlobal('MediaRecorder', undefined)
    render(<Harness onError={onError} />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Bắt đầu ghi' })[0])

    await waitFor(() =>
      expect(onError).toHaveBeenCalledWith(
        0,
        'Trình duyệt không hỗ trợ ghi âm bằng MediaRecorder.',
      ),
    )
  })

  it('reports denied microphone permission without crashing', async () => {
    const onError = vi.fn()
    vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(new DOMException('denied'))
    render(<Harness onError={onError} />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Bắt đầu ghi' })[0])

    await waitFor(() =>
      expect(onError).toHaveBeenCalledWith(
        0,
        'Không thể truy cập microphone. Hãy kiểm tra quyền trình duyệt.',
      ),
    )
  })
})
