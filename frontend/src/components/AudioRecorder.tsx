import { useEffect, useRef } from 'react'

import { selectSupportedAudioMimeType } from '../utils/mediaRecorder'

type AudioRecorderProps = {
  index: number
  isRecording: boolean
  disabled: boolean
  hasRecording: boolean
  onStarted: (index: number) => void
  onRecorded: (index: number, blob: Blob, mimeType: string) => void
  onError: (index: number, message: string) => void
  startLabel?: string
  stopLabel?: string
  rerecordLabel?: string
}

export function AudioRecorder({
  index,
  isRecording,
  disabled,
  hasRecording,
  onStarted,
  onRecorded,
  onError,
  startLabel = 'Bắt đầu ghi',
  stopLabel = 'Dừng ghi',
  rerecordLabel = 'Ghi lại',
}: AudioRecorderProps) {
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const mountedRef = useRef(true)
  const startingRef = useRef(false)
  const stopRequestedRef = useRef(false)

  const startRecording = async () => {
    if (
      typeof MediaRecorder === 'undefined' ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      onError(index, 'Trình duyệt không hỗ trợ ghi âm bằng MediaRecorder.')
      return
    }
    if (startingRef.current || recorderRef.current !== null) return

    startingRef.current = true
    onStarted(index)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        startingRef.current = false
        return
      }
      streamRef.current = stream
      const mimeType = selectSupportedAudioMimeType()
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)
      const chunks: BlobPart[] = []
      let settled = false

      recorderRef.current = recorder
      chunksRef.current = chunks
      stopRequestedRef.current = false

      const cleanup = () => {
        stream.getTracks().forEach((track) => track.stop())
        if (streamRef.current === stream) streamRef.current = null
        if (recorderRef.current === recorder) recorderRef.current = null
        if (chunksRef.current === chunks) chunksRef.current = []
        startingRef.current = false
        stopRequestedRef.current = false
      }

      recorder.addEventListener('dataavailable', (event) => {
        if (event.data.size > 0) chunks.push(event.data)
      })
      recorder.addEventListener('stop', () => {
        if (settled) return
        settled = true
        const actualMimeType = recorder.mimeType || mimeType || 'audio/webm'
        const blob = new Blob(chunks, { type: actualMimeType })
        try {
          if (!mountedRef.current) return
          if (blob.size === 0) {
            onError(index, 'Không thu được dữ liệu âm thanh. Vui lòng ghi lại.')
          } else {
            onRecorded(index, blob, actualMimeType)
          }
        } finally {
          cleanup()
        }
      })
      recorder.addEventListener('error', () => {
        if (settled) return
        settled = true
        try {
          if (mountedRef.current) {
            onError(index, 'Không thể ghi âm. Vui lòng thử lại.')
          }
        } finally {
          cleanup()
        }
      })
      recorder.start()
      startingRef.current = false
    } catch {
      const stream = streamRef.current
      stream?.getTracks().forEach((track) => track.stop())
      recorderRef.current = null
      streamRef.current = null
      chunksRef.current = []
      startingRef.current = false
      stopRequestedRef.current = false
      onError(index, 'Không thể truy cập microphone. Hãy kiểm tra quyền trình duyệt.')
    }
  }

  const stopRecording = () => {
    const recorder = recorderRef.current
    if (!recorder || stopRequestedRef.current) return
    if (recorder.state === 'recording' || recorder.state === 'paused') {
      stopRequestedRef.current = true
      recorder.stop()
    }
  }

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      const recorder = recorderRef.current
      if (
        recorder &&
        !stopRequestedRef.current &&
        (recorder.state === 'recording' || recorder.state === 'paused')
      ) {
        stopRequestedRef.current = true
        recorder.stop()
        return
      }
      if (recorder && stopRequestedRef.current) return

      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      recorderRef.current = null
      chunksRef.current = []
      startingRef.current = false
      stopRequestedRef.current = false
    }
  }, [])

  if (isRecording) {
    return (
      <button
        type="button"
        onClick={stopRecording}
        className="rounded-md bg-red-600 px-3 py-2 text-sm font-medium text-white hover:bg-red-700"
      >
        {stopLabel}
      </button>
    )
  }

  return (
    <button
      type="button"
      onClick={startRecording}
      disabled={disabled}
      className="rounded-md bg-blue-700 px-3 py-2 text-sm font-medium text-white hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-slate-300"
    >
      {hasRecording ? rerecordLabel : startLabel}
    </button>
  )
}
