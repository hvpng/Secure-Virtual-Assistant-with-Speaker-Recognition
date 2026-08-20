import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  ApiError,
  type Employee,
  type EnrollmentScript,
  type FailedEnrollmentItem,
  enrollEmployee,
  getEmployees,
  getEnrollmentScripts,
  reenrollEmployee,
  removeVoiceProfile,
} from '../api/client'
import { AudioRecorder } from '../components/AudioRecorder'

const EXPECTED_RECORDING_COUNT = 7

type RecordingStatus = 'empty' | 'recording' | 'ready' | 'failed'

type RecordingItem = {
  index: number
  scriptText: string
  blob: Blob | null
  objectUrl: string | null
  mimeType: string | null
  status: RecordingStatus
}

type EnrollmentMode =
  | { kind: 'enroll' }
  | { kind: 'reenroll'; employee: Employee }

const CHECK_LABELS = {
  duration_ok: 'Thời lượng',
  speech_ratio_ok: 'Tỷ lệ giọng nói',
  snr_ok: 'SNR',
  clipping_ok: 'Không bị vỡ tiếng',
  content_match_ok: 'Nội dung đọc',
} as const

function messageFromError(error: unknown): string {
  return error instanceof ApiError
    ? error.message
    : 'Không thể hoàn thành yêu cầu. Vui lòng thử lại.'
}

function emptyRecordings(scripts: EnrollmentScript[]): Record<number, RecordingItem> {
  return Object.fromEntries(
    scripts.map((script) => [
      script.index,
      {
        index: script.index,
        scriptText: script.text,
        blob: null,
        objectUrl: null,
        mimeType: null,
        status: 'empty' as const,
      },
    ]),
  )
}

export function Enrollment() {
  const [scripts, setScripts] = useState<EnrollmentScript[]>([])
  const [employees, setEmployees] = useState<Employee[]>([])
  const [recordings, setRecordings] = useState<Record<number, RecordingItem>>({})
  const recordingsRef = useRef(recordings)
  const [mode, setMode] = useState<EnrollmentMode>({ kind: 'enroll' })
  const [employeeId, setEmployeeId] = useState('')
  const [name, setName] = useState('')
  const [activeRecordingIndex, setActiveRecordingIndex] = useState<number | null>(null)
  const [failedItems, setFailedItems] = useState<FailedEnrollmentItem[]>([])
  const [recordingErrors, setRecordingErrors] = useState<Record<number, string>>({})
  const [scriptsLoading, setScriptsLoading] = useState(true)
  const [employeesLoading, setEmployeesLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [removingEmployeeId, setRemovingEmployeeId] = useState<string | null>(null)
  const [pageError, setPageError] = useState<string | null>(null)
  const [resultMessage, setResultMessage] = useState<string | null>(null)

  recordingsRef.current = recordings

  const revokeRecordingUrls = useCallback(() => {
    Object.values(recordingsRef.current).forEach((recording) => {
      if (recording.objectUrl) URL.revokeObjectURL(recording.objectUrl)
    })
  }, [])

  const resetRecordingState = useCallback(
    (nextScripts: EnrollmentScript[] = scripts) => {
      revokeRecordingUrls()
      setRecordings(emptyRecordings(nextScripts))
      setActiveRecordingIndex(null)
      setFailedItems([])
      setRecordingErrors({})
    },
    [revokeRecordingUrls, scripts],
  )

  const refreshEmployees = useCallback(async () => {
    setEmployeesLoading(true)
    try {
      setEmployees(await getEmployees())
    } catch (error) {
      setPageError(messageFromError(error))
    } finally {
      setEmployeesLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    setScriptsLoading(true)
    getEnrollmentScripts()
      .then((loadedScripts) => {
        if (!active) return
        setScripts(loadedScripts)
        setRecordings(emptyRecordings(loadedScripts))
      })
      .catch((error: unknown) => {
        if (active) setPageError(messageFromError(error))
      })
      .finally(() => {
        if (active) setScriptsLoading(false)
      })
    void refreshEmployees()

    return () => {
      active = false
      revokeRecordingUrls()
    }
  }, [refreshEmployees, revokeRecordingUrls])

  const failedByIndex = useMemo(
    () => new Map(failedItems.map((item) => [item.index, item])),
    [failedItems],
  )
  const orderedRecordings = scripts.map((script) => recordings[script.index])
  const recordedCount = orderedRecordings.filter((recording) => recording?.blob).length
  const fieldsValid = employeeId.trim().length > 0 && name.trim().length > 0
  const recordingsComplete =
    scripts.length === EXPECTED_RECORDING_COUNT &&
    orderedRecordings.every((recording) => recording?.blob instanceof Blob)
  const canSubmit =
    fieldsValid && recordingsComplete && activeRecordingIndex === null && !submitting

  const validationMessage = useMemo(() => {
    if (scriptsLoading) return 'Đang tải danh sách câu đọc.'
    if (scripts.length !== EXPECTED_RECORDING_COUNT) {
      return 'Không tải được đầy đủ 7 câu đăng ký từ máy chủ.'
    }
    if (!fieldsValid) return 'Vui lòng nhập đầy đủ mã nhân viên và họ tên.'
    if (activeRecordingIndex !== null) return 'Hãy dừng bản ghi đang chạy trước khi gửi.'
    if (!recordingsComplete) return 'Vui lòng ghi âm đầy đủ 7 câu trước khi đăng ký.'
    return null
  }, [activeRecordingIndex, fieldsValid, recordingsComplete, scripts.length, scriptsLoading])

  const handleStarted = (index: number) => {
    setPageError(null)
    setResultMessage(null)
    setActiveRecordingIndex(index)
    setFailedItems((current) => current.filter((item) => item.index !== index))
    setRecordingErrors((current) => {
      const next = { ...current }
      delete next[index]
      return next
    })
    setRecordings((current) => ({
      ...current,
      [index]: { ...current[index], status: 'recording' },
    }))
  }

  const handleRecorded = (index: number, blob: Blob, mimeType: string) => {
    const objectUrl = URL.createObjectURL(blob)
    setRecordings((current) => {
      const previous = current[index]
      if (previous.objectUrl) URL.revokeObjectURL(previous.objectUrl)
      return {
        ...current,
        [index]: {
          ...previous,
          blob,
          objectUrl,
          mimeType,
          status: 'ready',
        },
      }
    })
    setActiveRecordingIndex(null)
  }

  const handleRecordingError = (index: number, message: string) => {
    setRecordingErrors((current) => ({ ...current, [index]: message }))
    setRecordings((current) => ({
      ...current,
      [index]: {
        ...current[index],
        status: current[index]?.blob ? 'ready' : 'failed',
      },
    }))
    setActiveRecordingIndex(null)
  }

  const clearForm = useCallback(() => {
    resetRecordingState()
    setEmployeeId('')
    setName('')
    setMode({ kind: 'enroll' })
  }, [resetRecordingState])

  const submitEnrollment = async () => {
    if (!canSubmit) return
    setSubmitting(true)
    setPageError(null)
    setResultMessage(null)
    const payload = orderedRecordings.map((recording) => ({
      index: recording.index,
      blob: recording.blob as Blob,
    }))

    try {
      const response =
        mode.kind === 'reenroll'
          ? await reenrollEmployee(mode.employee.id, payload)
          : await enrollEmployee({ id: employeeId, name }, payload)
      if (response.success) {
        clearForm()
        setResultMessage(
          mode.kind === 'reenroll'
            ? 'Đăng ký lại giọng nói thành công.'
            : 'Đăng ký giọng nói thành công.',
        )
        await refreshEmployees()
      } else {
        setFailedItems(response.failed_items)
        const failedIndices = new Set(response.failed_items.map((item) => item.index))
        setRecordings((current) =>
          Object.fromEntries(
            Object.entries(current).map(([key, recording]) => [
              key,
              {
                ...recording,
                status: failedIndices.has(recording.index) ? 'failed' : 'ready',
              },
            ]),
          ),
        )
        setPageError('Một số bản ghi chưa đạt yêu cầu. Hãy ghi lại các câu được đánh dấu.')
      }
    } catch (error) {
      setPageError(messageFromError(error))
    } finally {
      setSubmitting(false)
    }
  }

  const beginReenroll = (employee: Employee) => {
    resetRecordingState()
    setMode({ kind: 'reenroll', employee })
    setEmployeeId(employee.id)
    setName(employee.name)
    setPageError(null)
    setResultMessage(null)
  }

  const beginEnrollmentFor = (employee: Employee) => {
    resetRecordingState()
    setMode({ kind: 'enroll' })
    setEmployeeId(employee.id)
    setName(employee.name)
    setPageError(null)
    setResultMessage(null)
  }

  const cancelReenroll = () => {
    clearForm()
    setPageError(null)
    setResultMessage(null)
  }

  const removeProfile = async (employee: Employee) => {
    if (!window.confirm(`Xóa hồ sơ giọng nói của ${employee.name}?`)) return
    setRemovingEmployeeId(employee.id)
    setPageError(null)
    setResultMessage(null)
    try {
      await removeVoiceProfile(employee.id)
      await refreshEmployees()
      setResultMessage('Đã xóa hồ sơ giọng nói. Hồ sơ nhân viên vẫn được giữ lại.')
    } catch (error) {
      setPageError(messageFromError(error))
    } finally {
      setRemovingEmployeeId(null)
    }
  }

  return (
    <section className="space-y-10">
      <div>
        <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">Module M5</p>
        <h2 className="mt-1 text-2xl font-bold">Đăng ký giọng nói</h2>
        <p className="mt-2 text-sm text-slate-600">
          Đọc đủ bảy câu theo thứ tự. Chất lượng và nội dung được kiểm tra tại backend.
        </p>
      </div>

      {pageError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800" role="alert">
          {pageError}
        </div>
      )}
      {resultMessage && (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800" role="status">
          {resultMessage}
        </div>
      )}

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-lg font-semibold">
              {mode.kind === 'reenroll' ? 'Đăng ký lại hồ sơ' : 'Đăng ký hồ sơ mới'}
            </h3>
            <p className="text-sm text-slate-500">Đã ghi {recordedCount}/{EXPECTED_RECORDING_COUNT} câu</p>
          </div>
          {mode.kind === 'reenroll' && (
            <button type="button" onClick={cancelReenroll} disabled={submitting || activeRecordingIndex !== null} className="rounded-md border border-slate-300 px-3 py-2 text-sm disabled:opacity-50">
              Hủy đăng ký lại
            </button>
          )}
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <label className="text-sm font-medium text-slate-700">
            Mã nhân viên
            <input value={employeeId} onChange={(event) => setEmployeeId(event.target.value)} readOnly={mode.kind === 'reenroll'} className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 read-only:bg-slate-100" placeholder="NV001" />
          </label>
          <label className="text-sm font-medium text-slate-700">
            Họ và tên
            <input value={name} onChange={(event) => setName(event.target.value)} readOnly={mode.kind === 'reenroll'} className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 read-only:bg-slate-100" placeholder="Nguyễn Văn A" />
          </label>
        </div>

        <div className="mt-6 space-y-4">
          {scriptsLoading && <p role="status">Đang tải câu đăng ký…</p>}
          {!scriptsLoading && scripts.length === 0 && <p className="text-sm text-red-700">Không có câu đăng ký để hiển thị.</p>}
          {scripts.map((script) => {
            const recording = recordings[script.index]
            const failure = failedByIndex.get(script.index)
            const recordingError = recordingErrors[script.index]
            const isRecording = activeRecordingIndex === script.index
            const status = isRecording
              ? 'Đang ghi…'
              : failure || recording?.status === 'failed'
                ? 'Cần ghi lại'
                : recording?.blob
                  ? 'Đã ghi'
                  : 'Chưa ghi'

            return (
              <article key={script.index} data-script-index={script.index} className={`rounded-lg border p-4 ${failure ? 'border-red-400 bg-red-50' : 'border-slate-200'}`}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-blue-700">Câu {script.index + 1}/{scripts.length}</p>
                    <p className="mt-1 text-slate-800">{script.text}</p>
                    <p className={`mt-2 text-xs font-medium ${isRecording ? 'text-red-700' : 'text-slate-500'}`}>
                      {status}{recording?.mimeType ? ` · ${recording.mimeType}` : ''}
                    </p>
                  </div>
                  <AudioRecorder index={script.index} isRecording={isRecording} disabled={activeRecordingIndex !== null || submitting} hasRecording={Boolean(recording?.blob)} onStarted={handleStarted} onRecorded={handleRecorded} onError={handleRecordingError} />
                </div>

                {recording?.objectUrl && !isRecording && <audio className="mt-3 w-full" controls src={recording.objectUrl}>Trình duyệt không hỗ trợ phát audio.</audio>}
                {recordingError && <p className="mt-3 text-sm text-red-700">{recordingError}</p>}
                {failure && (
                  <div className="mt-3 space-y-2 text-sm text-red-800">
                    <ul className="list-disc pl-5">{failure.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(failure.checks).map(([key, passed]) => (
                        <span key={key} className={`rounded-full px-2 py-1 text-xs ${passed ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800'}`}>
                          {passed ? '✓' : '✗'} {CHECK_LABELS[key as keyof typeof CHECK_LABELS] ?? key}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </article>
            )
          })}
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-4">
          <button type="button" onClick={submitEnrollment} disabled={!canSubmit} className="rounded-md bg-emerald-700 px-4 py-2 font-semibold text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-300">
            {submitting ? 'Đang gửi…' : mode.kind === 'reenroll' ? 'Gửi đăng ký lại' : 'Đăng ký giọng nói'}
          </button>
          {validationMessage && !submitting && <p className="text-sm text-slate-600">{validationMessage}</p>}
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-lg font-semibold">Hồ sơ giọng nói nhân viên</h3>
            <p className="text-sm text-slate-500">Xóa voice profile không xóa hồ sơ nhân viên.</p>
          </div>
          <button type="button" onClick={() => void refreshEmployees()} disabled={employeesLoading} className="rounded-md border border-slate-300 px-3 py-2 text-sm disabled:opacity-50">Làm mới</button>
        </div>

        {employeesLoading ? (
          <p className="mt-4" role="status">Đang tải nhân viên…</p>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full border-collapse text-left text-sm">
              <thead><tr className="border-b border-slate-200 text-slate-500"><th className="px-2 py-3">Mã</th><th className="px-2 py-3">Họ tên</th><th className="px-2 py-3">Trạng thái giọng nói</th><th className="px-2 py-3">Thao tác</th></tr></thead>
              <tbody>
                {employees.map((employee) => (
                  <tr key={employee.id} className="border-b border-slate-100">
                    <td className="px-2 py-3 font-mono">{employee.id}</td><td className="px-2 py-3">{employee.name}</td>
                    <td className="px-2 py-3"><span className={`rounded-full px-2 py-1 text-xs font-medium ${employee.voice_enrolled ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-700'}`}>{employee.voice_enrolled ? 'Đã đăng ký' : 'Chưa đăng ký'}</span></td>
                    <td className="px-2 py-3"><div className="flex flex-wrap gap-2">
                      {employee.voice_enrolled ? (
                        <>
                          <button type="button" onClick={() => beginReenroll(employee)} disabled={submitting || activeRecordingIndex !== null} className="rounded border border-blue-300 px-2 py-1 text-blue-800 disabled:opacity-50">Đăng ký lại</button>
                          <button type="button" onClick={() => void removeProfile(employee)} disabled={removingEmployeeId === employee.id} className="rounded border border-red-300 px-2 py-1 text-red-700 disabled:opacity-50">{removingEmployeeId === employee.id ? 'Đang xóa…' : 'Xóa hồ sơ giọng nói'}</button>
                        </>
                      ) : (
                        <button type="button" onClick={() => beginEnrollmentFor(employee)} disabled={submitting || activeRecordingIndex !== null} className="rounded border border-blue-300 px-2 py-1 text-blue-800 disabled:opacity-50">Điền biểu mẫu</button>
                      )}
                    </div></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {employees.length === 0 && <p className="py-4 text-sm text-slate-500">Chưa có nhân viên.</p>}
          </div>
        )}
      </div>
    </section>
  )
}
