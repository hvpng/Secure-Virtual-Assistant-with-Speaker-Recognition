import { useEffect, useMemo, useRef, useState } from 'react'

import {
  ApiError,
  type ChatResponse,
  type Employee,
  getEmployees,
  resolveApiUrl,
  sendVoiceCommand,
} from '../api/client'
import { AudioRecorder } from '../components/AudioRecorder'

const FUNCTION_LABELS: Record<string, string> = {
  answer_faq: 'Hỏi đáp chung',
  reset_password: 'Đặt lại mật khẩu',
  check_salary_insurance: 'Kiểm tra lương và bảo hiểm',
  check_leave_days: 'Kiểm tra ngày phép',
  check_today_meetings: 'Kiểm tra lịch họp hôm nay',
}

function messageFromError(error: unknown): string {
  return error instanceof ApiError
    ? error.message
    : 'Không thể xử lý yêu cầu. Vui lòng thử lại.'
}

function authStatus(response: ChatResponse): string {
  if (response.auth_type === null) return 'Không yêu cầu xác thực'
  const outcome = response.auth_passed ? 'Thành công' : 'Thất bại'
  return response.auth_type === 'SV'
    ? `Xác minh người nói: ${outcome}`
    : `Nhận diện người nói: ${outcome}`
}

export function Chat() {
  const [employees, setEmployees] = useState<Employee[]>([])
  const [employeesLoading, setEmployeesLoading] = useState(true)
  const [employeeError, setEmployeeError] = useState<string | null>(null)
  const [claimedEmployeeId, setClaimedEmployeeId] = useState('')
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null)
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null)
  const recordingUrlRef = useRef<string | null>(null)
  const [recordingMimeType, setRecordingMimeType] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [result, setResult] = useState<ChatResponse | null>(null)

  useEffect(() => {
    let active = true
    setEmployeesLoading(true)
    getEmployees()
      .then((loadedEmployees) => {
        if (active) setEmployees(loadedEmployees)
      })
      .catch((error: unknown) => {
        if (active) setEmployeeError(messageFromError(error))
      })
      .finally(() => {
        if (active) setEmployeesLoading(false)
      })

    return () => {
      active = false
      if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current)
    }
  }, [])

  const handleStarted = () => {
    setIsRecording(true)
    setRequestError(null)
  }

  const handleRecorded = (_index: number, blob: Blob, mimeType: string) => {
    const nextUrl = URL.createObjectURL(blob)
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current)
    recordingUrlRef.current = nextUrl
    setRecordingBlob(blob)
    setRecordingUrl(nextUrl)
    setRecordingMimeType(mimeType)
    setIsRecording(false)
    setRequestError(null)
  }

  const handleRecordingError = (_index: number, message: string) => {
    setIsRecording(false)
    setRequestError(message)
  }

  const submitCommand = async () => {
    if (!recordingBlob || recordingBlob.size <= 0 || isRecording || processing) return
    setProcessing(true)
    setRequestError(null)
    try {
      setResult(await sendVoiceCommand(recordingBlob, claimedEmployeeId || undefined))
    } catch (error) {
      setRequestError(messageFromError(error))
    } finally {
      setProcessing(false)
    }
  }

  const clearCommand = () => {
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current)
    recordingUrlRef.current = null
    setRecordingBlob(null)
    setRecordingUrl(null)
    setRecordingMimeType(null)
    setResult(null)
    setRequestError(null)
  }

  const resultEmployee = useMemo(() => {
    if (!result?.employee_id) return null
    const employee = employees.find((candidate) => candidate.id === result.employee_id)
    return employee ? `${employee.id} - ${employee.name}` : result.employee_id
  }, [employees, result])

  const resolvedReplyUrl = useMemo(() => {
    if (!result?.audio_reply_url) return null
    try {
      return resolveApiUrl(result.audio_reply_url)
    } catch {
      return null
    }
  }, [result])

  const canSend = Boolean(recordingBlob?.size) && !isRecording && !processing

  return (
    <section className="space-y-8">
      <div>
        <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">Module M6</p>
        <h2 className="mt-1 text-2xl font-bold">Trợ lý giọng nói</h2>
        <p className="mt-2 text-sm text-slate-600">
          Ghi một yêu cầu tiếng Việt, nghe lại rồi gửi để backend xử lý và xác thực khi cần.
        </p>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <label className="text-sm font-medium text-slate-700" htmlFor="claimed-employee">
          Xác nhận danh tính khi cần
        </label>
        <select
          id="claimed-employee"
          value={claimedEmployeeId}
          onChange={(event) => setClaimedEmployeeId(event.target.value)}
          disabled={employeesLoading || processing || isRecording}
          className="mt-2 w-full rounded-md border border-slate-300 bg-white px-3 py-2 disabled:bg-slate-100"
        >
          <option value="">Không chọn nhân viên</option>
          {employees.map((employee) => (
            <option key={employee.id} value={employee.id}>
              {employee.id} - {employee.name}
            </option>
          ))}
        </select>
        <p className="mt-2 text-xs text-slate-500">
          Đây chỉ là thông tin khai báo cho yêu cầu cần SV. Hệ thống vẫn xác thực hoặc nhận diện bằng giọng nói tại backend.
        </p>
        {employeesLoading && <p className="mt-2 text-sm text-slate-500" role="status">Đang tải nhân viên…</p>}
        {employeeError && <p className="mt-2 text-sm text-amber-700" role="alert">{employeeError}</p>}
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold">Yêu cầu bằng giọng nói</h3>
            <p className={`mt-1 text-sm ${isRecording ? 'font-medium text-red-700' : 'text-slate-500'}`} role="status">
              {processing
                ? 'Đang xử lý giọng nói…'
                : isRecording
                  ? 'Đang nghe…'
                  : recordingBlob
                    ? `Đã ghi · ${recordingMimeType ?? 'audio'}`
                    : 'Chưa có bản ghi'}
            </p>
          </div>
          <AudioRecorder
            index={0}
            isRecording={isRecording}
            disabled={processing}
            hasRecording={Boolean(recordingBlob)}
            onStarted={handleStarted}
            onRecorded={handleRecorded}
            onError={handleRecordingError}
            startLabel="Bắt đầu nói"
            stopLabel="Dừng ghi"
            rerecordLabel="Ghi lại"
          />
        </div>

        {recordingUrl && !isRecording && (
          <audio className="mt-4 w-full" controls src={recordingUrl}>
            Trình duyệt không hỗ trợ phát audio.
          </audio>
        )}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void submitCommand()}
            disabled={!canSend}
            className="rounded-md bg-emerald-700 px-4 py-2 font-semibold text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {processing ? 'Đang gửi…' : 'Gửi yêu cầu'}
          </button>
          {(recordingBlob || result) && (
            <button
              type="button"
              onClick={clearCommand}
              disabled={processing || isRecording}
              className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium disabled:opacity-50"
            >
              Ghi câu mới
            </button>
          )}
        </div>
      </div>

      {requestError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800" role="alert">
          {requestError}
        </div>
      )}

      {result && (
        <article
          data-testid="chat-result"
          className={`rounded-xl border p-5 shadow-sm ${result.success ? 'border-emerald-200 bg-white' : 'border-amber-300 bg-amber-50'}`}
        >
          <div className="grid gap-5 md:grid-cols-2">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Bạn đã nói</p>
              <p className="mt-2 text-slate-900">{result.text_asr}</p>
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">Trợ lý</p>
              <p className="mt-2 font-medium text-slate-900">{result.response_text}</p>
              {resolvedReplyUrl ? (
                <audio className="mt-3 w-full" controls src={resolvedReplyUrl}>
                  Trình duyệt không hỗ trợ phát phản hồi âm thanh.
                </audio>
              ) : (
                <p className="mt-3 text-xs text-slate-500">Không tạo được phản hồi âm thanh.</p>
              )}
            </div>
          </div>

          <div className="mt-5 border-t border-slate-200 pt-4">
            <h3 className="text-sm font-semibold">Chi tiết xử lý</h3>
            <dl className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
              <div><dt className="text-slate-500">Function</dt><dd className="font-medium">{result.function_called ? `${FUNCTION_LABELS[result.function_called] ?? 'Không xác định'} (${result.function_called})` : 'Không có'}</dd></div>
              <div><dt className="text-slate-500">Xác thực</dt><dd className={`font-medium ${result.auth_passed === false ? 'text-red-700' : 'text-slate-900'}`}>{authStatus(result)}</dd></div>
              <div><dt className="text-slate-500">Nhân viên từ backend</dt><dd className="font-medium">{resultEmployee ?? 'Không có'}</dd></div>
              <div><dt className="text-slate-500">Speaker score</dt><dd className="font-mono">{result.speaker_score === null ? 'Không có' : result.speaker_score.toFixed(3)}</dd></div>
            </dl>
          </div>
        </article>
      )}
    </section>
  )
}
