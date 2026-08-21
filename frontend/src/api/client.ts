const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

const FALLBACK_ERROR_MESSAGE = 'Không thể hoàn thành yêu cầu. Vui lòng thử lại.'

export type HealthResponse = {
  status: string
}

export type Employee = {
  id: string
  name: string
  voice_enrolled: boolean
}

export type EnrollmentScript = {
  index: number
  text: string
}

export type QualityChecks = {
  duration_ok: boolean
  speech_ratio_ok: boolean
  snr_ok: boolean
  clipping_ok: boolean
  content_match_ok: boolean
}

export type FailedEnrollmentItem = {
  index: number
  checks: QualityChecks
  reasons: string[]
}

export type EnrollmentResponse = {
  success: boolean
  failed_items: FailedEnrollmentItem[]
}

export type AuthType = 'SV' | 'SID' | null

export type ChatResponse = {
  success: boolean
  text_asr: string
  function_called: string | null
  auth_type: AuthType
  auth_passed: boolean | null
  employee_id: string | null
  speaker_score: number | null
  response_text: string
  audio_reply_url: string | null
}

export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseResponseBody(response: Response): Promise<unknown> {
  const body = await response.text()
  if (!body) return null

  try {
    return JSON.parse(body) as unknown
  } catch {
    return null
  }
}

function errorMessage(body: unknown): string {
  if (
    typeof body === 'object' &&
    body !== null &&
    'detail' in body &&
    typeof body.detail === 'string'
  ) {
    return body.detail
  }
  return FALLBACK_ERROR_MESSAGE
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new ApiError('Không thể kết nối đến máy chủ.')
  }

  const body = await parseResponseBody(response)
  if (!response.ok) {
    throw new ApiError(errorMessage(body), response.status)
  }
  if (body === null) {
    throw new ApiError(FALLBACK_ERROR_MESSAGE, response.status)
  }
  return body as T
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health')
}

export async function getEmployees(): Promise<Employee[]> {
  const response = await requestJson<{ employees: Employee[] }>('/api/employees')
  if (
    !Array.isArray(response.employees) ||
    !response.employees.every(
      (employee) =>
        typeof employee === 'object' &&
        employee !== null &&
        typeof employee.id === 'string' &&
        typeof employee.name === 'string' &&
        typeof employee.voice_enrolled === 'boolean',
    )
  ) {
    throw new ApiError(FALLBACK_ERROR_MESSAGE)
  }
  return response.employees
}

export async function getEnrollmentScripts(): Promise<EnrollmentScript[]> {
  const response = await requestJson<{ scripts: EnrollmentScript[] }>(
    '/api/enrollment-scripts',
  )
  if (
    !Array.isArray(response.scripts) ||
    !response.scripts.every(
      (script) =>
        typeof script === 'object' &&
        script !== null &&
        Number.isInteger(script.index) &&
        script.index >= 0 &&
        typeof script.text === 'string' &&
        script.text.trim().length > 0,
    )
  ) {
    throw new ApiError(FALLBACK_ERROR_MESSAGE)
  }
  return [...response.scripts].sort((left, right) => left.index - right.index)
}

export function buildEnrollmentFormData(
  recordings: Array<{ index: number; blob: Blob }>,
  employee?: { id: string; name: string },
): FormData {
  const formData = new FormData()
  if (employee) {
    formData.append('employee_id', employee.id.trim())
    formData.append('name', employee.name.trim())
  }
  for (const recording of [...recordings].sort((left, right) => left.index - right.index)) {
    formData.append('audio_files', recording.blob, `enroll_${recording.index}.webm`)
  }
  return formData
}

function isEnrollmentResponse(body: unknown): body is EnrollmentResponse {
  const checkKeys: Array<keyof QualityChecks> = [
    'duration_ok',
    'speech_ratio_ok',
    'snr_ok',
    'clipping_ok',
    'content_match_ok',
  ]
  return (
    typeof body === 'object' &&
    body !== null &&
    'success' in body &&
    typeof body.success === 'boolean' &&
    'failed_items' in body &&
    Array.isArray(body.failed_items) &&
    body.failed_items.every(
      (item) =>
        typeof item === 'object' &&
        item !== null &&
        'index' in item &&
        typeof item.index === 'number' &&
        Number.isInteger(item.index) &&
        'checks' in item &&
        typeof item.checks === 'object' &&
        item.checks !== null &&
        checkKeys.every(
          (key) =>
            key in item.checks &&
            typeof (item.checks as Record<string, unknown>)[key] === 'boolean',
        ) &&
        'reasons' in item &&
        Array.isArray(item.reasons) &&
        (item.reasons as unknown[]).every((reason: unknown) => typeof reason === 'string'),
    )
  )
}

async function submitMultipart(path: string, formData: FormData): Promise<EnrollmentResponse> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      body: formData,
    })
  } catch {
    throw new ApiError('Không thể kết nối đến máy chủ.')
  }

  const body = await parseResponseBody(response)
  if (isEnrollmentResponse(body)) return body
  if (!response.ok) throw new ApiError(errorMessage(body), response.status)
  throw new ApiError(FALLBACK_ERROR_MESSAGE, response.status)
}

export function enrollEmployee(
  employee: { id: string; name: string },
  recordings: Array<{ index: number; blob: Blob }>,
): Promise<EnrollmentResponse> {
  return submitMultipart('/api/enroll', buildEnrollmentFormData(recordings, employee))
}

export function reenrollEmployee(
  employeeId: string,
  recordings: Array<{ index: number; blob: Blob }>,
): Promise<EnrollmentResponse> {
  return submitMultipart(
    `/api/employees/${encodeURIComponent(employeeId)}/reenroll`,
    buildEnrollmentFormData(recordings),
  )
}

export async function removeVoiceProfile(employeeId: string): Promise<void> {
  await requestJson(`/api/employees/${encodeURIComponent(employeeId)}/voice-profile`, {
    method: 'DELETE',
  })
}

export function buildVoiceCommandFormData(
  audio: Blob,
  claimedEmployeeId?: string,
): FormData {
  const formData = new FormData()
  formData.append('audio', audio, 'voice-command.webm')
  const normalizedClaim = claimedEmployeeId?.trim()
  if (normalizedClaim) formData.append('claimed_employee_id', normalizedClaim)
  return formData
}

function isChatResponse(body: unknown): body is ChatResponse {
  if (typeof body !== 'object' || body === null) return false
  const response = body as Record<string, unknown>
  const nullableString = (value: unknown) => value === null || typeof value === 'string'
  const nullableBoolean = (value: unknown) => value === null || typeof value === 'boolean'
  const nullableFiniteNumber = (value: unknown) =>
    value === null || (typeof value === 'number' && Number.isFinite(value))

  return (
    typeof response.success === 'boolean' &&
    typeof response.text_asr === 'string' &&
    nullableString(response.function_called) &&
    (response.auth_type === null || response.auth_type === 'SV' || response.auth_type === 'SID') &&
    nullableBoolean(response.auth_passed) &&
    nullableString(response.employee_id) &&
    nullableFiniteNumber(response.speaker_score) &&
    typeof response.response_text === 'string' &&
    nullableString(response.audio_reply_url)
  )
}

export async function sendVoiceCommand(
  audio: Blob,
  claimedEmployeeId?: string,
): Promise<ChatResponse> {
  if (audio.size <= 0) throw new ApiError('Bản ghi âm không được rỗng.')

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      body: buildVoiceCommandFormData(audio, claimedEmployeeId),
    })
  } catch {
    throw new ApiError('Không thể kết nối đến máy chủ.')
  }

  const body = await parseResponseBody(response)
  if (!response.ok) throw new ApiError(errorMessage(body), response.status)
  if (!isChatResponse(body)) {
    throw new ApiError('Không thể xử lý yêu cầu. Vui lòng thử lại.', response.status)
  }
  return body
}

export function resolveApiUrl(path: string): string {
  try {
    const base = new URL(`${API_BASE_URL.replace(/\/+$/, '')}/`)
    const resolved = new URL(path, base)
    if (resolved.origin !== base.origin) throw new Error('Cross-origin audio URL')
    return resolved.toString()
  } catch {
    throw new ApiError('Đường dẫn phản hồi âm thanh không hợp lệ.')
  }
}
