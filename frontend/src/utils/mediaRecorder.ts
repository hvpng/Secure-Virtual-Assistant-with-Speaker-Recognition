const MIME_CANDIDATES = ['audio/webm;codecs=opus', 'audio/webm'] as const

export function selectSupportedAudioMimeType(): string | undefined {
  if (
    typeof MediaRecorder === 'undefined' ||
    typeof MediaRecorder.isTypeSupported !== 'function'
  ) {
    return undefined
  }
  return MIME_CANDIDATES.find((mimeType) => MediaRecorder.isTypeSupported(mimeType))
}
