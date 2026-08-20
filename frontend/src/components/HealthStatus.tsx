import { useEffect, useState } from 'react'

import { getHealth } from '../api/client'

type HealthState = 'loading' | 'ok' | 'error'

export function HealthStatus() {
  const [state, setState] = useState<HealthState>('loading')

  useEffect(() => {
    let active = true

    getHealth()
      .then((health) => {
        if (active) setState(health.status === 'ok' ? 'ok' : 'error')
      })
      .catch(() => {
        if (active) setState('error')
      })

    return () => {
      active = false
    }
  }, [])

  const message = {
    loading: 'Checking backend…',
    ok: 'Backend status: ok',
    error: 'Backend unavailable. Start it on port 8000.',
  }[state]

  return (
    <div
      className="rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm"
      role="status"
    >
      {message}
    </div>
  )
}

