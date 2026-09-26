// '/api' goes through the Vite dev proxy. A deployed frontend has no proxy, so
// VITE_API_BASE points it at the backend's public URL instead.
const BASE = (import.meta.env.VITE_API_BASE || '/api').replace(/\/$/, '')

// One id per browser session. Used to tie implicit signals (regenerate, copy,
// abandon) to the same visitor without any account or persistent identifier.
export const sessionId = (() => {
  const key = 'aicore7-session'
  let id = sessionStorage.getItem(key)
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem(key, id)
  }
  return id
})()

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json()
}

export const ask = (query, opts = {}) =>
  post('/ask', { query, session_id: sessionId, ...opts })

export const research = (query) =>
  post('/research', { query, session_id: sessionId })

export const sendFeedback = (requestId, kind, extra = {}) =>
  post('/feedback/event', {
    request_id: requestId,
    kind,
    session_id: sessionId,
    ...extra,
  })

export const sendCorrection = (payload) =>
  post('/feedback/correction', { session_id: sessionId, ...payload })

export const getStats = async () => (await fetch(`${BASE}/corpus/stats`)).json()
export const getCacheMetrics = async () => (await fetch(`${BASE}/cache/metrics`)).json()

// Fires on tab close. `sendBeacon` survives page teardown where fetch does not,
// which is the only way an abandonment signal reliably arrives.
export function reportAbandon(requestId) {
  if (!requestId) return
  const body = JSON.stringify({
    request_id: requestId,
    kind: 'abandon',
    session_id: sessionId,
  })
  navigator.sendBeacon(
    `${BASE}/feedback/event`,
    new Blob([body], { type: 'application/json' })
  )
}
