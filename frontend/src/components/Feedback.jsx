import { useState } from 'react'
import { sendFeedback } from '../api.js'

/**
 * One tap, shown immediately with the answer. The optional reason box only
 * appears AFTER a thumbs-down — asking for a reason up front suppresses the
 * rating itself, and a rating with no reason is still worth having.
 */
export default function Feedback({ requestId }) {
  const [sent, setSent] = useState(null)
  const [reason, setReason] = useState('')
  const [reasonSent, setReasonSent] = useState(false)

  async function rate(kind) {
    if (sent) return
    setSent(kind)
    try {
      await sendFeedback(requestId, kind)
    } catch {
      setSent(null)
    }
  }

  async function submitReason() {
    if (!reason.trim()) return
    await sendFeedback(requestId, 'thumb_down', { comment: reason })
    setReasonSent(true)
  }

  return (
    <div className="feedback">
      <button
        className={`thumb ${sent === 'thumb_up' ? 'on' : ''}`}
        disabled={!!sent}
        onClick={() => rate('thumb_up')}
        aria-label="Helpful"
      >
        ▲
      </button>
      <button
        className={`thumb ${sent === 'thumb_down' ? 'on' : ''}`}
        disabled={!!sent}
        onClick={() => rate('thumb_down')}
        aria-label="Not helpful"
      >
        ▼
      </button>

      {sent === 'thumb_down' && !reasonSent && (
        <span className="reason">
          <input
            value={reason}
            placeholder="What was wrong? (optional)"
            onChange={(e) => setReason(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submitReason()}
          />
          <button className="ghost" onClick={submitReason}>Send</button>
        </span>
      )}
      {reasonSent && <span className="thanks">Thanks.</span>}
    </div>
  )
}
