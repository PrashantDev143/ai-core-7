import { useState } from 'react'
import { sendCorrection, sendFeedback } from '../api.js'
import Citations from './Citations.jsx'
import Feedback from './Feedback.jsx'

/**
 * Renders [1]-style markers as buttons that scroll to and highlight the source
 * chunk. A citation the reader cannot follow in one click is decoration.
 */
function withCitations(text, onJump) {
  const parts = text.split(/(\[\d+(?:,\s*\d+)*\])/g)
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+(?:,\s*\d+)*)\]$/)
    if (!match) return <span key={i}>{part}</span>
    return match[1].split(/,\s*/).map((n) => (
      <button
        key={`${i}-${n}`}
        className="cite"
        onClick={() => onJump(Number(n))}
        title={`Jump to source ${n}`}
      >
        {n}
      </button>
    ))
  })
}

export default function Answer({ result, onRegenerate }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(result.answer)
  const [saved, setSaved] = useState(false)
  const [highlight, setHighlight] = useState(null)

  const jump = (n) => {
    setHighlight(n)
    document.getElementById(`cite-${n}`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    })
  }

  async function copy() {
    await navigator.clipboard.writeText(result.answer)
    // A copy is the strongest implicit positive available: the user thought
    // the answer was worth taking away.
    sendFeedback(result.request_id, 'copy').catch(() => {})
  }

  async function saveCorrection() {
    await sendCorrection({
      request_id: result.request_id,
      query: result.query,
      original_answer: result.answer,
      corrected_answer: draft,
      corpus_version: result.corpus_version,
    })
    setSaved(true)
    setEditing(false)
  }

  return (
    <section className="answer">
      <div className="answer-meta">
        <span className={`confidence ${result.confidence}`}>
          {result.confidence} confidence
        </span>
        {result.cache_layer && (
          <span className="tag cache">cache: {result.cache_layer}</span>
        )}
        {result.blocked && <span className="tag blocked">blocked</span>}
        {result.faithfulness && (
          <span
            className={`tag faith ${result.faithfulness.verdict}`}
            title={`${result.faithfulness.claims_unsupported} of ${result.faithfulness.claims_total} claims unmatched`}
          >
            grounding: {result.faithfulness.verdict}
          </span>
        )}
        {result.timings_ms?.generation_ms != null && (
          <span className="tag">
            {Math.round(
              Object.values(result.timings_ms).reduce((a, b) => a + b, 0)
            )}
            ms
          </span>
        )}
      </div>

      {/* Caveats sit ABOVE the answer. A warning under a confident-looking
          paragraph is read after the damage is done. */}
      {result.caveats?.length > 0 && (
        <div className="caveats">
          {result.caveats.map((c, i) => (
            <div key={i} className="caveat">{c}</div>
          ))}
        </div>
      )}

      {editing ? (
        <div className="editor">
          <textarea
            value={draft}
            rows={10}
            onChange={(e) => setDraft(e.target.value)}
          />
          <div className="row">
            <button onClick={saveCorrection}>Save correction</button>
            <button className="ghost" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
          <p className="hint">
            Your correction is stored for human review before it can be used for
            anything. It is not applied automatically.
          </p>
        </div>
      ) : (
        <div className="answer-body">{withCitations(result.answer, jump)}</div>
      )}

      {saved && (
        <div className="saved">Correction saved — queued for review.</div>
      )}

      <div className="actions">
        <Feedback requestId={result.request_id} />
        <button className="ghost" onClick={copy}>Copy</button>
        <button className="ghost" onClick={() => setEditing(true)}>Edit</button>
        <button className="ghost" onClick={onRegenerate}>Regenerate</button>
      </div>

      <Citations citations={result.citations} highlight={highlight} />
    </section>
  )
}
