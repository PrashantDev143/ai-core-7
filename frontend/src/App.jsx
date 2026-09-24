import { useEffect, useRef, useState } from 'react'
import { ask, reportAbandon, research, sendFeedback } from './api.js'
import Answer from './components/Answer.jsx'
import Trajectory from './components/Trajectory.jsx'
import StatsBar from './components/StatsBar.jsx'

const EXAMPLES = [
  'What effect does chunk size have on RAG performance?',
  'How does self-consistency improve chain-of-thought reasoning?',
  'What is DoRA and how does it differ from LoRA?',
]

export default function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState('ask')
  const [result, setResult] = useState(null)
  const [trajectory, setTrajectory] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const lastRequestId = useRef(null)

  // Session abandonment: the user left without rating. Weak evidence on its
  // own, but it is the only signal available for the large majority who never
  // touch a thumb.
  useEffect(() => {
    const onHide = () => {
      if (document.visibilityState === 'hidden') reportAbandon(lastRequestId.current)
    }
    document.addEventListener('visibilitychange', onHide)
    return () => document.removeEventListener('visibilitychange', onHide)
  }, [])

  async function run(text, isRegeneration = false) {
    const q = (text ?? query).trim()
    if (!q || loading) return

    setLoading(true)
    setError(null)
    setResult(null)
    setTrajectory(null)

    try {
      if (mode === 'research') {
        const t = await research(q)
        setTrajectory(t)
        lastRequestId.current = t.trace_id
      } else {
        // A regeneration bypasses the cache — otherwise the user gets the same
        // answer back and the signal means nothing.
        const r = await ask(q, { use_cache: !isRegeneration })
        setResult(r)
        lastRequestId.current = r.request_id
      }
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>AI-core 7</h1>
        <p className="sub">
          Q&amp;A over 99 arXiv papers on language models, retrieval and agents
        </p>
        <StatsBar />
      </header>

      <div className="modes">
        <button
          className={mode === 'ask' ? 'on' : ''}
          onClick={() => setMode('ask')}
        >
          Ask
        </button>
        <button
          className={mode === 'research' ? 'on' : ''}
          onClick={() => setMode('research')}
        >
          Research (agent)
        </button>
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          run()
        }}
      >
        <textarea
          value={query}
          rows={3}
          placeholder="Ask about the papers…"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run()
          }}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? 'Working…' : mode === 'research' ? 'Research' : 'Ask'}
        </button>
      </form>

      {!result && !trajectory && !loading && (
        <div className="examples">
          {EXAMPLES.map((e) => (
            <button key={e} onClick={() => { setQuery(e); run(e) }}>
              {e}
            </button>
          ))}
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {result && (
        <Answer
          result={result}
          onRegenerate={() => {
            // Recorded against the answer being rejected, not the new one.
            sendFeedback(result.request_id, 'regenerate').catch(() => {})
            run(result.query, true)
          }}
        />
      )}

      {trajectory && <Trajectory trajectory={trajectory} />}
    </div>
  )
}
