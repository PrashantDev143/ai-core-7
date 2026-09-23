import { useEffect, useState } from 'react'
import { getCacheMetrics, getStats } from '../api.js'

export default function StatsBar() {
  const [corpus, setCorpus] = useState(null)
  const [cache, setCache] = useState(null)

  useEffect(() => {
    getStats().then(setCorpus).catch(() => {})
    const tick = () => getCacheMetrics().then(setCache).catch(() => {})
    tick()
    const timer = setInterval(tick, 15000)
    return () => clearInterval(timer)
  }, [])

  if (!corpus) return null

  return (
    <div className="statsbar">
      <span>{corpus.documents} papers</span>
      <span>{corpus.chunks} chunks</span>
      <span>{corpus.embedding_model?.split('/').pop()}</span>
      {/* The corpus version is surfaced because it is what invalidates the
          cache — seeing it change explains why hit rate just dropped. */}
      {corpus.corpus_version && (
        <span title="corpus version — cache keys include this">
          v{corpus.corpus_version.slice(0, 8)}
        </span>
      )}
      {cache && cache.lookups > 0 && (
        <span title={`exact ${cache.exact_hits} / semantic ${cache.semantic_hits}`}>
          cache {(cache.hit_rate * 100).toFixed(0)}%
        </span>
      )}
    </div>
  )
}
