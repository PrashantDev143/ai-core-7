import { sendFeedback } from '../api.js'

export default function Citations({ citations, highlight }) {
  if (!citations?.length) return null

  return (
    <div className="citations">
      <h3>Sources</h3>
      {citations.map((c) => (
        <div
          key={c.index}
          id={`cite-${c.index}`}
          className={`citation ${highlight === c.index ? 'hl' : ''} ${
            c.cited ? '' : 'uncited'
          }`}
        >
          <div className="cit-head">
            <span className="num">{c.index}</span>
            <span className="title">{c.title || c.arxiv_id}</span>
            {c.arxiv_id && (
              <a
                href={`https://arxiv.org/abs/${c.arxiv_id}`}
                target="_blank"
                rel="noreferrer"
                // A citation click is a meaningful implicit signal: the reader
                // cared enough to check the source.
                onClick={() => sendFeedback(c.request_id, 'citation_click').catch(() => {})}
              >
                arXiv
              </a>
            )}
            <span className="score">{c.score}</span>
          </div>
          <div className="cit-meta">
            {c.section && <span>{c.section}</span>}
            <span>p. {c.page_start}</span>
            {/* Retrieved-but-uncited passages are shown too. Seeing what the
                model had available and chose not to use is half of debugging
                a bad answer. */}
            {!c.cited && <span className="muted">retrieved, not cited</span>}
          </div>
          <p className="excerpt">{c.excerpt}…</p>
        </div>
      ))}
    </div>
  )
}
