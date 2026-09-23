/**
 * The agent's full trajectory: every reasoning step, tool call and result.
 *
 * Shown by default rather than hidden behind a toggle. An agent whose
 * intermediate work is invisible cannot be debugged or trusted, and this is
 * the same structure the Phase 6 benchmark scores.
 */
export default function Trajectory({ trajectory }) {
  const breach = trajectory.budget_breach
  const budget = trajectory.budget

  return (
    <section className="trajectory">
      <div className="answer-meta">
        <span className="tag">{trajectory.steps.length} steps</span>
        <span className="tag">{trajectory.tokens?.prompt + trajectory.tokens?.completion} tokens</span>
        <span className="tag">{Math.round(trajectory.duration_ms)}ms</span>
        {budget && (
          <span className="tag">
            budget {budget.steps_used}/{budget.max_steps} steps
          </span>
        )}
        {breach && <span className="tag blocked">budget breach: {breach}</span>}
      </div>

      {breach && (
        <div className="caveats">
          <div className="caveat">
            The agent hit its {breach === 'max_steps' ? 'step' : 'token'} ceiling
            before finishing. The answer below is partial by construction.
          </div>
        </div>
      )}

      <div className="answer-body">{trajectory.answer}</div>

      <h3>Trajectory</h3>
      <ol className="steps">
        {trajectory.steps.map((step) => (
          <li key={step.step}>
            <div className="thought">{step.thought}</div>
            {step.tool_calls.map((call, i) => (
              <div key={i} className={`toolcall ${call.outcome}`}>
                <code>
                  {call.name}({JSON.stringify(call.arguments)})
                </code>
                <span className="dur">{Math.round(call.duration_ms)}ms</span>
                {call.error && <div className="err">{call.error}</div>}
                {call.result_summary && (
                  <pre className="result">{call.result_summary.slice(0, 500)}</pre>
                )}
              </div>
            ))}
          </li>
        ))}
      </ol>
    </section>
  )
}
