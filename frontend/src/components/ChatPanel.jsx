import { useState } from 'react'
import { useAgentChat } from '../hooks/useAgentChat'
import AgentTrace from './AgentTrace'

const SUGGESTIONS = [
  '¿Qué variables influyen más en el objetivo? Entrena y compara modelos.',
  'Describe el dataset y detecta problemas de calidad de datos.',
  'Compara tu regresión con descenso de gradiente contra scikit-learn.',
]

export default function ChatPanel({ dataset }) {
  const { turns, running, ask, stop, reset } = useAgentChat(dataset?.id)
  const [question, setQuestion] = useState('')

  const submit = (e) => {
    e.preventDefault()
    ask(question)
    setQuestion('')
  }

  return (
    <section className="panel chat">
      <div className="chat-header">
        <h2>2. Pregunta al equipo de agentes</h2>
        {turns.length > 0 && !running && <button className="link" onClick={reset}>Limpiar</button>}
      </div>

      {!dataset && <p className="muted">Primero carga un dataset.</p>}

      {dataset && turns.length === 0 && (
        <div className="samples">
          {SUGGESTIONS.map((s) => (
            <button key={s} className="chip" onClick={() => setQuestion(s)}>{s}</button>
          ))}
        </div>
      )}

      {turns.map((t, i) => (
        <article key={i} className="turn">
          <p className="question">🧑 {t.question}</p>
          <AgentTrace steps={t.steps} />
          {t.answer && <div className="answer">{t.answer}</div>}
          {t.error && <p className="error">{t.error}</p>}
          {running && i === turns.length - 1 && !t.answer && <p className="muted pulse">Los agentes están trabajando…</p>}
        </article>
      ))}

      <form className="ask" onSubmit={submit}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)}
               placeholder={dataset ? 'Ej. ¿Qué predice el precio?' : 'Carga un dataset primero'}
               disabled={!dataset || running} />
        {running
          ? <button type="button" onClick={stop}>Detener</button>
          : <button type="submit" disabled={!dataset || !question.trim()}>Enviar</button>}
      </form>
    </section>
  )
}
