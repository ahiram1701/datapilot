import { useState } from 'react'

const LABELS = {
  thought: 'Thought',
  action: 'Action',
  observation: 'Observation',
  final: 'Respuesta',
  error: 'Error',
}

function Step({ step }) {
  const [open, setOpen] = useState(step.type !== 'observation')
  const long = step.content.length > 160
  return (
    <li className={`step step-${step.type} agent-${step.agent}`}>
      <div className="step-head" onClick={() => long && setOpen(!open)}>
        <span className="badge">{step.agent}</span>
        <span className="kind">{LABELS[step.type] ?? step.type}</span>
        {long && <span className="toggle">{open ? '▾' : '▸'}</span>}
      </div>
      <pre className="step-body">{open || !long ? step.content : step.content.slice(0, 160) + '…'}</pre>
    </li>
  )
}

/** Muestra el razonamiento de los agentes: el ciclo ReAct hecho visible. */
export default function AgentTrace({ steps }) {
  if (!steps.length) return null
  return (
    <details className="trace" open>
      <summary>Traza de agentes ({steps.length} pasos)</summary>
      <ol>{steps.map((s, i) => <Step key={i} step={s} />)}</ol>
    </details>
  )
}
