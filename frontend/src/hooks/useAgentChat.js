import { useCallback, useRef, useState } from 'react'
import { streamChat } from '../api'

/** Encapsula el estado de una conversación con el equipo de agentes. */
export function useAgentChat(datasetId) {
  const [turns, setTurns] = useState([]) // [{question, steps, answer, error}]
  const [running, setRunning] = useState(false)
  const abortRef = useRef(null)

  const updateLast = (fn) =>
    setTurns((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])])

  const ask = useCallback(
    async (question) => {
      if (!datasetId || !question.trim()) return
      setTurns((prev) => [...prev, { question, steps: [], answer: null, error: null }])
      setRunning(true)
      abortRef.current = new AbortController()
      try {
        await streamChat(
          datasetId,
          question,
          (event, data) => {
            if (event === 'step') updateLast((t) => ({ ...t, steps: [...t.steps, data] }))
            if (event === 'answer') updateLast((t) => ({ ...t, answer: data.answer }))
            if (event === 'error') updateLast((t) => ({ ...t, error: data.error }))
          },
          abortRef.current.signal,
        )
      } catch (err) {
        if (err.name !== 'AbortError') updateLast((t) => ({ ...t, error: err.message }))
      } finally {
        setRunning(false)
      }
    },
    [datasetId],
  )

  const stop = () => abortRef.current?.abort()
  const reset = () => setTurns([])

  return { turns, running, ask, stop, reset }
}
