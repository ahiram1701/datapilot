const BASE = import.meta.env.VITE_API_URL || '/api'

async function json(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Error HTTP ${res.status}`)
  }
  return res.json()
}

export const getHealth = () => fetch(`${BASE}/health`).then(json)
export const listSamples = () => fetch(`${BASE}/samples`).then(json)
export const loadSample = (name) => fetch(`${BASE}/samples/${name}`, { method: 'POST' }).then(json)

export function uploadDataset(file) {
  const form = new FormData()
  form.append('file', file)
  return fetch(`${BASE}/datasets`, { method: 'POST', body: form }).then(json)
}

/**
 * POST /chat devuelve Server-Sent Events. EventSource solo soporta GET,
 * así que leemos el stream a mano y separamos eventos por línea en blanco.
 */
export async function streamChat(datasetId, question, onEvent, signal) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, question }),
    signal,
  })
  if (!res.ok) await json(res)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() // el último puede estar incompleto
    for (const block of blocks) {
      const lines = Object.fromEntries(
        block.split('\n').map((l) => [l.slice(0, l.indexOf(':')), l.slice(l.indexOf(':') + 2)]),
      )
      if (lines.event) onEvent(lines.event, JSON.parse(lines.data))
    }
  }
}
