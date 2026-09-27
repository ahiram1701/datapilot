import { useEffect, useState } from 'react'
import { getHealth } from './api'
import ChatPanel from './components/ChatPanel'
import DatasetPanel from './components/DatasetPanel'

export default function App() {
  const [dataset, setDataset] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'offline' }))
  }, [])

  return (
    <div className="app">
      <header>
        <h1>DataPilot</h1>
        <p>Equipo multi-agente de análisis de datos y machine learning</p>
        {health && (
          <span className={`status ${health.status === 'ok' ? 'ok' : 'off'}`}>
            API {health.status}{health.llm_provider && ` · LLM: ${health.llm_provider}`}
          </span>
        )}
      </header>
      <main>
        <DatasetPanel dataset={dataset} onLoaded={setDataset} />
        <ChatPanel key={dataset?.id} dataset={dataset} />
      </main>
    </div>
  )
}
