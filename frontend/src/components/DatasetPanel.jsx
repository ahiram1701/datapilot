import { useEffect, useState } from 'react'
import { listSamples, loadSample, uploadDataset } from '../api'

export default function DatasetPanel({ dataset, onLoaded }) {
  const [samples, setSamples] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    listSamples().then(setSamples).catch(() => setSamples([]))
  }, [])

  const run = async (promise) => {
    setLoading(true)
    setError(null)
    try {
      onLoaded(await promise)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <h2>1. Dataset</h2>
      <label className="upload">
        <input type="file" accept=".csv" disabled={loading}
               onChange={(e) => e.target.files[0] && run(uploadDataset(e.target.files[0]))} />
        <span>Subir CSV</span>
      </label>
      {samples.length > 0 && (
        <div className="samples">
          <small>o usa un ejemplo:</small>
          {samples.map((s) => (
            <button key={s} className="chip" disabled={loading} onClick={() => run(loadSample(s))}>
              {s}
            </button>
          ))}
        </div>
      )}
      {loading && <p className="muted">Cargando…</p>}
      {error && <p className="error">{error}</p>}

      {dataset && (
        <div className="dataset-info">
          <p><strong>{dataset.name}</strong> · {dataset.rows} filas · {dataset.columns.length} columnas</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>{dataset.columns.map((c) => <th key={c.name} title={c.dtype}>{c.name}</th>)}</tr>
              </thead>
              <tbody>
                {dataset.preview.map((row, i) => (
                  <tr key={i}>{dataset.columns.map((c) => <td key={c.name}>{String(row[c.name] ?? '')}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}
