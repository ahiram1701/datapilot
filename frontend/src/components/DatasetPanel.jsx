import { useEffect, useState } from 'react'
import { connectDatabase, disconnectDatabase, listSamples, loadSample, uploadDataset } from '../api'

export default function DatasetPanel({ dataset, onLoaded }) {
  const [samples, setSamples] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dbUrl, setDbUrl] = useState('')

  useEffect(() => {
    listSamples().then(setSamples).catch(() => setSamples([]))
  }, [])

  const run = async (promise) => {
    setLoading(true)
    setError(null)
    try {
      const loaded = await promise
      // Al cambiar de fuente, se libera la conexión anterior (el backend cierra el pool)
      if (dataset?.kind === 'sql') disconnectDatabase(dataset.id).catch(() => {})
      onLoaded(loaded)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const connect = (e) => {
    e.preventDefault()
    if (!dbUrl.trim()) return
    run(connectDatabase(dbUrl.trim()).then((conn) => {
      setDbUrl('') // no dejamos la contraseña en el estado del formulario
      return conn
    }))
  }

  return (
    <section className="panel">
      <h2>1. Datos</h2>
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
      <form className="db-connect" onSubmit={connect}>
        <small>o conecta tu base PostgreSQL (solo lectura):</small>
        <div className="db-row">
          <input type="password" value={dbUrl} disabled={loading} autoComplete="off"
                 placeholder="postgresql://usuario:clave@host:5432/base"
                 onChange={(e) => setDbUrl(e.target.value)} />
          <button type="submit" disabled={loading || !dbUrl.trim()}>Conectar</button>
        </div>
      </form>
      {loading && <p className="muted">Cargando…</p>}
      {error && <p className="error">{error}</p>}

      {dataset?.kind === 'sql' && (
        <div className="dataset-info">
          <p><strong>{dataset.name}</strong> · {dataset.tables.length} tablas</p>
          <ul className="table-list">
            {dataset.tables.map((t) => (
              <li key={t.name} title={t.columns.join('\n')}>
                <code>{t.name}</code>
                <small className="muted">
                  {' '}· {t.columns.length} columnas{t.rows_estimate != null && ` · ~${t.rows_estimate} filas`}
                </small>
              </li>
            ))}
          </ul>
        </div>
      )}

      {dataset && dataset.kind !== 'sql' && (
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
