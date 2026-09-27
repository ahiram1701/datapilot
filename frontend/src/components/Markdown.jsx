import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Los LLMs responden en markdown (tablas, negritas, listas). react-markdown
// no interpreta HTML crudo, así que es seguro con texto generado por el modelo.
// Los modelos a veces usan <br> dentro de celdas de tabla; como no renderizamos
// HTML, lo convertimos en un separador visual.
const cleanup = (text) => text.replace(/<br\s*\/?>/gi, ' · ')

export default function Markdown({ children }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{cleanup(children)}</ReactMarkdown>
    </div>
  )
}
