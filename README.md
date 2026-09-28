# DataPilot 🧭

**Equipo multi-agente basado en LLMs que analiza datasets y entrena modelos de machine learning.**
Subes un CSV, haces una pregunta en lenguaje natural ("¿qué variables predicen el precio?") y un orquestador
coordina agentes especialistas que exploran los datos, entrenan y validan modelos, y explican los resultados.
Cada paso del razonamiento (Thought → Action → Observation) se transmite en vivo a la interfaz.

![CI](../../actions/workflows/ci.yml/badge.svg) [![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)

![Demo de DataPilot: el equipo de agentes analiza clientes_churn.csv y propone cómo aumentar la retención](docs/demo.gif)

<sub>Pregunta: *"¿Cómo se puede aumentar la retención de clientes?"* sobre `clientes_churn.csv`. El orquestador delega al DataAnalyst (estadísticas, correlaciones, segmentos) y al MLEngineer (regresión logística con coeficientes con signo), y sintetiza recomendaciones con cifras que salen de las herramientas. Modelo: `openai/gpt-oss-120b` en el plan gratuito de Groq. Las esperas del LLM están aceleradas.</sub>

## Arquitectura

```mermaid
flowchart LR
    UI[React + Vite<br/>useAgentChat hook] -- REST + SSE --> API[FastAPI]
    API --> O[Orchestrator<br/>ReAct]
    O -- tool: ask_data_analyst --> DA[DataAnalyst<br/>ReAct]
    O -- tool: ask_ml_engineer --> ML[MLEngineer<br/>ReAct]
    O -- tool: ask_explainer --> EX[Explainer<br/>ReAct]
    DA --> T1[describe_dataset<br/>correlations · segment_analysis]
    ML --> T2[train_model · cross_validate<br/>gradient_descent_regression]
    EX --> V[(ChromaDB<br/>base vectorial)]
    O & DA & ML & EX -.-> LLM{{LLMProvider<br/>Anthropic · OpenAI · Groq · Gemini · Ollama · Mock}}
```

| Capa | Qué hace | Archivo |
|---|---|---|
| **Abstracción LLM** | Interfaz única `chat(system, messages, tools)`; cada proveedor traduce a su API. Cambiar de modelo = cambiar `LLM_PROVIDER`. | [`backend/app/llm/`](backend/app/llm) |
| **Agente ReAct** | Loop propio (sin frameworks): razona, pide una tool, observa el resultado, repite. Límite de pasos, errores devueltos al LLM para que se autocorrija, truncado de observaciones. | [`agents/react.py`](backend/app/agents/react.py) |
| **Multi-agente** | Patrón *agents-as-tools*: para el orquestador, cada especialista es una herramienta más. | [`agents/team.py`](backend/app/agents/team.py) |
| **Tool use** | Registro de herramientas con JSON Schema. | [`agents/tools.py`](backend/app/agents/tools.py) |
| **ML** | Regresión/clasificación (detección automática), análisis por segmentos, regresión lineal/logística (coeficientes con signo), árbol de decisión, random forest, train/test split, k-fold CV, R², MAE, RMSE, accuracy, F1. | [`ml/tools_ml.py`](backend/app/ml/tools_ml.py) |
| **Matemáticas** | Regresión lineal con **descenso de gradiente implementado en NumPy**, comparada contra la solución cerrada de scikit-learn. | [`ml/from_scratch.py`](backend/app/ml/from_scratch.py) |
| **RAG** | Base de conocimiento de conceptos de ML indexada en **ChromaDB**; el Explainer la consulta por similitud semántica. | [`rag/`](backend/app/rag) |
| **API REST** | FastAPI: carga de CSV, datasets de ejemplo, chat con streaming **Server-Sent Events**. | [`main.py`](backend/app/main.py) |
| **Frontend** | React con hooks (`useState`, `useEffect`, `useCallback`, `useRef`, hook propio `useAgentChat`), consumo de API y lectura manual de un stream SSE vía `fetch`. | [`frontend/src/`](frontend/src) |
| **DevOps** | Docker multi-stage, docker compose, GitHub Actions (tests + build + imagen). | [`docker-compose.yml`](docker-compose.yml) |

## Inicio rápido

### Con Docker
```bash
cp .env.example .env          # elige LLM_PROVIDER y pon tu API key
docker compose up --build
```
Abre http://localhost:8080

### Local
```bash
# Backend
python -m venv .venv && .venv/Scripts/activate    # Linux/Mac: source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp .env.example .env
cd backend && uvicorn app.main:app --reload

# Frontend (otra terminal)
cd frontend && npm install && npm run dev
```
Abre http://localhost:5173

Sin API key funciona con `LLM_PROVIDER=mock`, un proveedor determinista que recorre el flujo completo.

### Probar gratis en 2 minutos
1. Crea una key gratuita en [Groq](https://console.groq.com/keys) (sin tarjeta) o en [Google AI Studio](https://aistudio.google.com/apikey).
2. En `.env` pon `LLM_PROVIDER=groq` y `GROQ_API_KEY=...` (o `gemini` y `GEMINI_API_KEY=...`).
3. Reinicia el backend, carga `viviendas.csv` y pregunta: *"¿Qué variables influyen más en el precio? Compara modelos."*

### Conectar tu base de datos PostgreSQL
Además de CSV, DataPilot puede analizar una base PostgreSQL **en vivo**: los agentes exploran el esquema (`list_tables`, `describe_table`), consultan con SQL (`run_sql`) y entrenan modelos sobre el resultado de un `SELECT`, sin copiar tablas completas a memoria.

1. En el panel **Datos**, pega una URL `postgresql://usuario:clave@host:5432/base` y pulsa **Conectar**.
2. Pregunta, p. ej.: *"¿Qué factores explican el abandono de clientes?"*

Seguridad, porque el SQL lo escribe un LLM:
- La app solo acepta una consulta `SELECT`/`WITH` por llamada y le aplica un `LIMIT` externo.
- La sesión se abre con `default_transaction_read_only=on` y `statement_timeout`, así que el servidor rechaza cualquier escritura aunque algo evada el filtro.
- Usa un **usuario de solo lectura**. Las credenciales viven solo en memoria del backend y nunca se devuelven al navegador.

Con `docker compose up` se levanta una base de demo (`scripts/demo_db.sql`) con los datasets de ejemplo:
`postgresql://datapilot_ro:datapilot_ro@postgres:5432/demo`. Si tu base corre en tu equipo y el backend en Docker, usa `host.docker.internal` como host.

### Proveedores soportados
| `LLM_PROVIDER` | Variables | Costo | Notas |
|---|---|---|---|
| `groq` | `GROQ_API_KEY`, `GROQ_MODEL` | **Gratis** | GPT-OSS 120B o Qwen3, muy rápido |
| `gemini` | `GEMINI_API_KEY`, `GEMINI_MODEL` | **Gratis** | Gemini Flash |
| `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | **Gratis** (local) | Corre en tu equipo |
| `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | De pago | Tool use nativo de Claude |
| `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` | De pago | Function calling |
| `mock` | — | — | Tests y CI sin keys |

### Límites de tokens en planes gratuitos
Los planes gratuitos limitan tokens por minuto (Groq: 8K TPM, y algunos modelos también ~1K tokens de salida por minuto). Un flujo multi-agente hace varias llamadas, así que DataPilot permite ajustarlo:

| Variable | Efecto | Recomendado gratis |
|---|---|---|
| `LLM_MAX_TOKENS` | Tope de tokens de salida por petición (el proveedor lo reserva contra tu límite) | `800` |
| `LLM_REASONING_EFFORT` | Cuánto "piensa" un modelo de razonamiento (los tokens de razonamiento también cuentan como salida) | `low` (gpt-oss) · `none` (qwen) |
| `LLM_MAX_OBSERVATION_CHARS` | Recorte del resultado de cada tool; se reenvía al LLM en cada paso siguiente | `2000` |

Los errores 429 por minuto se reintentan automáticamente respetando `retry-after`. Si una respuesta se corta, la UI lo indica con *[respuesta truncada por max_tokens]*.

Groq, Gemini y Ollama exponen APIs compatibles con OpenAI, así que comparten `OpenAIProvider`. Cada uno es solo un *preset* (base_url, modelo por defecto y variable de la key) en [`openai_p.py`](backend/app/llm/openai_p.py).

## Tests
```bash
cd backend && pytest -q
```
Cubren el loop ReAct (ejecución de tools, errores, límite de pasos), la delegación multi-agente, las herramientas de ML, la búsqueda semántica, la API con streaming y que el descenso de gradiente converja a los coeficientes reales.

## Fundamentos

### Descenso de gradiente (implementado a mano)
Para ŷ = Xw + b y la pérdida MSE = (1/n)·Σ(ŷ − y)²:

- ∂L/∂w = (2/n)·Xᵀ(ŷ − y) → un solo producto matriz-vector da todas las derivadas.
- ∂L/∂b = (2/n)·Σ(ŷ − y)
- Actualización: w ← w − α·∂L/∂w

En el dataset de viviendas, la versión con NumPy alcanza **R² = 0.911, igual que scikit-learn**. El MSE es convexo, así que el método iterativo converge al mismo mínimo que la solución cerrada.

### ReAct
El agente alterna **razonamiento** y **acción**: el LLM decide qué herramienta usar, el sistema la ejecuta y devuelve la observación, y el ciclo se repite hasta que el LLM responde sin pedir herramientas. Es la base de casi todos los agentes modernos.

### Deep learning y Transformers
Los agentes funcionan sobre LLMs, que son **Transformers**: cada token calcula su atención sobre los demás con softmax(QKᵀ/√d)·V. A diferencia de las **RNN**, que procesan la secuencia paso a paso, la atención procesa todo en paralelo y captura dependencias largas. Las **CNN** aplican el mismo principio de pesos compartidos, pero sobre ventanas locales de una imagen. Todos se entrenan con el mismo ciclo que implementa `from_scratch.py`: pérdida → gradiente (backpropagation) → paso de descenso.

Para datos tabulares como los de este proyecto, los ensambles de árboles suelen superar a las redes neuronales; por eso el MLEngineer compara random forest, árbol y modelos lineales.

## Estructura
```
backend/app/
  llm/        base.py · anthropic_p.py · openai_p.py (+Groq, Gemini, Ollama) · mock_p.py · factory.py
  agents/     react.py · tools.py · team.py
  ml/         tools_ml.py · from_scratch.py
  rag/        store.py · docs/*.md
  db/         postgres.py (conexión SQL en vivo, solo lectura)
  main.py
frontend/src/ App.jsx · api.js · hooks/useAgentChat.js · components/
sample_data/  viviendas.csv (regresión) · clientes_churn.csv (clasificación)
scripts/      make_samples.py · demo_db.sql
```

## Posibles mejoras
- Memoria de conversación entre preguntas y persistencia de datasets (Redis/Postgres).
- Agente que genere gráficas (matplotlib → imagen en la UI).
- Ejecución de tools en paralelo y caché de resultados.
- Evaluación automática de las respuestas de los agentes (LLM-as-judge).

## Licencia
MIT © 2026 A. Hiram Saucedo G. Consulta [LICENSE](LICENSE).
