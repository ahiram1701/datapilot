-- Base de demo para probar DataPilot con PostgreSQL (docker compose la carga al iniciar).
-- Reutiliza los CSV de sample_data y crea un usuario de SOLO LECTURA para los agentes.

CREATE TABLE clientes (
    id SERIAL PRIMARY KEY,
    meses_cliente INTEGER,
    cargo_mensual NUMERIC(10, 2),
    contrato TEXT,
    tickets_soporte INTEGER,
    abandona SMALLINT
);
COPY clientes (meses_cliente, cargo_mensual, contrato, tickets_soporte, abandona)
    FROM '/sample_data/clientes_churn.csv' WITH (FORMAT csv, HEADER true);

CREATE TABLE viviendas (
    id SERIAL PRIMARY KEY,
    area_m2 NUMERIC,
    recamaras NUMERIC,
    antiguedad NUMERIC,
    distancia_centro_km NUMERIC,
    zona TEXT,
    precio NUMERIC
);
COPY viviendas (area_m2, recamaras, antiguedad, distancia_centro_km, zona, precio)
    FROM '/sample_data/viviendas.csv' WITH (FORMAT csv, HEADER true);

CREATE ROLE datapilot_ro LOGIN PASSWORD 'datapilot_ro';
GRANT CONNECT ON DATABASE demo TO datapilot_ro;
GRANT USAGE ON SCHEMA public TO datapilot_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO datapilot_ro;
ANALYZE;  -- llena pg_class.reltuples para el estimado de filas
