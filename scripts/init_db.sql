-- ================================================================
-- INIT DB — mlops-proyecto2
-- Ejecutar una sola vez al iniciar el cluster
-- ================================================================

-- ── SCHEMA RAW ──────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.diabetes_raw (
    id               SERIAL PRIMARY KEY,
    batch_id         VARCHAR(50)  NOT NULL,
    load_timestamp   TIMESTAMP    DEFAULT NOW(),
    source_file      VARCHAR(255),
    row_hash         VARCHAR(64)  UNIQUE,
    status           VARCHAR(20)  DEFAULT 'loaded',
    raw_data         JSONB        NOT NULL
);

-- ── SCHEMA CLEAN ────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS clean;

CREATE TABLE IF NOT EXISTS clean.diabetes_clean (
    id                        SERIAL PRIMARY KEY,
    batch_id                  VARCHAR(50),
    processed_timestamp       TIMESTAMP DEFAULT NOW(),
    age                       INTEGER,
    time_in_hospital          INTEGER,
    num_lab_procedures        INTEGER,
    num_procedures            INTEGER,
    num_medications           INTEGER,
    number_outpatient         INTEGER,
    number_emergency          INTEGER,
    number_inpatient          INTEGER,
    number_diagnoses          INTEGER,
    race_encoded              INTEGER,
    gender_encoded            INTEGER,
    admission_type_encoded    INTEGER,
    discharge_encoded         INTEGER,
    admission_source_encoded  INTEGER,
    a1c_result_encoded        INTEGER,
    metformin_encoded         INTEGER,
    insulin_encoded           INTEGER,
    readmitted                INTEGER NOT NULL
);

-- ── SCHEMA INFERENCE ────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS inference;

CREATE TABLE IF NOT EXISTS inference.inference_logs (
    id                  SERIAL PRIMARY KEY,
    request_id          VARCHAR(100) NOT NULL,
    inference_timestamp TIMESTAMP    DEFAULT NOW(),
    input_data          JSONB        NOT NULL,
    prediction          INTEGER      NOT NULL,
    probability         FLOAT,
    model_name          VARCHAR(100),
    model_version       VARCHAR(50),
    model_alias         VARCHAR(50),
    response_time_ms    FLOAT,
    status_code         INTEGER
);
