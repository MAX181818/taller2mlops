# =========================================================
# preprocessing.py
# Diabetes Readmission Dataset — MLOps Proyecto 2
# Persona 2 | feat/p2-airflow
#
# CONTRATO DE COLUMNAS (inmutable — acordado con Persona 3):
#   Features: age, time_in_hospital, num_lab_procedures,
#             num_procedures, num_medications, number_outpatient,
#             number_emergency, number_inpatient, number_diagnoses,
#             race_encoded, gender_encoded, admission_type_encoded,
#             discharge_encoded, admission_source_encoded,
#             a1c_result_encoded, metformin_encoded, insulin_encoded
#   Target:   readmitted  (0 = no readmitido, 1 = readmitido <30 días)
# =========================================================

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Mapeos fijos (deterministas, sin fit sobre datos) ─────────────────────────

# age: convierte rango "[X-Y)" al punto medio entero
AGE_MAP = {
    "[0-10)":  5,  "[10-20)": 15, "[20-30)": 25,
    "[30-40)": 35, "[40-50)": 45, "[50-60)": 55,
    "[60-70)": 65, "[70-80)": 75, "[80-90)": 85,
    "[90-100)": 95,
}

RACE_MAP = {
    "Caucasian":        1,
    "AfricanAmerican":  2,
    "Hispanic":         3,
    "Asian":            4,
    "Other":            5,
}

GENDER_MAP = {
    "Male":   0,
    "Female": 1,
}

# admission_type_id: 1=Emergency, 2=Urgent, 3=Elective, resto=Other
ADMISSION_TYPE_MAP = {1: 1, 2: 2, 3: 3}          # default → 0

# discharge_disposition_id: 1=Home, 2=Short-term hosp, 3=SNF, 11=Expired, resto=Other
DISCHARGE_MAP = {1: 1, 2: 2, 3: 3, 11: 4}        # default → 0

# admission_source_id: 1=Physician Referral, 7=Emergency Room, resto=Other
ADMISSION_SOURCE_MAP = {1: 1, 7: 2}               # default → 0

# a1c_result: Norm=1, >7=2, >8=3, None=0
A1C_MAP = {
    "Norm": 1,
    ">7":   2,
    ">8":   3,
    "None": 0,
}

# medicamentos: No=0, Steady=1, Up=2, Down=3
MED_MAP = {
    "No":     0,
    "Steady": 1,
    "Up":     2,
    "Down":   3,
}

# target
READMITTED_MAP = {
    "<30": 1,
    ">30": 0,
    "NO":  0,
}

# Columnas a eliminar (alta cardinalidad / no aportan al modelo)
COLS_TO_DROP = [
    "encounter_id", "patient_nbr",
    "weight",            # >96% nulos
    "payer_code",        # no clínico
    "medical_specialty", # alta cardinalidad
    # diagnósticos: strings libres, alta cardinalidad
    "diag_1", "diag_2", "diag_3",
]

# Medicamentos que usamos como features (contrato Persona 3)
MED_COLS_CONTRACT = ["metformin", "insulin"]

# Columnas numéricas que conservamos tal cual
NUM_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# ── Funciones de limpieza ─────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: duplicados y reemplazo de '?' por NaN."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.replace("?", np.nan)
    logger.info(f"Duplicados eliminados: {before - len(df)}")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas irrelevantes o con demasiados nulos."""
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)
    logger.info(f"Columnas eliminadas: {to_drop}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa nulos con estrategias apropiadas por tipo.
    Se usan mediana y moda fijas — NO se fitea sobre datos futuros.
    """
    # Numéricas → mediana
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Categóricas → moda
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    logger.info("Imputación de nulos completada")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding determinista con mapeos fijos.
    Produce exactamente las columnas del contrato con Persona 3.
    """
    # age → punto medio numérico
    df["age"] = df["age"].map(AGE_MAP).fillna(45).astype(int)

    # race_encoded
    df["race_encoded"] = df["race"].map(RACE_MAP).fillna(5).astype(int)

    # gender_encoded
    df["gender_encoded"] = df["gender"].map(GENDER_MAP).fillna(0).astype(int)

    # admission_type_encoded
    df["admission_type_encoded"] = (
        df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna(0).astype(int)
    )

    # discharge_encoded
    df["discharge_encoded"] = (
        df["discharge_disposition_id"].map(DISCHARGE_MAP).fillna(0).astype(int)
    )

    # admission_source_encoded
    df["admission_source_encoded"] = (
        df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna(0).astype(int)
    )

    # a1c_result_encoded
    a1c_col = "A1Cresult" if "A1Cresult" in df.columns else "a1cresult"
    df["a1c_result_encoded"] = (
        df[a1c_col].map(A1C_MAP).fillna(0).astype(int)
        if a1c_col in df.columns
        else 0
    )

    # metformin_encoded, insulin_encoded
    for med in MED_COLS_CONTRACT:
        if med in df.columns:
            df[f"{med}_encoded"] = df[med].map(MED_MAP).fillna(0).astype(int)
        else:
            df[f"{med}_encoded"] = 0

    logger.info("Encoding completado con mapeos fijos")
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte readmitted a binario: <30 → 1, resto → 0."""
    df["readmitted"] = df["readmitted"].map(READMITTED_MAP).fillna(0).astype(int)
    return df


def select_contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona EXACTAMENTE las columnas del contrato.
    Falla de forma explícita si falta alguna columna requerida.
    """
    contract_cols = NUM_COLS + [
        "age",
        "race_encoded",
        "gender_encoded",
        "admission_type_encoded",
        "discharge_encoded",
        "admission_source_encoded",
        "a1c_result_encoded",
        "metformin_encoded",
        "insulin_encoded",
        "readmitted",
    ]

    missing = [c for c in contract_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas requeridas por contrato faltantes: {missing}"
        )

    return df[contract_cols]


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame (un batch crudo desde raw.diabetes_raw)
    y retorna el DataFrame limpio con exactamente las columnas del contrato.

    Uso desde el DAG (t4_preprocess):
        from training.preprocessing import run_preprocessing
        df_clean = run_preprocessing(df_raw)
    """
    df = initial_cleaning(df)
    df = drop_columns(df)
    df = handle_missing_values(df)
    df = encode_features(df)
    df = binarize_target(df)
    df = select_contract_columns(df)
    logger.info(f"Preprocessing completado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# ── Validación de calidad (usada por t3_validate_quality) ────────────────────

def validate_quality(df: pd.DataFrame, null_threshold: float = 0.5) -> dict:
    """
    Valida que el batch crudo no supere umbrales de calidad.
    Retorna dict con resultado y detalle de columnas problemáticas.

    Uso desde el DAG (t3_validate_quality):
        from training.preprocessing import validate_quality
        result = validate_quality(df_batch)
        if not result["passed"]:
            raise ValueError(result["detail"])
    """
    null_ratios = df.isnull().mean()
    bad_cols = null_ratios[null_ratios > null_threshold].to_dict()

    result = {
        "passed": len(bad_cols) == 0,
        "total_rows": len(df),
        "null_threshold": null_threshold,
        "columns_above_threshold": bad_cols,
        "detail": (
            f"Columnas con >{null_threshold*100:.0f}% nulos: {list(bad_cols.keys())}"
            if bad_cols
            else "OK"
        ),
    }
    return result


# ── Ejecución local (debugging) ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "diabetic_data.csv"
    df_raw = load_data(path)

    # Validación
    qc = validate_quality(df_raw)
    print(f"Quality check: {qc['detail']}")

    # Pipeline
    df_clean = run_preprocessing(df_raw)
    print(f"Clean shape: {df_clean.shape}")
    print(df_clean.dtypes)
    print(df_clean.head(3))