import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://fastapi-service:8000/predict")

st.set_page_config(page_title="Diabetes - Inferencia API", layout="centered")

st.title("Predicción de Readmisión Hospitalaria")
st.write("Ingrese los datos del paciente para predecir si será readmitido en el hospital.")

VALID_AGES = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

def validate_inputs(data: dict) -> list:
    errors = []
    if data["age"] not in VALID_AGES:
        errors.append(f"Edad debe ser el punto medio de un rango de 10 años: {VALID_AGES}")
    if not 1 <= data["time_in_hospital"] <= 14:
        errors.append("Días en hospital debe estar entre 1 y 14")
    if not 0 <= data["num_lab_procedures"] <= 132:
        errors.append("Pruebas de laboratorio debe estar entre 0 y 132")
    if not 0 <= data["num_procedures"] <= 6:
        errors.append("Procedimientos debe estar entre 0 y 6")
    if not 1 <= data["num_medications"] <= 81:
        errors.append("Medicamentos debe estar entre 1 y 81")
    if data["number_outpatient"] < 0:
        errors.append("Consultas externas no puede ser negativo")
    if data["number_emergency"] < 0:
        errors.append("Emergencias no puede ser negativo")
    if data["number_inpatient"] < 0:
        errors.append("Hospitalizaciones previas no puede ser negativo")
    if not 1 <= data["number_diagnoses"] <= 16:
        errors.append("Diagnósticos debe estar entre 1 y 16")
    return errors

with st.form("predict_form"):
    st.subheader("Datos Clínicos y Demográficos")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox(
            "Edad (rango)", options=VALID_AGES,
            format_func=lambda x: f"{x-4}-{x+5} años",
            index=4,
            help="Seleccione el rango de edad del paciente"
        )
        time_in_hospital = st.number_input(
            "Días en hospital", min_value=1, max_value=14, value=5,
            help="Entre 1 y 14 días"
        )
        num_lab_procedures = st.number_input(
            "Pruebas de lab", min_value=0, max_value=132, value=40
        )
        num_procedures = st.number_input(
            "Procedimientos", min_value=0, max_value=6, value=1
        )
        num_medications = st.number_input(
            "Medicamentos", min_value=1, max_value=81, value=10
        )
        number_outpatient = st.number_input(
            "Consultas ext.", min_value=0, value=0
        )

    with col2:
        number_emergency = st.number_input(
            "Emergencias", min_value=0, value=0
        )
        number_inpatient = st.number_input(
            "Hosp. previas", min_value=0, value=1
        )
        number_diagnoses = st.number_input(
            "Diagnósticos", min_value=1, max_value=16, value=7
        )
        race_encoded = st.selectbox(
            "Raza",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Caucasian", 2: "AfricanAmerican",
                3: "Hispanic", 4: "Asian", 5: "Other"
            }[x]
        )
        gender_encoded = st.selectbox(
            "Género",
            options=[0, 1],
            format_func=lambda x: {0: "Male", 1: "Female"}[x]
        )
        admission_type_encoded = st.selectbox(
            "Tipo Admisión",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Other", 1: "Emergency", 2: "Urgent", 3: "Elective"
            }[x],
            index=1
        )

    with col3:
        discharge_encoded = st.selectbox(
            "Tipo Alta",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: "Other", 1: "Home", 2: "Short-term Hosp.",
                3: "Skilled Nursing", 4: "Expired"
            }[x],
            index=1
        )
        admission_source_encoded = st.selectbox(
            "Origen Admisión",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Other", 1: "Physician Referral", 2: "Emergency Room"
            }[x],
            index=2
        )
        a1c_result_encoded = st.selectbox(
            "Resultado A1C",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "No medido", 1: "Normal", 2: ">7", 3: ">8"
            }[x]
        )
        metformin_encoded = st.selectbox(
            "Metformina",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "No", 1: "Steady", 2: "Up", 3: "Down"}[x],
            index=1
        )
        insulin_encoded = st.selectbox(
            "Insulina",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "No", 1: "Steady", 2: "Up", 3: "Down"}[x],
            index=2
        )

    submitted = st.form_submit_button("Realizar Predicción", type="primary")

if submitted:
    payload = {
        "age": age,
        "time_in_hospital": int(time_in_hospital),
        "num_lab_procedures": int(num_lab_procedures),
        "num_procedures": int(num_procedures),
        "num_medications": int(num_medications),
        "number_outpatient": int(number_outpatient),
        "number_emergency": int(number_emergency),
        "number_inpatient": int(number_inpatient),
        "number_diagnoses": int(number_diagnoses),
        "race_encoded": race_encoded,
        "gender_encoded": gender_encoded,
        "admission_type_encoded": admission_type_encoded,
        "discharge_encoded": discharge_encoded,
        "admission_source_encoded": admission_source_encoded,
        "a1c_result_encoded": a1c_result_encoded,
        "metformin_encoded": metformin_encoded,
        "insulin_encoded": insulin_encoded,
    }

    errors = validate_inputs(payload)
    if errors:
        st.error("Errores de validación. Corrija los siguientes campos:")
        for e in errors:
            st.write(f"- {e}")
    else:
        try:
            with st.spinner("Consultando modelo..."):
                response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                st.success("Inferencia completada exitosamente.")

                if data["prediction"] == 1:
                    st.error("Resultado: PACIENTE READMITIDO (1)")
                else:
                    st.info("Resultado: NO READMITIDO (0)")

                st.write(f"**Probabilidad estimada:** `{data['probability']:.2%}`")

                st.divider()
                st.caption(
                    f"Modelo: `{data['model_name']}` | "
                    f"Version: `{data['model_version']}` | "
                    f"Alias: `{data['model_alias']}`"
                )
                st.caption(
                    f"Latencia: `{data['response_time_ms']:.2f} ms` | "
                    f"Request ID: `{data['request_id']}`"
                )

            elif response.status_code == 422:
                st.error("Error de validación en la API: datos fuera de rango.")
                detail = response.json().get("detail", [])
                for d in detail:
                    st.write(f"- `{' -> '.join(str(x) for x in d.get('loc', []))}`: {d.get('msg', '')}")
            else:
                st.error(
                    f"Error en la API (Status {response.status_code}): {response.text}"
                )
        except requests.exceptions.RequestException as e:
            st.error(
                f"No se pudo conectar con la API en `{API_URL}`. "
                f"Revisa que el pod de FastAPI este corriendo. Detalles: {e}"
            )
