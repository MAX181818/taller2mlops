from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    age: int = Field(..., ge=5, le=95)
    time_in_hospital: int = Field(..., ge=1, le=14)
    num_lab_procedures: int = Field(..., ge=0, le=132)
    num_procedures: int = Field(..., ge=0, le=6)
    num_medications: int = Field(..., ge=1, le=81)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    number_diagnoses: int = Field(..., ge=1, le=16)
    race_encoded: int = Field(..., ge=1, le=5)
    gender_encoded: int = Field(..., ge=0, le=1)
    admission_type_encoded: int = Field(..., ge=0, le=3)
    discharge_encoded: int = Field(..., ge=0, le=4)
    admission_source_encoded: int = Field(..., ge=0, le=2)
    a1c_result_encoded: int = Field(..., ge=0, le=3)
    metformin_encoded: int = Field(..., ge=0, le=3)
    insulin_encoded: int = Field(..., ge=0, le=3)

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    model_version: str
    model_alias: str
    response_time_ms: float
    request_id: str
