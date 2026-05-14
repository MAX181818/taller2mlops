from locust import HttpUser, task, between

SAMPLE_PAYLOAD = {
    "age": 3,
    "time_in_hospital": 5,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 10,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 7,
    "race_encoded": 1,
    "gender_encoded": 0,
    "admission_type_encoded": 1,
    "discharge_encoded": 1,
    "admission_source_encoded": 7,
    "a1c_result_encoded": 0,
    "metformin_encoded": 1,
    "insulin_encoded": 2
}

class DiabetesAPIUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(3)
    def predict(self):
        self.client.post("/predict", json=SAMPLE_PAYLOAD)

    @task(1)
    def health(self):
        self.client.get("/health")

    @task(1)
    def model_info(self):
        self.client.get("/model-info")
