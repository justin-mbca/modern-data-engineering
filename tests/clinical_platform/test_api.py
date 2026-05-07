import pytest
from fastapi.testclient import TestClient
from clinical_platform.api.main import app

client = TestClient(app)

PATIENT_EXAMPLE = {
    "resourceType": "Patient",
    "id": "123",
    "name": [{"text": "John Doe"}],
    "gender": "male",
    "birthDate": "1980-01-01",
    "address": [{"text": "123 Main St"}]
}

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_fhir_ingest_patient():
    resp = client.post("/api/v1/fhir/ingest", json={"resource": PATIENT_EXAMPLE})
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "Patient"
    assert data["normalized"]["patient_id"] == "123"
    assert data["normalized"]["name"] == "John Doe"

def test_get_patient():
    # Ingest first
    client.post("/api/v1/fhir/ingest", json={"resource": PATIENT_EXAMPLE})
    resp = client.get("/api/v1/patients/123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_id"] == "123"
    assert data["name"] == "John Doe"

def test_get_patient_not_found():
    resp = client.get("/api/v1/patients/doesnotexist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Patient not found"
