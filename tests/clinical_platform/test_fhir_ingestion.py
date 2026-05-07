import pytest
from clinical_platform.fhir_ingestion import ingest_fhir_resource

PATIENT_EXAMPLE = {
    "resourceType": "Patient",
    "id": "123",
    "name": [{"text": "John Doe"}],
    "gender": "male",
    "birthDate": "1980-01-01",
    "address": [{"text": "123 Main St"}]
}

def test_ingest_patient():
    result = ingest_fhir_resource(PATIENT_EXAMPLE)
    assert result["resourceType"] == "Patient"
    assert result["normalized"]["patient_id"] == "123"
    assert result["normalized"]["name"] == "John Doe"
    assert result["normalized"]["gender"] == "male"
    assert result["normalized"]["birth_date"] == "1980-01-01"
    assert result["normalized"]["address"] == "123 Main St"

def test_ingest_invalid_resource():
    bad = {"resourceType": "Unknown", "id": "1"}
    result = ingest_fhir_resource(bad)
    assert "error" in result
    assert "Unsupported resourceType" in result["error"]
