import pytest
from clinical_platform.fhir_models import FHIRPatient, FHIREncounter, FHIRObservation, FHIRCondition
from clinical_platform.normalization import normalize_patient, normalize_encounter, normalize_observation, normalize_condition
from datetime import date

def test_normalize_patient():
    patient = FHIRPatient(
        id="123",
        name=[{"text": "John Doe"}],
        gender="male",
        birthDate=date(1980, 1, 1),
        address=[{"text": "123 Main St"}]
    )
    norm = normalize_patient(patient)
    assert norm["patient_id"] == "123"
    assert norm["name"] == "John Doe"
    assert norm["gender"] == "male"
    assert norm["birth_date"] == "1980-01-01"
    assert norm["address"] == "123 Main St"

def test_normalize_encounter():
    encounter = FHIREncounter(
        id="enc1",
        status="finished",
        class_={"code": "AMB"},
        subject={"reference": "Patient/123"},
        period={"start": "2022-01-01T00:00:00Z", "end": "2022-01-01T01:00:00Z"}
    )
    norm = normalize_encounter(encounter)
    assert norm["encounter_id"] == "enc1"
    assert norm["patient_id"] == "Patient/123"
    assert norm["status"] == "finished"
    assert norm["class"] == "AMB"
    assert norm["start"] == "2022-01-01T00:00:00Z"
    assert norm["end"] == "2022-01-01T01:00:00Z"

def test_normalize_observation():
    obs = FHIRObservation(
        id="obs1",
        status="final",
        code={"coding": [{"code": "29463-7"}]},
        subject={"reference": "Patient/123"},
        valueQuantity={"value": 70, "unit": "kg"}
    )
    norm = normalize_observation(obs)
    assert norm["observation_id"] == "obs1"
    assert norm["patient_id"] == "Patient/123"
    assert norm["code"] == "29463-7"
    assert norm["value"] == 70

def test_normalize_condition():
    cond = FHIRCondition(
        id="cond1",
        subject={"reference": "Patient/123"},
        code={"coding": [{"code": "E11"}]}
    )
    norm = normalize_condition(cond)
    assert norm["condition_id"] == "cond1"
    assert norm["patient_id"] == "Patient/123"
    assert norm["code"] == "E11"
