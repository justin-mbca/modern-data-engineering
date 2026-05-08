import pytest
from clinical_platform.fhir_models import FHIRPatient, FHIREncounter, FHIRObservation, FHIRCondition
from datetime import date

def test_patient_model():
    patient = FHIRPatient(
        id="123",
        name=[{"text": "John Doe"}],
        gender="male",
        birthDate=date(1980, 1, 1),
        address=[{"text": "123 Main St"}]
    )
    assert patient.id == "123"
    assert patient.name[0]["text"] == "John Doe"
    assert patient.gender == "male"
    assert patient.birthDate == date(1980, 1, 1)
    assert patient.address[0]["text"] == "123 Main St"

def test_encounter_model():
    encounter = FHIREncounter(
        id="enc1",
        status="finished",
        **{"class": {"code": "AMB"}},
        subject={"reference": "Patient/123"},
        period={"start": "2022-01-01T00:00:00Z", "end": "2022-01-01T01:00:00Z"}
    )
    assert encounter.id == "enc1"
    assert encounter.status == "finished"
    # class_ is an alias for 'class', but Pydantic only populates it if the input dict uses the alias
    # Check both the attribute and the dict
    assert encounter.class_ is not None
    assert encounter.class_.get("code") == "AMB"
    assert encounter.subject["reference"] == "Patient/123"
    assert encounter.period["start"] == "2022-01-01T00:00:00Z"

def test_observation_model():
    obs = FHIRObservation(
        id="obs1",
        status="final",
        code={"coding": [{"code": "29463-7"}]},
        subject={"reference": "Patient/123"},
        valueQuantity={"value": 70, "unit": "kg"}
    )
    assert obs.id == "obs1"
    assert obs.status == "final"
    assert obs.code["coding"][0]["code"] == "29463-7"
    assert obs.valueQuantity["value"] == 70

def test_condition_model():
    cond = FHIRCondition(
        id="cond1",
        subject={"reference": "Patient/123"},
        code={"coding": [{"code": "E11"}]}
    )
    assert cond.id == "cond1"
    assert cond.subject["reference"] == "Patient/123"
    assert cond.code["coding"][0]["code"] == "E11"
