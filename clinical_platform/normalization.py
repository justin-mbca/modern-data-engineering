"""
Normalization utilities for FHIR resources.
Transforms FHIR JSON payloads into analytics-friendly pandas DataFrames or dicts.
"""
from typing import Dict, Any
import pandas as pd
from .fhir_models import FHIRPatient, FHIREncounter, FHIRObservation, FHIRCondition

def normalize_patient(patient: FHIRPatient) -> Dict[str, Any]:
    """Normalize FHIR Patient resource to demographics table shape."""
    return {
        "patient_id": patient.id,
        "name": patient.name[0]["text"] if patient.name and "text" in patient.name[0] else None,
        "gender": patient.gender,
        "birth_date": str(patient.birthDate) if patient.birthDate else None,
        "address": patient.address[0]["text"] if patient.address and "text" in patient.address[0] else None,
    }

def normalize_encounter(encounter: FHIREncounter) -> Dict[str, Any]:
    """Normalize FHIR Encounter resource to encounters table shape."""
    return {
        "encounter_id": encounter.id,
        "patient_id": encounter.subject.get("reference", None),
        "status": encounter.status,
        "class": encounter.class_.get("code") if encounter.class_ else None,
        "start": encounter.period["start"] if encounter.period and "start" in encounter.period else None,
        "end": encounter.period["end"] if encounter.period and "end" in encounter.period else None,
    }

def normalize_observation(obs: FHIRObservation) -> Dict[str, Any]:
    """Normalize FHIR Observation resource to vitals/labs table shape."""
    return {
        "observation_id": obs.id,
        "patient_id": obs.subject.get("reference", None),
        "code": obs.code.get("coding", [{}])[0].get("code") if obs.code.get("coding") else None,
        "value": obs.valueQuantity["value"] if obs.valueQuantity and "value" in obs.valueQuantity else (
            obs.valueString or (obs.valueCodeableConcept["text"] if obs.valueCodeableConcept and "text" in obs.valueCodeableConcept else None)
        ),
        "effective": str(obs.effectiveDateTime) if obs.effectiveDateTime else None,
    }

def normalize_condition(cond: FHIRCondition) -> Dict[str, Any]:
    """Normalize FHIR Condition resource to problems/diagnoses table shape."""
    return {
        "condition_id": cond.id,
        "patient_id": cond.subject.get("reference", None),
        "code": cond.code.get("coding", [{}])[0].get("code") if cond.code.get("coding") else None,
        "onset": str(cond.onsetDateTime) if cond.onsetDateTime else None,
        "clinical_status": cond.clinicalStatus["coding"][0]["code"] if cond.clinicalStatus and "coding" in cond.clinicalStatus else None,
        "verification_status": cond.verificationStatus["coding"][0]["code"] if cond.verificationStatus and "coding" in cond.verificationStatus else None,
    }
