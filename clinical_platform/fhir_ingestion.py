"""
FHIR Ingestion utilities for Clinical Platform
Handles validation, normalization, and error reporting for FHIR resource payloads.
"""
from typing import Dict, Any
from pydantic import ValidationError
from .fhir_models import FHIRPatient, FHIREncounter, FHIRObservation, FHIRCondition
from .normalization import normalize_patient, normalize_encounter, normalize_observation, normalize_condition

FHIR_MODEL_MAP = {
    "Patient": FHIRPatient,
    "Encounter": FHIREncounter,
    "Observation": FHIRObservation,
    "Condition": FHIRCondition,
}

NORMALIZE_MAP = {
    "Patient": normalize_patient,
    "Encounter": normalize_encounter,
    "Observation": normalize_observation,
    "Condition": normalize_condition,
}

def ingest_fhir_resource(resource_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a FHIR resource JSON payload.
    Returns normalized dict or error details.
    """
    resource_type = resource_json.get("resourceType")
    if resource_type not in FHIR_MODEL_MAP:
        return {"error": f"Unsupported resourceType: {resource_type}"}
    model_cls = FHIR_MODEL_MAP[resource_type]
    try:
        resource_obj = model_cls.model_validate(resource_json)
    except ValidationError as e:
        return {"error": "Validation failed", "details": e.errors()}
    normalize_fn = NORMALIZE_MAP[resource_type]
    normalized = normalize_fn(resource_obj)
    return {"resourceType": resource_type, "normalized": normalized}
