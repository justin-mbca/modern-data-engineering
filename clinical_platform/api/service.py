"""
Service layer for FHIR ingestion and patient retrieval.
"""
from ..fhir_ingestion import ingest_fhir_resource
from ..normalization import normalize_patient
from ..fhir_models import FHIRPatient
from typing import Dict, Any, Optional

# In-memory store for demo purposes
PATIENT_STORE = {}

def ingest_fhir(resource_json: Dict[str, Any]) -> Dict[str, Any]:
    result = ingest_fhir_resource(resource_json)
    # Store patient if resourceType is Patient and valid
    if result.get("resourceType") == "Patient" and "normalized" in result:
        patient_id = result["normalized"].get("patient_id")
        if patient_id:
            PATIENT_STORE[patient_id] = result["normalized"]
    return result

def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    return PATIENT_STORE.get(patient_id)
