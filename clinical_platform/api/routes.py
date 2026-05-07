"""
API Routes for Clinical Platform
Defines FastAPI endpoints for FHIR ingestion and patient retrieval.
"""
from fastapi import APIRouter, HTTPException
from .schemas import FHIRIngestRequest, FHIRIngestResponse, HealthCheckResponse, PatientResponse
from .service import ingest_fhir, get_patient

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse()

@router.post("/api/v1/fhir/ingest", response_model=FHIRIngestResponse)
def fhir_ingest(request: FHIRIngestRequest):
    result = ingest_fhir(request.resource)
    if "error" in result:
        return FHIRIngestResponse(resourceType=result.get("resourceType", "Unknown"), normalized={}, error=result["error"], details=result.get("details"))
    return FHIRIngestResponse(resourceType=result["resourceType"], normalized=result["normalized"])

@router.get("/api/v1/patients/{patient_id}", response_model=PatientResponse)
def get_patient_route(patient_id: str):
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse(**patient)
