"""
API Schemas for Clinical Platform
Defines Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class FHIRIngestRequest(BaseModel):
    resource: Dict[str, Any] = Field(..., description="FHIR resource JSON payload")

class FHIRIngestResponse(BaseModel):
    resourceType: str
    normalized: Dict[str, Any]
    error: Optional[str] = None
    details: Optional[Any] = None

class HealthCheckResponse(BaseModel):
    status: str = "ok"

class PatientResponse(BaseModel):
    patient_id: str
    name: Optional[str]
    gender: Optional[str]
    birth_date: Optional[str]
    address: Optional[str]
