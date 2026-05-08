"""
FHIR Resource Models for Clinical Platform
Defines Pydantic models for core FHIR resources: Patient, Encounter, Observation, Condition.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import date, datetime

class FHIRPatient(BaseModel):
    resourceType: Literal["Patient"] = "Patient"
    id: str
    name: List[Dict[str, Any]]
    gender: Optional[str] = None
    birthDate: Optional[date] = None
    address: Optional[List[Dict[str, Any]]] = None

class FHIREncounter(BaseModel):
    resourceType: Literal["Encounter"] = "Encounter"
    id: str
    status: str
    class_: Optional[Dict[str, Any]] = Field(default=None, alias="class")
    subject: Dict[str, Any]
    period: Optional[Dict[str, Any]] = None

class FHIRObservation(BaseModel):
    resourceType: Literal["Observation"] = "Observation"
    id: str
    status: str
    code: Dict[str, Any]
    subject: Dict[str, Any]
    effectiveDateTime: Optional[datetime] = None
    valueQuantity: Optional[Dict[str, Any]] = None
    valueString: Optional[str] = None
    valueCodeableConcept: Optional[Dict[str, Any]] = None

class FHIRCondition(BaseModel):
    resourceType: Literal["Condition"] = "Condition"
    id: str
    subject: Dict[str, Any]
    code: Dict[str, Any]
    onsetDateTime: Optional[datetime] = None
    clinicalStatus: Optional[Dict[str, Any]] = None
    verificationStatus: Optional[Dict[str, Any]] = None
