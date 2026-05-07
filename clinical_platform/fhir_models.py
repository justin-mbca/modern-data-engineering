"""
FHIR Resource Models for Clinical Platform
Defines Pydantic models for core FHIR resources: Patient, Encounter, Observation, Condition.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime

class FHIRPatient(BaseModel):
    resourceType: str = Field("Patient", const=True)
    id: str
    name: List[Dict[str, Any]]
    gender: Optional[str]
    birthDate: Optional[date]
    address: Optional[List[Dict[str, Any]]]

class FHIREncounter(BaseModel):
    resourceType: str = Field("Encounter", const=True)
    id: str
    status: str
    class_: Optional[Dict[str, Any]] = Field(None, alias="class")
    subject: Dict[str, Any]
    period: Optional[Dict[str, Any]]

class FHIRObservation(BaseModel):
    resourceType: str = Field("Observation", const=True)
    id: str
    status: str
    code: Dict[str, Any]
    subject: Dict[str, Any]
    effectiveDateTime: Optional[datetime]
    valueQuantity: Optional[Dict[str, Any]]
    valueString: Optional[str]
    valueCodeableConcept: Optional[Dict[str, Any]]

class FHIRCondition(BaseModel):
    resourceType: str = Field("Condition", const=True)
    id: str
    subject: Dict[str, Any]
    code: Dict[str, Any]
    onsetDateTime: Optional[datetime]
    clinicalStatus: Optional[Dict[str, Any]]
    verificationStatus: Optional[Dict[str, Any]]
