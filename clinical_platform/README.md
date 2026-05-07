# Clinical Platform Interoperability Module

This module provides a reference architecture for healthcare interoperability and clinical data engineering, with a focus on FHIR resource ingestion, normalization, and API-led integration.

## Features
- FHIR resource ingestion (Patient, Encounter, Observation, Condition)
- Normalization to analytics-friendly structures
- FastAPI-based API layer with Pydantic validation
- Example in-memory patient store
- Extensible for future HL7/FHIR mapping rules

## Directory Structure
- `fhir_models.py`: Pydantic models for FHIR resources
- `fhir_ingestion.py`: Ingestion and validation utilities
- `normalization.py`: Normalization to analytics/warehouse shapes
- `api/`: FastAPI service for FHIR ingestion and patient retrieval

## Usage
1. Install dependencies:
   ```bash
   pip install fastapi pydantic uvicorn
   ```
2. Run the API:
   ```bash
   uvicorn clinical_platform.api.main:app --reload
   ```
3. Access docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## Extending
- Add new FHIR resource models in `fhir_models.py`
- Add normalization logic in `normalization.py`
- Add new API endpoints in `api/routes.py`

## Security & Architecture
See `ARCHITECTURE.md` and `SECURITY.md` for design and security considerations.
