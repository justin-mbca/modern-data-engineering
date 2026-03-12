"""
data_ingestion/ingest_clinical_data.py
=======================================
Clinical data ingestion module for the AI Healthcare Data Platform.

Responsibilities
----------------
* Load raw clinical records from CSV / JSON / FHIR-style JSON bundles.
* Validate mandatory fields (patient_id, encounter_date, diagnosis_code).
* Standardise date formats and ICD-10 code formatting.
* Anonymise / pseudonymise PII columns in place.
* Persist cleansed records to a Parquet staging area for downstream processing.

Usage
-----
    python ingest_clinical_data.py \
        --source data/raw/clinical_records.csv \
        --output data/staging/clinical

    python ingest_clinical_data.py \
        --source data/raw/fhir_bundle.json \
        --format fhir \
        --output data/staging/clinical
"""

import argparse
import hashlib
import logging
import os
import re
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("clinical-ingestion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {"patient_id", "encounter_date", "diagnosis_code"}
PII_COLUMNS = ["patient_name", "date_of_birth", "ssn", "email", "phone"]
ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$")


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_csv(source_path: str) -> pd.DataFrame:
    """Load clinical records from a CSV file."""
    logger.info("Reading CSV: %s", source_path)
    df = pd.read_csv(source_path, dtype=str)
    logger.info("Loaded %d rows from CSV.", len(df))
    return df


def extract_json(source_path: str) -> pd.DataFrame:
    """Load clinical records from a flat JSON file."""
    logger.info("Reading JSON: %s", source_path)
    df = pd.read_json(source_path, dtype=str)
    logger.info("Loaded %d rows from JSON.", len(df))
    return df


def extract_fhir(source_path: str) -> pd.DataFrame:
    """
    Parse a FHIR R4 Bundle (JSON) and flatten Encounter resources.

    Only the most common FHIR fields are extracted; extend as needed for
    your specific bundle structure.
    """
    import json

    logger.info("Reading FHIR bundle: %s", source_path)
    with open(source_path, encoding="utf-8") as fh:
        bundle = json.load(fh)

    records = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Encounter":
            continue
        patient_ref = resource.get("subject", {}).get("reference", "")
        patient_id = patient_ref.split("/")[-1] if "/" in patient_ref else patient_ref
        encounter_date = resource.get("period", {}).get("start", "")[:10]
        diagnosis_list = resource.get("diagnosis", [])
        for diag in diagnosis_list:
            code = (
                diag.get("condition", {})
                .get("code", {})
                .get("coding", [{}])[0]
                .get("code", "")
            )
            records.append(
                {
                    "patient_id": patient_id,
                    "encounter_date": encounter_date,
                    "diagnosis_code": code,
                    "encounter_id": resource.get("id", ""),
                    "status": resource.get("status", ""),
                }
            )

    df = pd.DataFrame(records).astype(str)
    logger.info("Extracted %d Encounter records from FHIR bundle.", len(df))
    return df


def extract(source_path: str, fmt: str = "csv") -> pd.DataFrame:
    """
    Dispatch extraction to the appropriate loader.

    Parameters
    ----------
    source_path : str
        Path to the source file.
    fmt : str
        One of ``"csv"``, ``"json"``, or ``"fhir"``.

    Returns
    -------
    pd.DataFrame
        Raw clinical records.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    loaders = {"csv": extract_csv, "json": extract_json, "fhir": extract_fhir}
    if fmt not in loaders:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from {list(loaders)}.")
    return loaders[fmt](source_path)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate mandatory columns and remove invalid rows.

    Rules
    -----
    * Rows missing ``patient_id``, ``encounter_date``, or ``diagnosis_code``
      are dropped and logged.
    * Rows with malformed ICD-10 codes are flagged in a new
      ``diagnosis_code_valid`` boolean column (not dropped).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with invalid rows removed and a validation flag column added.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Source data is missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing required fields.", dropped)

    df["diagnosis_code"] = df["diagnosis_code"].str.upper().str.strip()
    df["diagnosis_code_valid"] = df["diagnosis_code"].str.match(ICD10_PATTERN)
    invalid_count = (~df["diagnosis_code_valid"]).sum()
    if invalid_count:
        logger.warning(
            "%d rows have non-standard ICD-10 codes (kept but flagged).", invalid_count
        )

    return df


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def _pseudonymise(value: str) -> str:
    """Return a SHA-256 hex digest of *value* (first 16 chars)."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise and anonymise clinical records.

    Steps
    -----
    1. Normalise ``encounter_date`` to ISO-8601 (``YYYY-MM-DD``).
    2. Pseudonymise PII columns found in the DataFrame.
    3. Strip leading/trailing whitespace from all string columns.
    4. Append an ``ingested_at`` audit timestamp.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Transforming %d clinical records …", len(df))

    # 1. Normalise encounter dates
    df["encounter_date"] = pd.to_datetime(
        df["encounter_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # 2. Pseudonymise PII columns
    for col in PII_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: _pseudonymise(str(v)) if pd.notna(v) and str(v).strip() else v
            )
            logger.info("Pseudonymised column '%s'.", col)

    # 3. Trim whitespace
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # 4. Audit timestamp
    df["ingested_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    logger.info("Transformation complete.")
    return df


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(df: pd.DataFrame, output_dir: str) -> str:
    """
    Write the cleansed clinical records to a Parquet file in *output_dir*.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str

    Returns
    -------
    str
        Path of the written Parquet file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(output_dir, f"clinical_{timestamp}.parquet")
    df.to_parquet(out_path, index=False)
    logger.info("Written %d clinical records → %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(source_path: str, output_dir: str, fmt: str = "csv") -> None:
    """Execute Extract → Validate → Transform → Load for clinical data."""
    logger.info("=== Clinical data ingestion start ===")
    raw = extract(source_path, fmt=fmt)
    validated = validate(raw)
    cleansed = transform(validated)
    load(cleansed, output_dir)
    logger.info("=== Clinical data ingestion complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Healthcare Platform — Clinical Data Ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Path to source file.")
    parser.add_argument(
        "--format",
        dest="fmt",
        default="csv",
        choices=["csv", "json", "fhir"],
        help="Source file format.",
    )
    parser.add_argument(
        "--output",
        default="data/staging/clinical",
        help="Output directory for Parquet staging files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(source_path=args.source, output_dir=args.output, fmt=args.fmt)
