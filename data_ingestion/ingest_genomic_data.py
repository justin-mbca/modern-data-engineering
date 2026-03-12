"""
data_ingestion/ingest_genomic_data.py
======================================
Genomic data ingestion and pre-processing module for the AI Healthcare Data
Platform.

Responsibilities
----------------
* Ingest raw genomic variant data from VCF-like TSV / CSV files or JSON.
* Validate required variant fields (sample_id, chromosome, position, ref, alt).
* Filter low-quality variants (quality score, read depth thresholds).
* Annotate variants with consequence types when annotation data is provided.
* Persist cleansed variant records to a Parquet staging area.

Supported input formats
-----------------------
* ``csv``  — Comma-separated file with a header row.
* ``tsv``  — Tab-separated file (common for VCF-derived exports).
* ``json`` — Flat JSON array of variant records.

Usage
-----
    python ingest_genomic_data.py \
        --source data/raw/variants.tsv \
        --format tsv \
        --output data/staging/genomic

    python ingest_genomic_data.py \
        --source data/raw/variants.csv \
        --min-qual 30 \
        --min-depth 10 \
        --output data/staging/genomic
"""

import argparse
import logging
import os
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
logger = logging.getLogger("genomic-ingestion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {"sample_id", "chromosome", "position", "ref", "alt"}
VALID_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
# Normalised canonical bases
VALID_BASES = set("ACGTN")

# Default quality thresholds (overridable via CLI)
DEFAULT_MIN_QUAL = 20.0
DEFAULT_MIN_DEPTH = 5


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract(source_path: str, fmt: str = "csv") -> pd.DataFrame:
    """
    Load raw genomic variant records from *source_path*.

    Parameters
    ----------
    source_path : str
        Path to the source file.
    fmt : str
        ``"csv"``, ``"tsv"``, or ``"json"``.

    Returns
    -------
    pd.DataFrame
        Raw variant records with all columns as strings.

    Raises
    ------
    FileNotFoundError
        If *source_path* does not exist.
    ValueError
        If *fmt* is not supported.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    sep_map = {"csv": ",", "tsv": "\t"}
    logger.info("Reading %s file: %s", fmt.upper(), source_path)

    if fmt in sep_map:
        df = pd.read_csv(source_path, sep=sep_map[fmt], dtype=str, comment="#")
    elif fmt == "json":
        df = pd.read_json(source_path, dtype=str)
    else:
        raise ValueError(f"Unsupported format '{fmt}'. Use csv, tsv, or json.")

    # Normalise column names: strip whitespace, lower-case
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    logger.info("Loaded %d variant records (%d columns).", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check required columns are present and remove structurally invalid rows.

    Validation rules
    ~~~~~~~~~~~~~~~~
    * Missing any of ``sample_id``, ``chromosome``, ``position``, ``ref``,
      ``alt`` → row dropped.
    * ``chromosome`` not in the recognised set → flagged in
      ``chromosome_valid`` column.
    * ``ref`` / ``alt`` containing characters outside ``ACGTN`` → flagged in
      ``allele_valid`` column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    logger.info("Dropped %d rows with missing required fields.", before - len(df))

    # Normalise chromosome representation
    df["chromosome"] = (
        df["chromosome"].str.strip().str.upper().str.replace("CHR", "", regex=False)
    )
    df["chromosome_valid"] = df["chromosome"].isin(VALID_CHROMOSOMES)

    # Validate allele bases (single-nucleotide variants and short indels)
    def _allele_valid(allele: str) -> bool:
        return all(c in VALID_BASES for c in allele.upper())

    df["allele_valid"] = df["ref"].apply(_allele_valid) & df["alt"].apply(_allele_valid)

    invalid_chr = (~df["chromosome_valid"]).sum()
    invalid_allele = (~df["allele_valid"]).sum()
    if invalid_chr:
        logger.warning("%d rows have unrecognised chromosomes (flagged).", invalid_chr)
    if invalid_allele:
        logger.warning("%d rows have invalid allele bases (flagged).", invalid_allele)

    return df


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_variants(
    df: pd.DataFrame,
    min_qual: float = DEFAULT_MIN_QUAL,
    min_depth: int = DEFAULT_MIN_DEPTH,
) -> pd.DataFrame:
    """
    Remove low-confidence variants based on quality and read-depth thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Validated variant DataFrame.
    min_qual : float
        Minimum variant quality score (column ``qual`` if present).
    min_depth : int
        Minimum read depth (column ``depth`` or ``dp`` if present).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    before = len(df)

    if "qual" in df.columns:
        df["qual"] = pd.to_numeric(df["qual"], errors="coerce")
        df = df[df["qual"].isna() | (df["qual"] >= min_qual)]
        logger.info(
            "Quality filter (>= %.1f): retained %d / %d rows.",
            min_qual, len(df), before,
        )

    depth_col = next((c for c in ("depth", "dp") if c in df.columns), None)
    if depth_col:
        before_depth = len(df)
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
        df = df[df[depth_col].isna() | (df[depth_col] >= min_depth)]
        logger.info(
            "Depth filter (>= %d): retained %d / %d rows.",
            min_depth, len(df), before_depth,
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Annotate
# ---------------------------------------------------------------------------

def annotate(df: pd.DataFrame, annotation_path: str | None = None) -> pd.DataFrame:
    """
    Merge variant consequence annotations if an annotation file is supplied.

    The annotation file must be a CSV/TSV with at minimum:
    ``chromosome``, ``position``, ``ref``, ``alt``, ``consequence``.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered variant DataFrame.
    annotation_path : str or None
        Path to a variant annotation file.  If ``None``, returns *df* unchanged.

    Returns
    -------
    pd.DataFrame
    """
    if annotation_path is None:
        logger.info("No annotation file provided — skipping annotation step.")
        return df

    logger.info("Loading annotations from %s …", annotation_path)
    sep = "\t" if annotation_path.endswith(".tsv") else ","
    ann = pd.read_csv(annotation_path, sep=sep, dtype=str)
    ann.columns = ann.columns.str.strip().str.lower().str.replace(" ", "_")

    merge_cols = ["chromosome", "position", "ref", "alt"]
    missing = set(merge_cols) - set(ann.columns)
    if missing:
        logger.warning("Annotation file missing columns %s — skipping.", missing)
        return df

    df = df.merge(
        ann[merge_cols + ["consequence"]],
        on=merge_cols,
        how="left",
    )
    annotated = df["consequence"].notna().sum()
    logger.info("Annotated %d / %d variants with consequence types.", annotated, len(df))
    return df


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(df: pd.DataFrame, output_dir: str) -> str:
    """
    Persist genomic variant records to a Parquet file.

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
    out_path = os.path.join(output_dir, f"genomic_{timestamp}.parquet")
    df["ingested_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    df.to_parquet(out_path, index=False)
    logger.info("Written %d variant records → %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    source_path: str,
    output_dir: str,
    fmt: str = "csv",
    min_qual: float = DEFAULT_MIN_QUAL,
    min_depth: int = DEFAULT_MIN_DEPTH,
    annotation_path: str | None = None,
) -> None:
    """Execute the full genomic data ingestion pipeline."""
    logger.info("=== Genomic data ingestion start ===")
    raw = extract(source_path, fmt=fmt)
    validated = validate(raw)
    filtered = filter_variants(validated, min_qual=min_qual, min_depth=min_depth)
    annotated = annotate(filtered, annotation_path=annotation_path)
    load(annotated, output_dir)
    logger.info("=== Genomic data ingestion complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Healthcare Platform — Genomic Data Ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Path to source variant file.")
    parser.add_argument(
        "--format",
        dest="fmt",
        default="csv",
        choices=["csv", "tsv", "json"],
        help="Source file format.",
    )
    parser.add_argument(
        "--output",
        default="data/staging/genomic",
        help="Output directory for Parquet staging files.",
    )
    parser.add_argument(
        "--min-qual",
        type=float,
        default=DEFAULT_MIN_QUAL,
        help="Minimum variant quality score.",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=DEFAULT_MIN_DEPTH,
        help="Minimum read depth.",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Optional path to a variant annotation file (CSV or TSV).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        source_path=args.source,
        output_dir=args.output,
        fmt=args.fmt,
        min_qual=args.min_qual,
        min_depth=args.min_depth,
        annotation_path=args.annotations,
    )
