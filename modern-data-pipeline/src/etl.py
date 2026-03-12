"""
modern-data-pipeline/src/etl.py
================================
End-to-end ETL (Extract → Transform → Load) pipeline for the Modern Data
Pipeline project.

Responsibilities
----------------
* Extract raw data from a configurable source (CSV, JSON, or a relational DB).
* Apply lightweight transformations (type coercion, null handling, dedup).
* Load the cleansed data into a Delta-Lake-style Parquet table.

Usage
-----
    python etl.py --source data/raw/sales.csv --output storage/delta_table
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
logger = logging.getLogger("modern-etl")


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract(source_path: str) -> pd.DataFrame:
    """
    Load raw data from *source_path*.

    Supports ``.csv`` and ``.json`` files.  For database connectivity swap
    this function with a SQLAlchemy-backed implementation.

    Parameters
    ----------
    source_path : str
        Absolute or relative path to the source file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified data as a DataFrame.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If *source_path* does not exist.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    ext = os.path.splitext(source_path)[-1].lower()
    logger.info("Extracting data from '%s' (format: %s)", source_path, ext)

    if ext == ".csv":
        df = pd.read_csv(source_path)
    elif ext == ".json":
        df = pd.read_json(source_path)
    else:
        raise ValueError(f"Unsupported source format '{ext}'. Use .csv or .json.")

    logger.info("Extracted %d rows × %d columns.", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a standard set of cleansing and enrichment transformations.

    Steps applied in order
    ~~~~~~~~~~~~~~~~~~~~~~
    1. Strip leading/trailing whitespace from string columns.
    2. Drop fully duplicate rows.
    3. Drop rows where *all* values are null.
    4. Coerce obvious numeric columns (those whose values parse as numbers).
    5. Append an ``etl_processed_at`` audit timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame produced by :func:`extract`.

    Returns
    -------
    pd.DataFrame
        Cleansed and enriched DataFrame.
    """
    logger.info("Starting transformation (input: %d rows).", len(df))

    # 1. Trim string whitespace
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # 2. Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dedup: removed %d duplicate rows.", before - len(df))

    # 3. Drop all-null rows
    before = len(df)
    df = df.dropna(how="all")
    logger.info("Null purge: removed %d all-null rows.", before - len(df))

    # 4. Coerce numeric-looking columns
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted

    # 5. Audit timestamp
    df["etl_processed_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    logger.info("Transformation complete (output: %d rows).", len(df))
    return df


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(df: pd.DataFrame, output_dir: str, partition_col: str | None = None) -> str:
    """
    Persist the transformed DataFrame to a Parquet file in *output_dir*.

    The output filename encodes the UTC timestamp of the run so multiple
    pipeline executions never overwrite each other.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed DataFrame ready for storage.
    output_dir : str
        Target directory.  Created automatically if it does not exist.
    partition_col : str or None
        Optional column name to use for Hive-style directory partitioning
        (e.g. ``"region"`` → ``output_dir/region=EMEA/…``).

    Returns
    -------
    str
        Path of the written Parquet file (or partition root directory).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if partition_col and partition_col in df.columns:
        # Write one Parquet file per partition value
        for value, group in df.groupby(partition_col):
            part_dir = os.path.join(output_dir, f"{partition_col}={value}")
            os.makedirs(part_dir, exist_ok=True)
            out_path = os.path.join(part_dir, f"data_{timestamp}.parquet")
            group.drop(columns=[partition_col]).to_parquet(out_path, index=False)
            logger.info("Written partition '%s=%s' → %s", partition_col, value, out_path)
        return output_dir
    else:
        out_path = os.path.join(output_dir, f"data_{timestamp}.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Written %d rows → %s", len(df), out_path)
        return out_path


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(source_path: str, output_dir: str, partition_col: str | None = None) -> None:
    """
    Execute the full Extract → Transform → Load pipeline.

    Parameters
    ----------
    source_path : str
        Path to the raw source file.
    output_dir : str
        Destination directory for Parquet output.
    partition_col : str or None
        Optional partitioning column (forwarded to :func:`load`).
    """
    logger.info("=== Pipeline start ===")
    raw_df = extract(source_path)
    clean_df = transform(raw_df)
    out = load(clean_df, output_dir, partition_col=partition_col)
    logger.info("=== Pipeline complete — output: %s ===", out)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modern Data Pipeline — ETL runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the source CSV or JSON file.",
    )
    parser.add_argument(
        "--output",
        default="storage/delta_table",
        help="Output directory for Parquet files.",
    )
    parser.add_argument(
        "--partition-col",
        default=None,
        help="Column to use for Hive-style directory partitioning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        source_path=args.source,
        output_dir=args.output,
        partition_col=args.partition_col,
    )

