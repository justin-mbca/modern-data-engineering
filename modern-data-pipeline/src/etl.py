"""
Modern Data Pipeline - ETL Process
Batch ETL pipeline for integrating data from multiple sources using PySpark and Delta Lake.
"""

import logging
from datetime import datetime
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

from monitoring import ETLMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "ModernDataPipeline") -> SparkSession:
    """Create and return a configured SparkSession with Delta Lake support."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def extract(spark: SparkSession, source_path: str, file_format: str = "csv") -> DataFrame:
    """
    Extract data from the specified source path.

    Args:
        spark: Active SparkSession.
        source_path: Path to the source data.
        file_format: Format of the source data (csv, json, parquet).

    Returns:
        Raw DataFrame extracted from source.
    """
    logger.info("Extracting data from %s (format=%s)", source_path, file_format)
    reader = spark.read.option("header", "true").option("inferSchema", "true")
    if file_format == "csv":
        df = reader.csv(source_path)
    elif file_format == "json":
        df = reader.json(source_path)
    elif file_format == "parquet":
        df = reader.parquet(source_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    logger.info("Extracted %d rows from source", df.count())
    return df


def transform(df: DataFrame) -> DataFrame:
    """
    Apply business-logic transformations to the raw DataFrame.

    Transformations:
    - Drop duplicate rows.
    - Drop rows with null values in critical columns.
    - Standardise string columns to uppercase.
    - Add an `etl_timestamp` audit column.

    Args:
        df: Raw DataFrame to transform.

    Returns:
        Transformed DataFrame.
    """
    logger.info("Starting transformation on %d rows", df.count())

    # Remove duplicates
    df = df.dropDuplicates()

    # Drop rows with any null value
    df = df.dropna()

    # Normalise string columns to uppercase
    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    for col in string_cols:
        df = df.withColumn(col, F.upper(F.trim(F.col(col))))

    # Add audit timestamp
    df = df.withColumn("etl_timestamp", F.current_timestamp())

    logger.info("Transformation complete – %d rows retained", df.count())
    return df


def load(df: DataFrame, target_path: str, mode: str = "append") -> None:
    """
    Load the transformed DataFrame into a Delta Lake table.

    Args:
        df: Transformed DataFrame to write.
        target_path: Destination Delta Lake path.
        mode: Write mode ('append' or 'overwrite').
    """
    logger.info("Loading data to Delta Lake at %s (mode=%s)", target_path, mode)
    df.write.format("delta").mode(mode).save(target_path)
    logger.info("Load complete")


def run_etl(
    source_path: str,
    target_path: str,
    file_format: str = "csv",
    write_mode: str = "append",
    metrics: Optional[ETLMetrics] = None,
) -> None:
    """
    Orchestrate the full Extract → Transform → Load pipeline.

    Args:
        source_path: Input data location.
        target_path: Delta Lake output location.
        file_format: Source file format.
        write_mode: Delta Lake write mode.
        metrics: Optional ETLMetrics instance for Prometheus instrumentation.
    """
    run_start = datetime.utcnow()
    spark = create_spark_session()

    try:
        # Extract
        raw_df = extract(spark, source_path, file_format)
        rows_extracted = raw_df.count()

        # Transform
        transformed_df = transform(raw_df)
        rows_transformed = transformed_df.count()

        # Load
        load(transformed_df, target_path, write_mode)

        duration = (datetime.utcnow() - run_start).total_seconds()
        logger.info(
            "ETL pipeline completed successfully in %.2fs "
            "(extracted=%d, loaded=%d)",
            duration,
            rows_extracted,
            rows_transformed,
        )

        if metrics:
            metrics.record_run(
                success=True,
                rows_extracted=rows_extracted,
                rows_loaded=rows_transformed,
                duration_seconds=duration,
            )

    except Exception as exc:
        duration = (datetime.utcnow() - run_start).total_seconds()
        logger.error("ETL pipeline failed after %.2fs: %s", duration, exc, exc_info=True)
        if metrics:
            metrics.record_run(success=False, duration_seconds=duration)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Modern Data ETL pipeline")
    parser.add_argument("--source", required=True, help="Source data path")
    parser.add_argument("--target", required=True, help="Delta Lake target path")
    parser.add_argument("--format", default="csv", help="Source file format")
    parser.add_argument("--mode", default="append", help="Delta write mode")
    args = parser.parse_args()

    run_etl(
        source_path=args.source,
        target_path=args.target,
        file_format=args.format,
        write_mode=args.mode,
    )
