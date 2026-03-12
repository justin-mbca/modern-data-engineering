"""
Airflow DAG – Daily Batch ETL Run.

Schedules the Modern Data Pipeline ETL to execute every day at midnight UTC.
The DAG:
1. Validates source data with Great Expectations.
2. Runs the PySpark ETL job (extract → transform → load to Delta Lake).
3. Records Prometheus metrics for the run.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# Pipeline configuration – override via Airflow Variables in production
# ---------------------------------------------------------------------------

SOURCE_PATH = "/data/raw/source"
TARGET_PATH = "/data/delta/output"
FILE_FORMAT = "csv"
WRITE_MODE = "append"
NOT_NULL_COLUMNS = ["id", "event_date", "value"]
METRICS_PORT = 8000


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def task_validate_data(**context):
    """Run Great Expectations validation against the raw source data."""
    from pyspark.sql import SparkSession
    from data_quality import validate_dataframe

    spark = (
        SparkSession.builder.appName("DataQualityCheck")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )
    try:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(SOURCE_PATH)
        result = validate_dataframe(df, not_null_columns=NOT_NULL_COLUMNS)
        context["ti"].xcom_push(key="validation_success", value=result["success"])
    finally:
        spark.stop()


def task_run_etl(**context):
    """Execute the full ETL pipeline and push row-count metrics to XCom."""
    from monitoring import ETLMetrics, start_metrics_server
    from etl import run_etl

    start_metrics_server(port=METRICS_PORT)
    metrics = ETLMetrics()

    run_etl(
        source_path=SOURCE_PATH,
        target_path=TARGET_PATH,
        file_format=FILE_FORMAT,
        write_mode=WRITE_MODE,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="modern_data_pipeline_daily_etl",
    description="Daily batch ETL run for the Modern Data Pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["etl", "batch", "delta-lake"],
) as dag:

    validate_data = PythonOperator(
        task_id="validate_source_data",
        python_callable=task_validate_data,
        provide_context=True,
    )

    run_etl_task = PythonOperator(
        task_id="run_etl_pipeline",
        python_callable=task_run_etl,
        provide_context=True,
    )

    validate_data >> run_etl_task
