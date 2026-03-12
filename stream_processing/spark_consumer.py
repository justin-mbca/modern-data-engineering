"""
stream_processing/spark_consumer.py
=====================================
PySpark Structured Streaming consumer for the Real-Time Data Pipeline.

Responsibilities
----------------
* Read a continuous stream of JSON transaction events from a Kafka topic.
* Parse and cast the raw JSON payload into a typed schema.
* Compute real-time aggregations (revenue per region per minute).
* Write enriched records to Parquet (or Delta) and aggregations to the console.

Environment Variables
---------------------
KAFKA_BOOTSTRAP_SERVERS  Comma-separated broker list.  Default: ``localhost:9092``
KAFKA_TOPIC              Topic to subscribe to.  Default: ``transactions``
CHECKPOINT_DIR           Spark checkpoint location.  Default: ``/tmp/spark-checkpoint``
OUTPUT_DIR               Parquet output for enriched stream.
                         Default: ``storage/delta_table/transactions``

Usage
-----
    spark-submit stream_processing/spark_consumer.py

    # Or directly with Python (requires a local Spark installation):
    python stream_processing/spark_consumer.py
"""

import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("spark-consumer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC: str = os.getenv("KAFKA_TOPIC", "transactions")
CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", "/tmp/spark-checkpoint")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "storage/delta_table/transactions")

# ---------------------------------------------------------------------------
# Schema definition (mirrors the event produced by kafka_producer.py)
# ---------------------------------------------------------------------------
EVENT_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), nullable=False),
        StructField("timestamp", StringType(), nullable=False),
        StructField("product", StringType(), nullable=True),
        StructField("quantity", IntegerType(), nullable=True),
        StructField("unit_price", DoubleType(), nullable=True),
        StructField("total_amount", DoubleType(), nullable=True),
        StructField("currency", StringType(), nullable=True),
        StructField("region", StringType(), nullable=True),
        StructField("customer_id", StringType(), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------

def build_spark() -> SparkSession:
    """
    Create a :class:`SparkSession` configured for Kafka-based streaming.

    The session is set up with the ``spark-sql-kafka`` connector package.
    Adjust ``spark.jars.packages`` for your Spark / Kafka version.
    """
    logger.info("Initialising SparkSession …")
    spark = (
        SparkSession.builder.appName("RealTimePipeline-KafkaConsumer")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        )
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession ready.")
    return spark


# ---------------------------------------------------------------------------
# Stream processing helpers
# ---------------------------------------------------------------------------

def read_kafka_stream(spark: SparkSession):
    """
    Return a streaming DataFrame backed by the configured Kafka topic.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.

    Returns
    -------
    pyspark.sql.DataFrame
        Streaming DataFrame with columns ``key``, ``value``, ``topic``,
        ``partition``, ``offset``, ``timestamp``, ``timestampType``.
    """
    logger.info("Subscribing to Kafka topic '%s' at %s …", TOPIC, BOOTSTRAP_SERVERS)
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )


def parse_events(raw_df):
    """
    Decode the raw Kafka ``value`` bytes into a typed DataFrame.

    Parameters
    ----------
    raw_df : pyspark.sql.DataFrame
        Raw streaming DataFrame from Kafka (``value`` column is binary).

    Returns
    -------
    pyspark.sql.DataFrame
        Parsed DataFrame with columns matching :data:`EVENT_SCHEMA` plus
        an additional ``event_time`` :class:`~pyspark.sql.types.TimestampType`
        column derived from the ``timestamp`` string field.
    """
    parsed = (
        raw_df.select(
            F.from_json(F.col("value").cast("string"), EVENT_SCHEMA).alias("data")
        )
        .select("data.*")
        .withColumn("event_time", F.to_timestamp(F.col("timestamp")))
    )
    return parsed


def compute_aggregations(parsed_df):
    """
    Compute a 1-minute tumbling-window revenue aggregation per region.

    Parameters
    ----------
    parsed_df : pyspark.sql.DataFrame
        Typed event DataFrame with an ``event_time`` column.

    Returns
    -------
    pyspark.sql.DataFrame
        Aggregated streaming DataFrame with columns:
        ``window``, ``region``, ``total_revenue``, ``transaction_count``.
    """
    return (
        parsed_df.withWatermark("event_time", "2 minutes")
        .groupBy(
            F.window(F.col("event_time"), "1 minute"),
            F.col("region"),
        )
        .agg(
            F.sum("total_amount").alias("total_revenue"),
            F.count("event_id").alias("transaction_count"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.col("region"),
            F.round(F.col("total_revenue"), 2).alias("total_revenue"),
            F.col("transaction_count"),
        )
    )


# ---------------------------------------------------------------------------
# Stream writers
# ---------------------------------------------------------------------------

def start_enriched_writer(parsed_df, output_dir: str, checkpoint_dir: str):
    """
    Write the enriched event stream to Parquet files.

    Parameters
    ----------
    parsed_df : pyspark.sql.DataFrame
        Typed, enriched streaming DataFrame.
    output_dir : str
        Root directory for Parquet output.
    checkpoint_dir : str
        Spark checkpoint root.  A sub-directory ``enriched/`` is used.

    Returns
    -------
    pyspark.sql.streaming.StreamingQuery
    """
    return (
        parsed_df.writeStream.format("parquet")
        .option("path", output_dir)
        .option("checkpointLocation", os.path.join(checkpoint_dir, "enriched"))
        .partitionBy("region")
        .trigger(processingTime="30 seconds")
        .start()
    )


def start_aggregation_writer(agg_df, checkpoint_dir: str):
    """
    Write the windowed aggregations to the console (stdout) for debugging.

    Switch ``.format("console")`` to ``"parquet"`` or ``"delta"`` in
    production.

    Parameters
    ----------
    agg_df : pyspark.sql.DataFrame
        Aggregated streaming DataFrame.
    checkpoint_dir : str
        Spark checkpoint root.  A sub-directory ``agg/`` is used.

    Returns
    -------
    pyspark.sql.streaming.StreamingQuery
    """
    return (
        agg_df.writeStream.format("console")
        .option("truncate", "false")
        .option("checkpointLocation", os.path.join(checkpoint_dir, "agg"))
        .outputMode("update")
        .trigger(processingTime="30 seconds")
        .start()
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_streaming_pipeline() -> None:
    """
    Build and start the complete Structured Streaming pipeline.

    Blocks until the streaming job is terminated (Ctrl+C or cluster shutdown).
    """
    spark = build_spark()

    raw_df = read_kafka_stream(spark)
    parsed_df = parse_events(raw_df)
    agg_df = compute_aggregations(parsed_df)

    enriched_query = start_enriched_writer(parsed_df, OUTPUT_DIR, CHECKPOINT_DIR)
    agg_query = start_aggregation_writer(agg_df, CHECKPOINT_DIR)

    logger.info(
        "Streaming pipeline running — enriched query: %s | agg query: %s",
        enriched_query.name,
        agg_query.name,
    )

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Interrupt received — stopping streaming queries …")
        enriched_query.stop()
        agg_query.stop()
    finally:
        spark.stop()
        logger.info("SparkSession stopped.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_streaming_pipeline()
