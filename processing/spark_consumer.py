"""
Real-Time Data Pipeline – Spark Structured Streaming Consumer.

Reads events from Kafka topics, applies schema-based parsing and
transformations, then writes results to Delta Lake (micro-batch) and
optionally forwards enriched messages to a downstream message queue.
"""

import logging

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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Event schemas
# ---------------------------------------------------------------------------

USER_ACTION_SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_type", StringType()),
        StructField("user_id", StringType()),
        StructField("action", StringType()),
        StructField("timestamp", StringType()),
    ]
)

ORDER_SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_type", StringType()),
        StructField("order_id", StringType()),
        StructField("user_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("status", StringType()),
        StructField("timestamp", StringType()),
    ]
)

PAGE_VIEW_SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_type", StringType()),
        StructField("user_id", StringType()),
        StructField("page", StringType()),
        StructField("duration_ms", IntegerType()),
        StructField("timestamp", StringType()),
    ]
)

SCHEMA_MAP = {
    "user-actions": USER_ACTION_SCHEMA,
    "orders": ORDER_SCHEMA,
    "page-views": PAGE_VIEW_SCHEMA,
}


def create_spark_session(app_name: str = "RealTimePipeline") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def read_kafka_stream(spark: SparkSession, brokers: str, topics: str):
    """
    Create a Kafka streaming DataFrame.

    Args:
        spark: Active SparkSession.
        brokers: Kafka bootstrap server address(es).
        topics: Comma-separated list of topic names to subscribe to.

    Returns:
        Streaming DataFrame with raw Kafka columns.
    """
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", brokers)
        .option("subscribe", topics)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )


def parse_events(raw_df, schema: StructType):
    """
    Deserialise the Kafka value column using the given schema.

    Args:
        raw_df: Raw Kafka streaming DataFrame.
        schema: Target schema for JSON parsing.

    Returns:
        Parsed streaming DataFrame.
    """
    return (
        raw_df.selectExpr("CAST(value AS STRING) AS json_value", "topic", "timestamp AS kafka_ts")
        .withColumn("data", F.from_json(F.col("json_value"), schema))
        .select("topic", "kafka_ts", "data.*")
        .withColumn("ingested_at", F.current_timestamp())
    )


def write_to_delta(streaming_df, output_path: str, checkpoint_path: str):
    """
    Write a streaming DataFrame to Delta Lake using micro-batch processing.

    Args:
        streaming_df: Parsed streaming DataFrame.
        output_path: Delta Lake destination path.
        checkpoint_path: Checkpoint location for fault-tolerance.

    Returns:
        StreamingQuery handle.
    """
    return (
        streaming_df.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime="30 seconds")
        .start(output_path)
    )


def start_streaming_pipeline(
    brokers: str = "localhost:9092",
    topics: str = "user-actions,orders,page-views",
    output_base: str = "/data/delta/realtime",
    checkpoint_base: str = "/data/checkpoints/realtime",
) -> None:
    """
    Start the full Spark Structured Streaming pipeline.

    Args:
        brokers: Kafka broker address(es).
        topics: Comma-separated topic names to consume.
        output_base: Base path for Delta Lake output (one sub-folder per topic).
        checkpoint_base: Base path for Spark checkpoints.
    """
    spark = create_spark_session()
    queries = []

    for topic in [t.strip() for t in topics.split(",")]:
        schema = SCHEMA_MAP.get(topic)
        if schema is None:
            logger.warning("No schema found for topic '%s' – skipping.", topic)
            continue

        logger.info("Starting stream for topic: %s", topic)
        raw_stream = read_kafka_stream(spark, brokers, topic)
        parsed_stream = parse_events(raw_stream, schema)

        query = write_to_delta(
            parsed_stream,
            output_path=f"{output_base}/{topic}",
            checkpoint_path=f"{checkpoint_base}/{topic}",
        )
        queries.append(query)
        logger.info("Stream query started for topic '%s'", topic)

    # Block until all queries terminate
    for q in queries:
        q.awaitTermination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spark Structured Streaming consumer")
    parser.add_argument("--brokers", default="localhost:9092")
    parser.add_argument("--topics", default="user-actions,orders,page-views")
    parser.add_argument("--output", default="/data/delta/realtime")
    parser.add_argument("--checkpoint", default="/data/checkpoints/realtime")
    args = parser.parse_args()

    start_streaming_pipeline(
        brokers=args.brokers,
        topics=args.topics,
        output_base=args.output,
        checkpoint_base=args.checkpoint,
    )
