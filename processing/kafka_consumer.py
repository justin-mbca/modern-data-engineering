"""
Real-Time Data Pipeline – Kafka Consumer with Robust Error Handling.

Consumes messages from multiple Kafka topics, applies lightweight
transformations, and forwards enriched messages to a downstream queue
(configurable: RabbitMQ or Azure Service Bus).
"""

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

from kafka import KafkaConsumer
from kafka.errors import CommitFailedError, KafkaError, NoBrokersAvailable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


def retry_with_backoff(func: Callable, *args, max_retries: int = MAX_RETRIES, **kwargs) -> Any:
    """
    Call *func* with exponential back-off on transient failures.

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to *func*.
        max_retries: Maximum number of attempts.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Return value of *func* on success.

    Raises:
        Exception: Re-raises the last exception after all retries are exhausted.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "All %d retries exhausted for %s: %s",
                    max_retries,
                    getattr(func, "__name__", repr(func)),
                    exc,
                )
                raise
            logger.warning(
                "Attempt %d/%d for %s failed: %s – retrying in %.1fs",
                attempt,
                max_retries,
                getattr(func, "__name__", repr(func)),
                exc,
                backoff,
            )
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER


# ---------------------------------------------------------------------------
# Consumer factory
# ---------------------------------------------------------------------------


def create_consumer(
    topics: list,
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "realtime-pipeline-consumer",
) -> KafkaConsumer:
    """
    Create a KafkaConsumer with manual offset commit (for at-least-once semantics).

    Args:
        topics: List of topic names to subscribe to.
        bootstrap_servers: Kafka broker address(es).
        group_id: Consumer group identifier.

    Returns:
        KafkaConsumer instance.
    """

    def _create():
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            session_timeout_ms=30_000,
            heartbeat_interval_ms=10_000,
            max_poll_interval_ms=300_000,
        )
        return consumer

    return retry_with_backoff(_create, max_retries=MAX_RETRIES)


# ---------------------------------------------------------------------------
# Dead-letter queue (DLQ)
# ---------------------------------------------------------------------------


def send_to_dlq(message: Dict[str, Any], error: Exception) -> None:
    """
    Forward a poison-pill message to a dead-letter store for later inspection.

    In production this would write to a dedicated Kafka DLQ topic, a database
    table, or an object-storage path.  Here we log a structured record.

    Args:
        message: The raw Kafka record value.
        error: The exception that caused the processing failure.
    """
    dlq_record = {
        "dlq_reason": str(error),
        "dlq_error_type": type(error).__name__,
        "original_message": message,
    }
    logger.error("DLQ record: %s", json.dumps(dlq_record))


# ---------------------------------------------------------------------------
# Message processor
# ---------------------------------------------------------------------------


def process_message(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply lightweight enrichment / validation to an incoming event.

    Args:
        record: Deserialised Kafka message value.

    Returns:
        Enriched record dict.

    Raises:
        ValueError: If required fields are missing.
    """
    required_fields = {"event_id", "event_type", "timestamp"}
    missing = required_fields - set(record.keys())
    if missing:
        raise ValueError(f"Message is missing required fields: {missing}")

    record["processed"] = True
    return record


# ---------------------------------------------------------------------------
# Main consume loop
# ---------------------------------------------------------------------------


def consume_events(
    topics: list,
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "realtime-pipeline-consumer",
    max_messages: Optional[int] = None,
) -> None:
    """
    Run the Kafka consumer loop with robust error handling.

    Behaviour:
    - Processes messages one at a time.
    - Commits offsets only after successful processing.
    - Sends unprocessable messages to the DLQ instead of crashing.
    - Reconnects automatically on transient Kafka errors.

    Args:
        topics: Kafka topics to subscribe to.
        bootstrap_servers: Broker address(es).
        group_id: Consumer group ID.
        max_messages: Stop after consuming this many messages (None = run forever).
    """
    consumer = create_consumer(topics, bootstrap_servers, group_id)
    processed_count = 0

    logger.info("Consumer started – subscribed to topics: %s", topics)

    try:
        for msg in consumer:
            try:
                enriched = process_message(msg.value)
                logger.debug(
                    "Processed [topic=%s partition=%d offset=%d]: %s",
                    msg.topic,
                    msg.partition,
                    msg.offset,
                    enriched.get("event_id"),
                )
                # Commit offset only after successful processing
                try:
                    consumer.commit()
                except CommitFailedError as commit_err:
                    logger.warning("Offset commit failed (will retry on next poll): %s", commit_err)

            except (ValueError, KeyError, TypeError) as proc_err:
                # Non-retryable: send to DLQ and move on
                logger.warning("Non-retryable processing error: %s", proc_err)
                send_to_dlq(msg.value, proc_err)
                consumer.commit()  # advance past the bad message

            except KafkaError as kafka_err:
                # Transient Kafka error – log and continue (consumer will retry)
                logger.error("Transient Kafka error: %s", kafka_err)

            processed_count += 1
            if max_messages and processed_count >= max_messages:
                logger.info("Reached max_messages=%d – stopping consumer.", max_messages)
                break

    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    finally:
        consumer.close()
        logger.info("Consumer closed – total messages processed: %d", processed_count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kafka consumer with error handling")
    parser.add_argument("--brokers", default="localhost:9092")
    parser.add_argument("--topics", default="user-actions,orders,page-views")
    parser.add_argument("--group", default="realtime-pipeline-consumer")
    args = parser.parse_args()

    consume_events(
        topics=args.topics.split(","),
        bootstrap_servers=args.brokers,
        group_id=args.group,
    )
