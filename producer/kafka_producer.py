"""
producer/kafka_producer.py
==========================
Kafka event producer for the Real-Time Data Pipeline.

Responsibilities
----------------
* Connect to a Kafka broker (configurable via environment variables).
* Simulate a continuous stream of e-commerce transaction events.
* Serialise each event as JSON and publish it to a Kafka topic.
* Provide graceful shutdown on SIGINT / SIGTERM.

Environment Variables
---------------------
KAFKA_BOOTSTRAP_SERVERS  Comma-separated list of broker addresses.
                         Default: ``localhost:9092``
KAFKA_TOPIC              Topic to produce messages to.
                         Default: ``transactions``
EVENTS_PER_SECOND        Target throughput (float).  Default: ``5``

Usage
-----
    # Run with default settings
    python kafka_producer.py

    # Override settings inline
    KAFKA_TOPIC=orders EVENTS_PER_SECOND=10 python kafka_producer.py
"""

import json
import logging
import os
import random
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from kafka import KafkaProducer
from kafka.errors import KafkaError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("kafka-producer")

# ---------------------------------------------------------------------------
# Configuration (from environment or defaults)
# ---------------------------------------------------------------------------
BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC: str = os.getenv("KAFKA_TOPIC", "transactions")
EVENTS_PER_SECOND: float = float(os.getenv("EVENTS_PER_SECOND", "5"))
SLEEP_INTERVAL: float = 1.0 / max(EVENTS_PER_SECOND, 0.001)

# Simulated reference data
PRODUCTS = ["laptop", "phone", "tablet", "headphones", "monitor", "keyboard", "mouse"]
REGIONS = ["EMEA", "APAC", "AMER", "LATAM"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY"]


# ---------------------------------------------------------------------------
# Event generator
# ---------------------------------------------------------------------------

def generate_event() -> dict[str, Any]:
    """
    Create a single synthetic transaction event.

    Returns
    -------
    dict
        A dictionary representing a transaction with the following keys:

        ``event_id``
            UUID v4 string uniquely identifying this event.
        ``timestamp``
            ISO-8601 UTC timestamp of when the event was generated.
        ``product``
            Randomly chosen product name.
        ``quantity``
            Integer units purchased (1–10).
        ``unit_price``
            Price per unit in the chosen currency (float, 2 d.p.).
        ``total_amount``
            ``quantity × unit_price``.
        ``currency``
            Three-letter ISO currency code.
        ``region``
            Geographic sales region.
        ``customer_id``
            Synthetic customer identifier.
    """
    quantity = random.randint(1, 10)
    unit_price = round(random.uniform(9.99, 999.99), 2)
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds"),
        "product": random.choice(PRODUCTS),
        "quantity": quantity,
        "unit_price": unit_price,
        "total_amount": round(quantity * unit_price, 2),
        "currency": random.choice(CURRENCIES),
        "region": random.choice(REGIONS),
        "customer_id": f"CUST-{random.randint(1000, 9999)}",
    }


# ---------------------------------------------------------------------------
# Delivery callbacks
# ---------------------------------------------------------------------------

def _on_send_success(record_metadata: Any) -> None:
    logger.debug(
        "Delivered → topic=%s partition=%d offset=%d",
        record_metadata.topic,
        record_metadata.partition,
        record_metadata.offset,
    )


def _on_send_error(exc: KafkaError) -> None:
    logger.error("Failed to deliver message: %s", exc)


# ---------------------------------------------------------------------------
# Producer lifecycle
# ---------------------------------------------------------------------------

def build_producer() -> KafkaProducer:
    """
    Construct and return a :class:`KafkaProducer` instance.

    The producer is configured to:
    * Serialise message values as UTF-8 encoded JSON.
    * Retry up to 5 times on transient errors.
    * Wait for all in-sync replica acknowledgements (``acks='all'``).
    """
    logger.info("Connecting to Kafka brokers at %s …", BOOTSTRAP_SERVERS)
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=5,
        linger_ms=5,
    )


def produce_events(producer: KafkaProducer) -> None:
    """
    Continuously generate and publish events until interrupted.

    Parameters
    ----------
    producer : KafkaProducer
        An already-connected Kafka producer.
    """
    sent = 0
    logger.info(
        "Producing to topic '%s' at %.1f events/s. Press Ctrl+C to stop.",
        TOPIC,
        EVENTS_PER_SECOND,
    )

    while True:
        event = generate_event()
        (
            producer.send(TOPIC, value=event)
            .add_callback(_on_send_success)
            .add_errback(_on_send_error)
        )
        sent += 1
        if sent % 100 == 0:
            logger.info("Published %d events so far …", sent)
        time.sleep(SLEEP_INTERVAL)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    producer: KafkaProducer | None = None

    def _shutdown(signum: int, _frame: Any) -> None:
        logger.info("Received signal %d — flushing and shutting down …", signum)
        if producer is not None:
            producer.flush(timeout=10)
            producer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        producer = build_producer()
        produce_events(producer)
    except Exception:
        logger.exception("Unhandled exception in producer — exiting.")
        if producer is not None:
            producer.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
