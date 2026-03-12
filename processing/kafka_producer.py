"""
Real-Time Data Pipeline – Kafka Producer.

Simulates multiple event types (user_action, order, page_view) and publishes
them to dedicated Kafka topics with configurable throughput.
"""

import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Event-type definitions
# ---------------------------------------------------------------------------

EVENT_TYPES = ["user_action", "order", "page_view"]

TOPIC_MAP = {
    "user_action": "user-actions",
    "order": "orders",
    "page_view": "page-views",
}


def _build_user_action_event(user_id: str) -> Dict[str, Any]:
    actions = ["login", "logout", "click", "search", "add_to_cart"]
    return {
        "event_type": "user_action",
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "action": random.choice(actions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_order_event(user_id: str) -> Dict[str, Any]:
    return {
        "event_type": "order",
        "event_id": str(uuid.uuid4()),
        "order_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": round(random.uniform(5.0, 500.0), 2),
        "currency": "USD",
        "status": random.choice(["placed", "confirmed", "shipped", "delivered"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_page_view_event(user_id: str) -> Dict[str, Any]:
    pages = ["/home", "/products", "/cart", "/checkout", "/account"]
    return {
        "event_type": "page_view",
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "page": random.choice(pages),
        "duration_ms": random.randint(100, 10000),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


EVENT_BUILDERS = {
    "user_action": _build_user_action_event,
    "order": _build_order_event,
    "page_view": _build_page_view_event,
}


# ---------------------------------------------------------------------------
# Producer helpers
# ---------------------------------------------------------------------------


def _on_send_success(record_metadata):
    logger.debug(
        "Message delivered to %s [partition=%d, offset=%d]",
        record_metadata.topic,
        record_metadata.partition,
        record_metadata.offset,
    )


def _on_send_error(exc):
    logger.error("Failed to deliver message: %s", exc)


def create_producer(bootstrap_servers: str = "localhost:9092") -> KafkaProducer:
    """
    Create and return a configured KafkaProducer.

    Args:
        bootstrap_servers: Comma-separated Kafka broker addresses.

    Returns:
        KafkaProducer instance.
    """
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",                  # strongest delivery guarantee
        retries=3,
        retry_backoff_ms=500,
        request_timeout_ms=30_000,
        compression_type="gzip",
    )


def produce_events(
    producer: KafkaProducer,
    num_events: int = 100,
    events_per_second: float = 10.0,
) -> None:
    """
    Produce a stream of mixed event types to their respective Kafka topics.

    Args:
        producer: Active KafkaProducer.
        num_events: Total number of events to produce.
        events_per_second: Target throughput (approximate).
    """
    delay = 1.0 / events_per_second
    user_pool = [str(uuid.uuid4()) for _ in range(50)]

    for i in range(num_events):
        event_type = random.choice(EVENT_TYPES)
        user_id = random.choice(user_pool)
        event = EVENT_BUILDERS[event_type](user_id)
        topic = TOPIC_MAP[event_type]

        try:
            future = producer.send(topic, key=event["event_id"], value=event)
            future.add_callback(_on_send_success).add_errback(_on_send_error)
        except KafkaError as exc:
            logger.error("KafkaError while sending event %d: %s", i, exc)

        if i % 50 == 0:
            producer.flush()
            logger.info("Produced %d / %d events", i, num_events)

        time.sleep(delay)

    producer.flush()
    logger.info("Finished producing %d events across topics %s", num_events, list(TOPIC_MAP.values()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kafka event producer")
    parser.add_argument("--brokers", default="localhost:9092")
    parser.add_argument("--num-events", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=10.0, help="Events per second")
    args = parser.parse_args()

    p = create_producer(args.brokers)
    try:
        produce_events(p, num_events=args.num_events, events_per_second=args.rate)
    finally:
        p.close()
