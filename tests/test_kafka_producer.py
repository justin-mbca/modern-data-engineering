"""Unit tests for the Kafka producer module."""
import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "processing"))

from kafka_producer import (
    EVENT_TYPES,
    TOPIC_MAP,
    _build_order_event,
    _build_page_view_event,
    _build_user_action_event,
    produce_events,
)


class TestEventBuilders:
    def test_user_action_schema(self):
        event = _build_user_action_event("user-123")
        assert event["event_type"] == "user_action"
        assert "event_id" in event
        assert "user_id" in event
        assert event["user_id"] == "user-123"
        assert "action" in event
        assert "timestamp" in event

    def test_order_schema(self):
        event = _build_order_event("user-456")
        assert event["event_type"] == "order"
        assert isinstance(event["amount"], float)
        assert event["amount"] > 0
        assert "order_id" in event

    def test_page_view_schema(self):
        event = _build_page_view_event("user-789")
        assert event["event_type"] == "page_view"
        assert isinstance(event["duration_ms"], int)
        assert "page" in event


class TestTopicMapping:
    def test_all_event_types_have_topics(self):
        for event_type in EVENT_TYPES:
            assert event_type in TOPIC_MAP

    def test_topic_map_values_are_strings(self):
        for topic in TOPIC_MAP.values():
            assert isinstance(topic, str)


class TestProduceEvents:
    @patch("kafka_producer.KafkaProducer")
    def test_produce_calls_send(self, mock_producer_class):
        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_future.add_callback.return_value = mock_future
        mock_future.add_errback.return_value = mock_future
        mock_producer.send.return_value = mock_future

        produce_events(mock_producer, num_events=5, events_per_second=1000)

        assert mock_producer.send.call_count == 5
        mock_producer.flush.assert_called()

    @patch("kafka_producer.KafkaProducer")
    def test_produce_flushes_on_completion(self, mock_producer_class):
        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_future.add_callback.return_value = mock_future
        mock_future.add_errback.return_value = mock_future
        mock_producer.send.return_value = mock_future

        produce_events(mock_producer, num_events=3, events_per_second=1000)

        mock_producer.flush.assert_called()
