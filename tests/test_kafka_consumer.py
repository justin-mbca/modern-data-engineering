"""Unit tests for the Kafka consumer module."""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "processing"))

from kafka_consumer import process_message, retry_with_backoff, send_to_dlq


class TestProcessMessage:
    def test_valid_message_returns_processed(self):
        msg = {
            "event_id": "abc-123",
            "event_type": "user_action",
            "timestamp": "2024-01-01T00:00:00Z",
            "action": "login",
        }
        result = process_message(msg)
        assert result["processed"] is True

    def test_missing_field_raises_value_error(self):
        msg = {"event_type": "order"}  # missing event_id and timestamp
        with pytest.raises(ValueError, match="missing required fields"):
            process_message(msg)

    def test_original_fields_preserved(self):
        msg = {
            "event_id": "xyz",
            "event_type": "page_view",
            "timestamp": "2024-06-01T12:00:00Z",
            "page": "/home",
        }
        result = process_message(msg)
        assert result["page"] == "/home"


class TestRetryWithBackoff:
    def test_succeeds_on_first_attempt(self):
        func = MagicMock(return_value="ok")
        result = retry_with_backoff(func, max_retries=3)
        assert result == "ok"
        assert func.call_count == 1

    def test_retries_on_transient_failure(self):
        func = MagicMock(side_effect=[RuntimeError("fail"), RuntimeError("fail"), "ok"])
        result = retry_with_backoff(func, max_retries=3)
        assert result == "ok"
        assert func.call_count == 3

    def test_raises_after_max_retries(self):
        func = MagicMock(side_effect=RuntimeError("always fails"))
        with pytest.raises(RuntimeError, match="always fails"):
            retry_with_backoff(func, max_retries=3)

    def test_backoff_sleep_is_called(self):
        func = MagicMock(side_effect=[RuntimeError("x"), "ok"])
        with patch("kafka_consumer.time.sleep") as mock_sleep:
            retry_with_backoff(func, max_retries=2)
        mock_sleep.assert_called_once()


class TestDeadLetterQueue:
    def test_send_to_dlq_does_not_raise(self, caplog):
        msg = {"bad": "data"}
        exc = ValueError("test error")
        send_to_dlq(msg, exc)  # should not raise
