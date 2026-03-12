"""Unit tests for the ETL monitoring module."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modern-data-pipeline", "src"))

from monitoring import ETLMetrics


class TestETLMetrics:
    def test_record_successful_run(self):
        metrics = ETLMetrics()
        # Should not raise
        metrics.record_run(
            success=True, rows_extracted=1000, rows_loaded=950, duration_seconds=45.5
        )

    def test_record_failed_run(self):
        metrics = ETLMetrics()
        metrics.record_run(success=False, duration_seconds=5.0)

    def test_record_multiple_runs(self):
        metrics = ETLMetrics()
        for i in range(5):
            metrics.record_run(success=(i % 2 == 0), rows_extracted=100, rows_loaded=100)
