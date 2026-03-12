"""
Prometheus Metrics for the Modern Data Pipeline.

Exposes ETL run statistics via a Prometheus HTTP endpoint so that
Grafana (or any Prometheus-compatible scraper) can collect and visualise them.
"""

import logging
import threading

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
    REGISTRY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

ETL_RUNS_TOTAL = Counter(
    "etl_runs_total",
    "Total number of ETL pipeline runs",
    ["status"],  # labels: 'success' | 'failure'
)

ETL_ROWS_EXTRACTED = Counter(
    "etl_rows_extracted_total",
    "Cumulative number of rows extracted by the ETL pipeline",
)

ETL_ROWS_LOADED = Counter(
    "etl_rows_loaded_total",
    "Cumulative number of rows written to Delta Lake",
)

ETL_DURATION_SECONDS = Histogram(
    "etl_duration_seconds",
    "Wall-clock time (seconds) for a complete ETL run",
    buckets=[30, 60, 120, 300, 600, 1800, 3600],
)

ETL_LAST_RUN_TIMESTAMP = Gauge(
    "etl_last_run_timestamp_seconds",
    "Unix timestamp of the most recent ETL run (successful or not)",
)


class ETLMetrics:
    """Thin wrapper that records ETL run statistics into Prometheus metrics."""

    def record_run(
        self,
        success: bool,
        rows_extracted: int = 0,
        rows_loaded: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        """
        Record the outcome of a single ETL run.

        Args:
            success: Whether the run completed without errors.
            rows_extracted: Number of rows read from the source.
            rows_loaded: Number of rows written to Delta Lake.
            duration_seconds: Wall-clock duration of the run.
        """
        import time

        status = "success" if success else "failure"
        ETL_RUNS_TOTAL.labels(status=status).inc()
        ETL_DURATION_SECONDS.observe(duration_seconds)
        ETL_LAST_RUN_TIMESTAMP.set(time.time())

        if success:
            ETL_ROWS_EXTRACTED.inc(rows_extracted)
            ETL_ROWS_LOADED.inc(rows_loaded)

        logger.info(
            "Metrics recorded: status=%s rows_extracted=%d rows_loaded=%d duration=%.2fs",
            status,
            rows_extracted,
            rows_loaded,
            duration_seconds,
        )


def start_metrics_server(port: int = 8000) -> None:
    """
    Start the Prometheus metrics HTTP server in a background daemon thread.

    Args:
        port: TCP port for the /metrics endpoint (default 8000).
    """
    logger.info("Starting Prometheus metrics server on port %d", port)
    thread = threading.Thread(target=start_http_server, args=(port,), daemon=True)
    thread.start()
    logger.info("Metrics server started – scrape at http://localhost:%d/metrics", port)
