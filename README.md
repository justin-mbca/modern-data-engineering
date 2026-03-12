# Modern Data Engineering Portfolio

**Description**  
This repository showcases a portfolio of production-grade modern data engineering solutions, featuring three distinct pipeline implementations with CI/CD, monitoring, and ML explainability. The focus is on creating scalable, efficient, and resilient data workflows for real-world use cases.

---

## Repository Structure

```
modern-data-engineering/
├── modern-data-pipeline/          # Batch ETL with Airflow, Great Expectations & Prometheus
│   ├── src/
│   │   ├── etl.py                 # PySpark ETL: extract → transform → load (Delta Lake)
│   │   ├── data_quality.py        # Great Expectations validation suite
│   │   └── monitoring.py          # Prometheus metrics for ETL runs
│   ├── airflow/
│   │   └── dags/etl_dag.py        # Airflow DAG – daily batch schedule
│   ├── docker-compose.yml         # Prometheus + Grafana + Airflow stack
│   ├── prometheus.yml             # Prometheus scrape configuration
│   └── requirements.txt
├── processing/                    # Real-Time Pipeline (Kafka + Spark Streaming)
│   ├── kafka_producer.py          # Multi-event Kafka producer (user_action, order, page_view)
│   ├── spark_consumer.py          # Spark Structured Streaming consumer → Delta Lake
│   ├── kafka_consumer.py          # Python consumer with retry, DLQ, and error handling
│   └── requirements.txt
├── data_ingestion/                # Healthcare data preprocessing
│   ├── health_preprocessing.py   # Imputation, outlier capping, normalisation, encoding
│   └── requirements.txt
├── ml/                            # AI Healthcare ML Platform
│   ├── feature_store.py           # Versioned file-based feature store
│   ├── risk_classification.py     # GBT risk classifier with SHAP explainability
│   └── requirements.txt
├── analytics/                     # Azure integrations
│   └── azure_integration.py       # ADF pipeline trigger, Synapse load, ADLS upload
├── storage/
│   └── delta_table/               # Delta Lake output (runtime)
├── tests/                         # Unit tests for all components
│   ├── test_health_preprocessing.py
│   ├── test_feature_store.py
│   ├── test_kafka_producer.py
│   ├── test_kafka_consumer.py
│   └── test_monitoring.py
├── .github/workflows/ci.yml       # GitHub Actions CI/CD pipeline
└── requirements.txt               # Root dev tooling (pytest, black, flake8)
```

---

## Pipelines

### 1. Modern Data Pipeline

Automated daily batch ETL pipeline backed by Apache Airflow, validated by Great Expectations, and instrumented with Prometheus metrics visualised in Grafana.

| Component | Technology |
|-----------|-----------|
| Compute | PySpark 3.5 |
| Storage | Delta Lake |
| Orchestration | Apache Airflow 2.8 |
| Data Quality | Great Expectations |
| Monitoring | Prometheus + Grafana |

**Walkthrough:**

```bash
# 1. Start the monitoring + Airflow stack
cd modern-data-pipeline
docker compose up -d

# 2. Open the Airflow UI and trigger the DAG
#    http://localhost:8080  (user: airflow / pass: airflow)

# 3. View metrics in Grafana
#    http://localhost:3000  (user: admin / pass: admin)

# 4. Run the ETL manually
python src/etl.py --source /data/raw --target /data/delta/output
```

The Airflow DAG (`modern_data_pipeline_daily_etl`) runs at midnight UTC and performs two tasks:
1. `validate_source_data` – Great Expectations checks for null values and row count.
2. `run_etl_pipeline` – PySpark ETL writes results to Delta Lake and records Prometheus metrics.

---

### 2. Real-Time Data Pipeline

Event-driven streaming pipeline consuming multiple Kafka topics with Spark Structured Streaming and robust error handling.

| Component | Technology |
|-----------|-----------|
| Message Bus | Apache Kafka |
| Streaming | Spark Structured Streaming |
| Error Handling | Retry + Dead-Letter Queue |
| Downstream | RabbitMQ / Azure Service Bus |

**Event types produced:**
- `user_action` → topic `user-actions` (login, click, search, …)
- `order` → topic `orders` (placed, confirmed, shipped, …)
- `page_view` → topic `page-views` (page URL, duration)

**Walkthrough:**

```bash
cd processing

# Start Kafka (assumes a running broker at localhost:9092)
# Produce mixed events at 50 events/second
python kafka_producer.py --brokers localhost:9092 --num-events 5000 --rate 50

# Start the Spark consumer (writes to Delta Lake)
python spark_consumer.py --brokers localhost:9092 --output /data/delta/realtime

# Or run the lightweight Python consumer with DLQ support
python kafka_consumer.py --brokers localhost:9092 --topics user-actions,orders,page-views
```

The Python consumer (`kafka_consumer.py`) implements:
- Manual offset commits (at-least-once semantics).
- Exponential back-off retries for transient Kafka errors.
- Dead-Letter Queue routing for unprocessable messages.

---

### 3. AI Healthcare Data Platform

End-to-end ML platform for patient risk classification with explainability, a versioned feature store, and Azure cloud integrations.

| Component | Technology |
|-----------|-----------|
| Preprocessing | pandas + scikit-learn |
| Feature Store | File-based versioned store |
| ML Model | GradientBoostingClassifier |
| Explainability | SHAP TreeExplainer |
| Cloud | Azure Data Factory, Synapse Analytics, ADLS |

**Walkthrough:**

```python
import pandas as pd
from data_ingestion.health_preprocessing import preprocess_health_data
from ml.feature_store import FeatureStore
from ml.risk_classification import run_risk_classification_pipeline

# 1. Preprocess raw health data
raw = pd.read_csv("data/patients.csv")
result = preprocess_health_data(raw, not_null_columns=["patient_id"])
clean_df = result["dataframe"]

# 2. Save processed features to the versioned store
store = FeatureStore(store_dir="/data/feature_store")
store.save("patient_risk_features", clean_df, version=1, description="v1 – baseline features")

# 3. Load features and train risk classifier with SHAP explanations
features = store.load("patient_risk_features", version=1)
pipeline_output = run_risk_classification_pipeline(
    features, target_column="risk_level", output_dir="/data/shap_plots"
)
print(pipeline_output["metrics"]["classification_report"])
print("Sample local explanation:", pipeline_output["sample_explanation"])
```

SHAP summary plots are saved to the `output_dir` for inclusion in reports or dashboards.

**Azure Integration:**

```python
import os
os.environ["AZURE_TENANT_ID"] = "..."
os.environ["AZURE_CLIENT_ID"] = "..."
os.environ["AZURE_CLIENT_SECRET"] = "..."

from analytics.azure_integration import trigger_adf_pipeline, load_dataframe_to_synapse

# Trigger ADF ingestion pipeline
run = trigger_adf_pipeline("IngestHealthData", parameters={"source_date": "2024-01-01"})

# Load processed features to Synapse Analytics
load_dataframe_to_synapse(clean_df, table_name="patient_risk_features")
```

---

## CI/CD

GitHub Actions automatically runs on every push or pull request to `main`:

| Job | Description |
|-----|-------------|
| `lint` | Black format check + Flake8 linting |
| `test-data-ingestion` | Unit tests for health preprocessing |
| `test-ml` | Unit tests for feature store |
| `test-realtime` | Unit tests for Kafka producer / consumer |
| `test-monitoring` | Unit tests for Prometheus metrics module |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Prometheus/Grafana/Airflow)
- Apache Kafka (for real-time pipeline)
- Java 11+ (for PySpark)

### Install dependencies

```bash
# Root dev tooling
pip install -r requirements.txt

# Per-pipeline dependencies
pip install -r modern-data-pipeline/requirements.txt
pip install -r processing/requirements.txt
pip install -r data_ingestion/requirements.txt
pip install -r ml/requirements.txt
```

### Run all tests

```bash
pytest tests/ -v --tb=short
```

---

## Contributions

Feel free to fork this repository and submit pull requests. All contributions are welcome!

## License

This project is licensed under the MIT License – see the `LICENSE` file for details.