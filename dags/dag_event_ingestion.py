from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.event_generator import generate_events

# Recruiter question: "Why SQLite instead of a real database?"
# Answer: SQLite is zero-setup, zero-cost, and sufficient
# for a local development project. In production this
# would be BigQuery — which is exactly what I use
# in my current role. The DAG logic is identical
# regardless of which database sits underneath.

DB_PATH = "data/pipeline_events.db"

default_args = {
    "owner": "raghav",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


def initialize_db():
    """Create tables if they don't exist."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT,
            pipeline_name TEXT,
            event_type TEXT,
            severity TEXT,
            expected_rows INTEGER,
            actual_rows INTEGER,
            error_message TEXT,
            duration_seconds INTEGER,
            affected_table TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pipeline_name TEXT,
            event_type TEXT,
            anomaly_score REAL,
            explanation TEXT,
            root_cause TEXT,
            recommended_action TEXT,
            severity TEXT,
            tokens_used INTEGER
        )
    """)

    conn.commit()
    conn.close()


def ingest_events(**context):
    """
    Generate and store a batch of pipeline events.

    Recruiter question: "Why synthetic events?"
    Answer: In production this reads from a real Kafka topic.
    Synthetic generation demonstrates the full architecture
    without requiring access to production infrastructure.
    """
    initialize_db()
    events = generate_events(total=20, anomaly_rate=0.2)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted = 0
    for event in events:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id, event.timestamp, event.pipeline_name,
                    event.event_type, event.severity, event.expected_rows,
                    event.actual_rows, event.error_message,
                    event.duration_seconds, event.affected_table,
                )
            )
            inserted += 1
        except Exception as e:
            print(f"Failed to insert event {event.event_id}: {e}")

    conn.commit()
    conn.close()
    print(f"Ingested {inserted} new events")


with DAG(
    dag_id="pipeline_event_ingestion",
    default_args=default_args,
    description="Generates and stores synthetic pipeline events every 15 minutes",
    schedule_interval="*/15 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["pipelineiq", "ingestion"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_pipeline_events",
        python_callable=ingest_events,
    )
