from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sqlite3
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.anomaly_detector import detect_anomalies

DB_PATH = "data/pipeline_events.db"

default_args = {
    "owner": "raghav",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


def run_anomaly_detection(**context):
    """
    Load recent events and run anomaly detection.
    Pushes detected anomalies to XCom.

    Recruiter question: "What is XCom?"
    Answer: Airflow's mechanism for passing small amounts
    of data between tasks without writing to a database.
    Keeps the pipeline clean and fast.
    """
    if not os.path.exists(DB_PATH):
        print("No database yet. Skipping detection.")
        return []

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM events ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()

    if df.empty:
        print("No events to analyze.")
        return []

    anomalies = detect_anomalies(df)
    print(f"Detected {len(anomalies)} anomalies")

    anomaly_dicts = [{
        "pipeline_name": a.pipeline_name,
        "event_type": a.event_type,
        "severity": a.severity,
        "error_message": a.error_message,
        "expected_rows": a.expected_rows,
        "actual_rows": a.actual_rows,
        "anomaly_score": a.anomaly_score,
        "is_anomaly": a.is_anomaly,
        "anomaly_reason": a.anomaly_reason,
        "timestamp": a.timestamp,
    } for a in anomalies]

    context["ti"].xcom_push(key="anomalies", value=anomaly_dicts)
    return anomaly_dicts


with DAG(
    dag_id="pipeline_anomaly_detection",
    default_args=default_args,
    description="Detects anomalies in pipeline events every 30 minutes",
    schedule_interval="*/30 * * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["pipelineiq", "detection"],
) as dag:

    detect_task = PythonOperator(
        task_id="detect_anomalies",
        python_callable=run_anomaly_detection,
    )

    # Recruiter question: "Why separate detection and explanation into different DAGs?"
    # Answer: Separation of concerns and cost control. Detection is cheap
    # and runs frequently. Explanation calls the OpenAI API and costs money.
    # Keeping them separate means API calls only happen when anomalies
    # are actually detected — not on every detection run.
    trigger_task = TriggerDagRunOperator(
        task_id="trigger_explanation_dag",
        trigger_dag_id="pipeline_anomaly_explanation",
        conf={"anomalies": "{{ ti.xcom_pull(task_ids='detect_anomalies', key='anomalies') }}"},
        wait_for_completion=False,
    )

    detect_task >> trigger_task
