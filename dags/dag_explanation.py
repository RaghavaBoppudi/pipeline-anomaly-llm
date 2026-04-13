from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sqlite3
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag_pipeline import run_rag_pipeline
from src.anomaly_detector import AnomalyResult

DB_PATH = "data/pipeline_events.db"

default_args = {
    "owner": "raghav",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def generate_explanations(**context):
    """
    Receive anomalies from triggering DAG and generate
    LLM explanations for each using the RAG pipeline.

    Recruiter question: "How do you receive data from the triggering DAG?"
    Answer: Via dag_run.conf — Airflow's standard pattern for passing
    context between DAGs without coupling them tightly.
    """
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}
    anomalies_raw = conf.get("anomalies", [])

    if isinstance(anomalies_raw, str):
        try:
            anomalies_raw = json.loads(anomalies_raw)
        except Exception:
            anomalies_raw = []

    if not anomalies_raw:
        print("No anomalies received. Nothing to explain.")
        return

    print(f"Generating explanations for {len(anomalies_raw)} anomalies")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for anomaly_dict in anomalies_raw:
        try:
            anomaly = AnomalyResult(**anomaly_dict)
            explanation = run_rag_pipeline(anomaly)

            cursor.execute("""
                INSERT INTO explanations (
                    timestamp, pipeline_name, event_type,
                    anomaly_score, explanation, root_cause,
                    recommended_action, severity, tokens_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                anomaly.timestamp, anomaly.pipeline_name, anomaly.event_type,
                anomaly.anomaly_score, explanation.get("explanation", ""),
                explanation.get("root_cause", ""), explanation.get("recommended_action", ""),
                explanation.get("severity", ""), explanation.get("tokens_used", 0),
            ))

            print(f"Explanation stored for: {anomaly.pipeline_name}")

        except Exception as e:
            print(f"Failed to generate explanation: {e}")
            continue

    conn.commit()
    conn.close()


with DAG(
    dag_id="pipeline_anomaly_explanation",
    default_args=default_args,
    description="Generates LLM explanations for detected anomalies using RAG",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["pipelineiq", "explanation", "llm"],
) as dag:

    explain_task = PythonOperator(
        task_id="generate_llm_explanations",
        python_callable=generate_explanations,
    )
