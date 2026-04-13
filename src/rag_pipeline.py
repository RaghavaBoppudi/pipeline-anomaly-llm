from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import os
from src.anomaly_detector import AnomalyResult
from src.cost_tracker import log_api_call

load_dotenv()

client = OpenAI()

# Recruiter question: "Why persistent storage for ChromaDB?"
# Answer: Without persistence ChromaDB resets every time
# the script runs. Persistent storage means historical
# runbooks are embedded once and reused forever —
# not re-embedded on every run which would waste API cost.

CHROMA_PATH = "data/chroma_store"


def get_chroma_collection():
    """Get or create persistent ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="pipeline_incidents")
    return collection


def get_embedding(text: str) -> list:
    """Generate embedding using cheapest OpenAI model."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def load_runbooks():
    """
    Load and embed historical incident runbooks into ChromaDB.
    Only runs if collection is empty — prevents re-embedding.

    Recruiter question: "How do you avoid re-embedding runbooks on every run?"
    Answer: Check collection count first. If runbooks are already
    stored, skip the embedding step entirely. This saves API cost
    and speeds up startup time.
    """
    collection = get_chroma_collection()

    if collection.count() > 0:
        print(f"ChromaDB already has {collection.count()} runbooks. Skipping embedding.")
        return collection

    print("Embedding runbooks into ChromaDB for first time...")

    runbooks = [
        {
            "id": "rb_001",
            "text": "Memory threshold exceeded in ingestion pipeline — root cause was upstream table growing 3x. Resolved by increasing Airflow task memory from 512MB to 2GB and adding incremental load pattern.",
            "incident_type": "FAILURE", "severity": "high", "resolution_time": "2 hours",
        },
        {
            "id": "rb_002",
            "text": "Kafka consumer lag spiked on transactions topic — consumer group fell behind due to upstream volume spike. Resolved by scaling consumer group from 2 to 6 partitions.",
            "incident_type": "FAILURE", "severity": "medium", "resolution_time": "30 mins",
        },
        {
            "id": "rb_003",
            "text": "Schema drift detected in source table — source team added 3 new columns without notice. Resolved by updating schema validation config and adding schema change alerting.",
            "incident_type": "SCHEMA_DRIFT", "severity": "medium", "resolution_time": "1 hour",
        },
        {
            "id": "rb_004",
            "text": "SLA breach on morning ETL — upstream API was throttling requests causing delays. Resolved by adding retry logic with exponential backoff and adjusting batch size.",
            "incident_type": "SLA_BREACH", "severity": "high", "resolution_time": "3 hours",
        },
        {
            "id": "rb_005",
            "text": "Volume anomaly detected — row count 45% below average due to source system maintenance window not communicated. Resolved by adding source system health check before pipeline start.",
            "incident_type": "VOLUME_ANOMALY", "severity": "medium", "resolution_time": "45 mins",
        },
        {
            "id": "rb_006",
            "text": "Out of memory error in data processing job — JVM heap space exhausted during large join operation. Resolved by increasing executor memory and switching to broadcast join for smaller table.",
            "incident_type": "FAILURE", "severity": "high", "resolution_time": "1.5 hours",
        },
        {
            "id": "rb_007",
            "text": "Duplicate records detected in customer table — upstream system sent duplicate events during retry storm. Resolved by adding deduplication step using event_id as unique key.",
            "incident_type": "VOLUME_ANOMALY", "severity": "high", "resolution_time": "2 hours",
        },
        {
            "id": "rb_008",
            "text": "Connection timeout in ingestion pipeline — database connection pool exhausted during peak load. Resolved by increasing connection pool size from 10 to 50 and adding connection timeout handling.",
            "incident_type": "FAILURE", "severity": "high", "resolution_time": "1 hour",
        },
        {
            "id": "rb_009",
            "text": "Zero records ingested from payment gateway — API authentication token expired silently. Resolved by adding token expiry monitoring and automatic refresh logic.",
            "incident_type": "FAILURE", "severity": "high", "resolution_time": "45 mins",
        },
        {
            "id": "rb_010",
            "text": "SLA breach on risk aggregation pipeline — upstream fraud detection model taking 3x longer than usual. Resolved by adding timeout on model inference call and falling back to rule-based scoring.",
            "incident_type": "SLA_BREACH", "severity": "high", "resolution_time": "2 hours",
        },
    ]

    for runbook in runbooks:
        embedding = get_embedding(runbook["text"])
        collection.add(
            ids=[runbook["id"]],
            embeddings=[embedding],
            documents=[runbook["text"]],
            metadatas=[{
                "incident_type": runbook["incident_type"],
                "severity": runbook["severity"],
                "resolution_time": runbook["resolution_time"],
            }]
        )

    print(f"Embedded {collection.count()} runbooks into ChromaDB")
    return collection


def retrieve_similar_incidents(anomaly: AnomalyResult, collection, n_results: int = 3) -> str:
    """
    Retrieve most similar historical incidents from ChromaDB.

    Args:
        anomaly: The detected anomaly to find similar incidents for
        collection: ChromaDB collection
        n_results: Number of similar incidents to retrieve

    Returns:
        Formatted string of similar incidents for LLM context
    """
    query = f"{anomaly.pipeline_name} {anomaly.event_type} {anomaly.error_message} {anomaly.anomaly_reason}"
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    context = ""
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        context += f"Historical Incident {i+1}:\n"
        context += f"Description: {doc}\n"
        context += f"Type: {metadata['incident_type']}\n"
        context += f"Severity: {metadata['severity']}\n"
        context += f"Resolution Time: {metadata['resolution_time']}\n\n"

    return context


def generate_explanation(anomaly: AnomalyResult, historical_context: str) -> dict:
    """
    Generate LLM explanation grounded in historical context.

    Recruiter question: "Why temperature 0.2?"
    Answer: Anomaly explanations need to be consistent and
    deterministic. If the same error fires twice I want the
    same explanation both times. Low temperature keeps the LLM
    focused on facts rather than generating creative variations.

    Args:
        anomaly: Detected anomaly details
        historical_context: Retrieved similar incidents

    Returns:
        Structured explanation dict
    """
    system_prompt = """You are a senior data engineering expert
analyzing pipeline anomalies for a fintech company.
You will be given details about a new pipeline failure
and similar historical incidents from our runbook.
Provide specific, actionable analysis based on the
historical context provided. Do not give generic answers.
Always base recommendations on the historical resolutions."""

    user_prompt = f"""New Pipeline Anomaly Detected:
Pipeline: {anomaly.pipeline_name}
Event Type: {anomaly.event_type}
Severity: {anomaly.severity}
Error: {anomaly.error_message}
Expected Rows: {anomaly.expected_rows:,}
Actual Rows: {anomaly.actual_rows:,}
Anomaly Reason: {anomaly.anomaly_reason}
Anomaly Score: {anomaly.anomaly_score}

Similar Historical Incidents from Runbook:
{historical_context}

Respond in exactly this format:
EXPLANATION: [plain English explanation of what happened]
ROOT CAUSE: [most probable root cause based on history]
RECOMMENDED ACTION: [specific next step based on historical resolutions]
SEVERITY: [LOW/MEDIUM/HIGH] — [justification]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    tokens_used = response.usage.total_tokens
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    log_api_call(
        pipeline_name=anomaly.pipeline_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    lines = content.strip().split("\n")
    parsed = {
        "explanation": "",
        "root_cause": "",
        "recommended_action": "",
        "severity": "",
        "finish_reason": finish_reason,
        "tokens_used": tokens_used,
        "raw_response": content,
    }

    for line in lines:
        if line.startswith("EXPLANATION:"):
            parsed["explanation"] = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("ROOT CAUSE:"):
            parsed["root_cause"] = line.replace("ROOT CAUSE:", "").strip()
        elif line.startswith("RECOMMENDED ACTION:"):
            parsed["recommended_action"] = line.replace("RECOMMENDED ACTION:", "").strip()
        elif line.startswith("SEVERITY:"):
            parsed["severity"] = line.replace("SEVERITY:", "").strip()

    return parsed


def run_rag_pipeline(anomaly: AnomalyResult) -> dict:
    """
    Full RAG pipeline for a single anomaly.

    Steps:
    1. Load runbooks into ChromaDB (skips if already loaded)
    2. Retrieve similar historical incidents
    3. Generate grounded LLM explanation

    Args:
        anomaly: Detected anomaly to explain

    Returns:
        Complete explanation dict
    """
    collection = load_runbooks()
    historical_context = retrieve_similar_incidents(anomaly, collection)
    explanation = generate_explanation(anomaly, historical_context)
    return explanation


if __name__ == "__main__":
    from datetime import datetime

    test_anomaly = AnomalyResult(
        pipeline_name="customer_transactions_ingestion",
        event_type="FAILURE",
        severity="HIGH",
        error_message="java.lang.OutOfMemoryError: Java heap space",
        expected_rows=500000,
        actual_rows=0,
        anomaly_score=0.85,
        is_anomaly=True,
        anomaly_reason="High failure rate: 60% of recent events failed",
        timestamp=datetime.now().isoformat(),
    )

    result = run_rag_pipeline(test_anomaly)
    print("\n" + "=" * 60)
    print("RAG PIPELINE OUTPUT:")
    print("=" * 60)
    for key, value in result.items():
        if key != "raw_response":
            print(f"{key.upper()}: {value}")
