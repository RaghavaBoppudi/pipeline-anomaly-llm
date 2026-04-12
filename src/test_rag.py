from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = OpenAI()

# ── Step 1: Set up ChromaDB with historical incidents ──────────
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="pipeline_incidents")

historical_incidents = [
    {
        "text": "Memory threshold exceeded in ingestion pipeline — resolved by increasing Airflow task memory allocation from 512MB to 2GB",
        "severity": "high",
        "resolution_time": "45 mins"
    },
    {
        "text": "DAG crashed due to out of memory error — root cause was upstream table growing 3x overnight, fixed by adding incremental load",
        "severity": "high",
        "resolution_time": "2 hours"
    },
    {
        "text": "Kafka consumer lag spiked on transactions topic — resolved by scaling up consumer group from 2 to 6 partitions",
        "severity": "medium",
        "resolution_time": "30 mins"
    },
    {
        "text": "Schema drift detected — source team added 3 new columns without notice, fixed by updating schema validation config",
        "severity": "medium",
        "resolution_time": "1 hour"
    },
    {
        "text": "SLA breach on morning ETL — upstream API was throttling requests, fixed by adding retry logic with exponential backoff",
        "severity": "high",
        "resolution_time": "3 hours"
    },
]

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Store all incidents in ChromaDB
print("Loading historical incidents into ChromaDB...")
for i, incident in enumerate(historical_incidents):
    embedding = get_embedding(incident["text"])
    collection.add(
        ids=[f"incident_{i}"],
        embeddings=[embedding],
        documents=[incident["text"]],
        metadatas=[{
            "severity": incident["severity"],
            "resolution_time": incident["resolution_time"]
        }]
    )

print(f"Stored {collection.count()} incidents")
print()

# ── Step 2: New anomaly comes in ───────────────────────────────
new_anomaly = {
    "pipeline_name": "customer_transactions_ingestion",
    "event_type": "FAILURE",
    "error_message": "java.lang.OutOfMemoryError: Java heap space",
    "expected_rows": 500000,
    "actual_rows": 0
}

print(f"New anomaly detected: {new_anomaly['pipeline_name']}")
print(f"Error: {new_anomaly['error_message']}")
print()

# ── Step 3: Retrieve similar historical incidents ──────────────
anomaly_description = f"{new_anomaly['pipeline_name']} failed with error: {new_anomaly['error_message']}"
anomaly_embedding = get_embedding(anomaly_description)

results = collection.query(
    query_embeddings=[anomaly_embedding],
    n_results=3
)

# Format retrieved incidents for LLM context
retrieved_context = ""
for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    retrieved_context += f"Historical Incident {i+1}:\n"
    retrieved_context += f"Description: {doc}\n"
    retrieved_context += f"Severity: {metadata['severity']}\n"
    retrieved_context += f"Resolution Time: {metadata['resolution_time']}\n\n"

print("Retrieved similar historical incidents:")
print(retrieved_context)

# ── Step 4: Generate grounded explanation using LLM ───────────
system_prompt = """You are a senior data engineering expert analyzing pipeline anomalies.
You will be given details about a new pipeline failure and similar historical incidents.
Your job is to provide:
1. A plain English explanation of what likely went wrong
2. The most probable root cause
3. A specific recommended action based on historical resolutions
4. A severity assessment

Be specific and technical. Do not give generic answers.
Base your response on the historical incidents provided."""

user_prompt = f"""New Pipeline Anomaly:
Pipeline: {new_anomaly['pipeline_name']}
Error: {new_anomaly['error_message']}
Expected rows: {new_anomaly['expected_rows']}
Actual rows: {new_anomaly['actual_rows']}

Similar Historical Incidents:
{retrieved_context}

Provide your analysis in exactly this format:
EXPLANATION: [what happened in plain English]
ROOT CAUSE: [most probable cause]
RECOMMENDED ACTION: [specific next step]
SEVERITY: [LOW/MEDIUM/HIGH and why]"""

print("Generating LLM explanation...")
print()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.2  # Low temperature for consistent, reliable explanations
)

explanation = response.choices[0].message.content
finish_reason = response.choices[0].finish_reason
tokens_used = response.usage.total_tokens

print("=" * 60)
print("LLM GENERATED EXPLANATION:")
print("=" * 60)
print(explanation)
print()
print(f"Finish reason: {finish_reason}")
print(f"Tokens used: {tokens_used}")
