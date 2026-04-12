import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Initialize ChromaDB — runs locally, no setup needed
chroma_client = chromadb.Client()

# Create a collection — think of this like a table in a database
collection = chroma_client.create_collection(name="pipeline_incidents")

# Some historical incident descriptions
incidents = [
    "Pipeline failed due to memory error in the ingestion job",
    "DAG crashed because it ran out of RAM during transformation",
    "Kafka consumer lag detected on transactions topic",
    "Schema drift detected in source table — new column added",
    "SLA breach — morning ETL job did not complete by 6AM",
    "Out of memory exception in data processing job",
    "Unexpected null values detected in customer ID field",
    "Pipeline succeeded but row count lower than expected threshold",
]

# Generate embeddings for each incident
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

print("Generating embeddings and storing in ChromaDB...")

# Store each incident with its embedding in ChromaDB
for i, incident in enumerate(incidents):
    embedding = get_embedding(incident)
    collection.add(
        ids=[f"incident_{i}"],
        embeddings=[embedding],
        documents=[incident],
        metadatas=[{"incident_id": i, "severity": "high" if i % 2 == 0 else "medium"}]
    )

print(f"Stored {collection.count()} incidents in ChromaDB")
print()

# Now search for similar incidents to a new anomaly
new_anomaly = "Memory threshold exceeded causing job failure"
new_embedding = get_embedding(new_anomaly)

results = collection.query(
    query_embeddings=[new_embedding],
    n_results=3
)

print(f"New anomaly: {new_anomaly}")
print()
print("Top 3 most similar historical incidents:")
for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"{i+1}. {doc}")
    print(f"   Incident ID: {metadata['incident_id']}")
    print(f"   Severity: {metadata['severity']}")
    print()
