# PipelineIQ

An LLM-powered data pipeline monitoring system that automatically detects anomalies in data pipelines and generates plain-English root cause explanations grounded in historical incident context using RAG architecture.

---

## The Problem It Solves

Data engineers spend significant time manually investigating pipeline failures - reading logs, searching through past incidents, and diagnosing root causes. PipelineIQ automates this by combining statistical anomaly detection with LLM-powered explanation generation, grounded in your team's own historical incident runbooks.

---

## Architecture

```
Synthetic Event Generator
         |
   SQLite Storage
         |
Airflow DAG 1 (every 15 min)
   Event Ingestion
         |
Airflow DAG 2 (every 30 min)
   Anomaly Detection
   (Z-score, failure rate, SLA breach frequency)
         | (if anomaly detected)
Airflow DAG 3 (triggered)
   RAG Explanation Pipeline
         |
   ChromaDB <-- Historical Runbooks
         |
   OpenAI GPT-4o-mini
         |
   Streamlit Dashboard
   (Live Monitor | Incident Search | Cost Tracker)
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.9+ | Production standard for data engineering |
| LLM | OpenAI GPT-4o-mini | Cost-efficient, sufficient for structured explanation tasks |
| Vector Store | ChromaDB | Zero-cost local vector DB, same API as Pinecone for production swap |
| RAG Framework | LangChain | Industry standard for RAG pipeline construction |
| Orchestration | Apache Airflow | Production orchestration - mirrors real work stack |
| Dashboard | Streamlit | Rapid internal tooling - already used in production |
| Storage | SQLite | Zero-setup local DB - BigQuery in production |
| Cost Tracking | tiktoken | Pre-call token counting for cost awareness |

---

## Technical Decisions

**Why statistical anomaly detection instead of ML-based?**
Three reasons: explainability (I can tell exactly why something was flagged), no training data required (works from day one), and lower latency and cost (no model inference). ML-based detection would make sense if patterns were more complex or subtle.

**Why ChromaDB instead of Pinecone?**
ChromaDB runs locally at zero cost with zero setup - keeping the entire project under $10. Pinecone makes sense in production when you need a managed, highly available vector store at scale. The API is similar enough that swapping ChromaDB for Pinecone in production is a configuration change, not an architectural one.

**Why GPT-4o-mini instead of GPT-4o?**
Anomaly explanations require consistent, structured output - not creative reasoning. GPT-4o-mini handles structured explanation tasks at roughly 10x lower cost than GPT-4o. Temperature is set to 0.2 to ensure deterministic, repeatable explanations.

**Why separate anomaly detection and explanation into different DAGs?**
Separation of concerns and cost control. Detection is cheap and runs frequently. Explanation calls the OpenAI API and costs money. Keeping them separate means API calls only happen when anomalies are actually detected.

**Why temperature 0.2?**
Pipeline anomaly explanations need to be consistent. If the same error fires twice I want the same explanation both times - not a creative variation. Low temperature keeps the LLM focused on facts.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/RaghavaBoppudi/pipeline-anomaly-llm.git
cd pipeline-anomaly-llm

# Create virtual environment (Python 3.9-3.12 recommended for Airflow)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here

# Generate initial events
python3 src/event_generator.py

# Run anomaly detection
python3 src/anomaly_detector.py

# Test the full RAG pipeline
python3 -m src.rag_pipeline

# Launch dashboard
streamlit run dashboard/app.py

# Run Airflow (Python 3.9-3.12)
airflow standalone
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| OpenAI Embeddings (text-embedding-3-small) | ~$0.50 across full development |
| GPT-4o-mini explanations | ~$2.00 across full development |
| ChromaDB | Free (local) |
| Airflow | Free (local) |
| **Total** | **~$2.50 - $5.00** |

---

## What Would Change At Production Scale

| Current (Dev) | Production |
|---------------|------------|
| SQLite | BigQuery |
| Local ChromaDB | Pinecone or Weaviate |
| Local Airflow | Cloud Composer on GCP |
| Synthetic events | Real Kafka event streams |
| Static runbooks | Auto-ingested from PagerDuty + Confluence |

---

## Connection To Production Experience

Built on the same principles used in daily production work - Airflow orchestration, SLA monitoring, data quality validation, and cost-aware pipeline design. The LLM layer is what this project adds on top of that foundation. In production this would sit on top of real Kafka streams and BigQuery, with runbooks auto-populated from PagerDuty and Confluence rather than manually written.

---

## Notes

- Airflow orchestration tested with Python 3.9-3.12. Python 3.14 has a known state-tracking conflict with Airflow 3.x that does not affect DAG logic execution.
- All other components run correctly on Python 3.14.

---

## Author

Raghava Boppudi - Data Engineer
[LinkedIn](https://linkedin.com/in/raghavaboppudi) | [GitHub](https://github.com/RaghavaBoppudi)