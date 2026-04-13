import streamlit as st
import pandas as pd
import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cost_tracker import get_cost_summary
from src.rag_pipeline import get_chroma_collection, get_embedding

st.set_page_config(page_title="PipelineIQ", page_icon="🔍", layout="wide")

page = st.sidebar.selectbox(
    "Navigate",
    ["🔴 Live Monitor", "🔍 Incident Search", "💰 Cost Monitor"]
)

DB_PATH = "data/pipeline_events.db"


# ══════════════════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════
if page == "🔴 Live Monitor":
    st.title("🔴 PipelineIQ — Live Monitor")
    st.caption("Real-time pipeline health and anomaly explanations")

    if not os.path.exists(DB_PATH):
        st.warning("No pipeline data yet. Run the Airflow DAGs to start ingesting events.")
        st.stop()

    conn = sqlite3.connect(DB_PATH)

    st.subheader("Pipeline Health")
    events_df = pd.read_sql("""
        SELECT pipeline_name, event_type, severity, timestamp
        FROM events ORDER BY timestamp DESC LIMIT 100
    """, conn)

    if events_df.empty:
        st.info("No events recorded yet.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total = len(events_df)
        failures = len(events_df[events_df["event_type"] == "FAILURE"])
        sla_breaches = len(events_df[events_df["event_type"] == "SLA_BREACH"])
        success_rate = len(events_df[events_df["event_type"] == "SUCCESS"]) / total

        col1.metric("Total Events", total)
        col2.metric("Failures", failures, delta=f"-{failures}" if failures > 0 else "0", delta_color="inverse")
        col3.metric("SLA Breaches", sla_breaches)
        col4.metric("Success Rate", f"{success_rate:.0%}")

        st.subheader("Recent Events")

        def color_severity(val):
            colors = {"HIGH": "background-color: #ffcccc", "MEDIUM": "background-color: #fff3cc", "LOW": "background-color: #ccffcc"}
            return colors.get(val, "")

        styled = events_df.head(20).style.map(color_severity, subset=["severity"])
        st.dataframe(styled, use_container_width=True)

    st.subheader("Anomaly Explanations")
    try:
        explanations_df = pd.read_sql("""
            SELECT pipeline_name, explanation, root_cause,
                   recommended_action, severity, timestamp
            FROM explanations ORDER BY timestamp DESC LIMIT 10
        """, conn)

        if explanations_df.empty:
            st.info("No anomalies detected yet.")
        else:
            for _, row in explanations_df.iterrows():
                icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(row.get("severity", ""), "⚪")
                with st.expander(f"{icon} {row['pipeline_name']} — {row['timestamp']}"):
                    st.markdown(f"**Explanation:** {row['explanation']}")
                    st.markdown(f"**Root Cause:** {row['root_cause']}")
                    st.markdown(f"**Recommended Action:** {row['recommended_action']}")
                    st.markdown(f"**Severity:** {row['severity']}")
    except Exception:
        st.info("No anomaly explanations yet.")

    conn.close()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — INCIDENT SEARCH
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Incident Search":
    st.title("🔍 Incident Search")
    st.caption("Search historical incidents using natural language")

    query = st.text_input(
        "Describe the issue you're investigating",
        placeholder="e.g. memory error causing pipeline failure"
    )

    if query:
        with st.spinner("Searching similar incidents..."):
            try:
                collection = get_chroma_collection()
                if collection.count() == 0:
                    st.warning("No runbooks loaded yet. Run the RAG pipeline first.")
                else:
                    query_embedding = get_embedding(query)
                    results = collection.query(query_embeddings=[query_embedding], n_results=3)
                    st.subheader("Most Similar Historical Incidents")
                    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                        with st.expander(f"Incident {i+1} — {metadata.get('incident_type', 'Unknown')}"):
                            st.markdown(f"**Description:** {doc}")
                            st.markdown(f"**Severity:** {metadata.get('severity', 'Unknown')}")
                            st.markdown(f"**Resolution Time:** {metadata.get('resolution_time', 'Unknown')}")
            except Exception as e:
                st.error(f"Search error: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — COST MONITOR
# ══════════════════════════════════════════════════════════════
elif page == "💰 Cost Monitor":
    st.title("💰 API Cost Monitor")
    st.caption("Real-time OpenAI API spend tracking")

    summary = get_cost_summary()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spend", f"${summary['total_cost_usd']:.4f}")
    col2.metric("Total API Calls", summary["total_calls"])
    col3.metric("Avg Cost per Call", f"${summary['average_cost_per_call']:.6f}")

    if summary["daily_breakdown"]:
        st.subheader("Daily Spend")
        daily_df = pd.DataFrame(summary["daily_breakdown"])
        st.bar_chart(daily_df.set_index("date")["cost"])
    else:
        st.info("No API calls logged yet.")
