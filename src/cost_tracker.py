import csv
import os
from datetime import datetime


# Recruiter question: "Why build a cost tracker?"
# Answer: In production, LLM API costs scale with usage.
# At 10,000 anomaly explanations per day GPT-4o-mini
# costs roughly $15/day. Without tracking you have
# no visibility into spend until the invoice arrives.
# This tracker gives real-time cost awareness —
# the same principle applied to cloud resource monitoring.

COST_LOG_PATH = "data/cost_log.csv"

# GPT-4o-mini pricing — verify at platform.openai.com/docs/pricing
INPUT_COST_PER_TOKEN = 0.150 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.600 / 1_000_000


def log_api_call(pipeline_name: str, prompt_tokens: int, completion_tokens: int):
    """
    Log a single API call with cost breakdown.
    Appends to CSV — simple, readable, no database needed.

    Args:
        pipeline_name: Which pipeline triggered this call
        prompt_tokens: Input tokens used
        completion_tokens: Output tokens generated
    """
    os.makedirs("data", exist_ok=True)

    input_cost = prompt_tokens * INPUT_COST_PER_TOKEN
    output_cost = completion_tokens * OUTPUT_COST_PER_TOKEN
    total_cost = input_cost + output_cost

    file_exists = os.path.exists(COST_LOG_PATH)

    with open(COST_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "pipeline_name", "prompt_tokens",
            "completion_tokens", "total_tokens",
            "input_cost_usd", "output_cost_usd", "total_cost_usd",
        ])

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "pipeline_name": pipeline_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": round(output_cost, 8),
            "total_cost_usd": round(total_cost, 8),
        })


def get_cost_summary() -> dict:
    """
    Read cost log and return summary statistics.
    Used by Streamlit dashboard cost monitor page.

    Returns:
        Dict with total spend, call count, and daily breakdown
    """
    if not os.path.exists(COST_LOG_PATH):
        return {
            "total_cost_usd": 0,
            "total_calls": 0,
            "average_cost_per_call": 0,
            "daily_breakdown": [],
        }

    import pandas as pd
    df = pd.read_csv(COST_LOG_PATH)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby("date")["total_cost_usd"].sum().reset_index()
    daily.columns = ["date", "cost"]
    daily["date"] = daily["date"].astype(str)

    return {
        "total_cost_usd": round(df["total_cost_usd"].sum(), 6),
        "total_calls": len(df),
        "average_cost_per_call": round(df["total_cost_usd"].mean(), 6),
        "daily_breakdown": daily.to_dict(orient="records"),
    }
