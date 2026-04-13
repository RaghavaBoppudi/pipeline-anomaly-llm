import pandas as pd
import numpy as np
from dataclasses import dataclass
import os


# Recruiter question: "Why statistical detection over ML-based?"
# Answer: Three reasons:
# 1. Explainability — I can tell exactly why something was flagged
# 2. No training data needed — works from day one
# 3. Lower latency and cost — no model inference required
# ML would make sense if patterns were more complex or subtle


@dataclass
class AnomalyResult:
    """Structured output from anomaly detection."""
    pipeline_name: str
    event_type: str
    severity: str
    error_message: str
    expected_rows: int
    actual_rows: int
    anomaly_score: float
    is_anomaly: bool
    anomaly_reason: str
    timestamp: str


def load_recent_events(filepath: str = "data/events/events.csv", n: int = 50) -> pd.DataFrame:
    """Load the most recent N events from CSV."""
    df = pd.read_csv(filepath)
    df = df.sort_values("timestamp", ascending=False)
    return df.head(n)


def detect_anomalies(df: pd.DataFrame) -> list:
    """
    Run statistical anomaly detection on pipeline events.

    Three detection methods:
    1. Failure rate — too many failures in recent window
    2. Volume anomaly — row counts far from historical mean (z-score)
    3. SLA breach frequency — too many SLA breaches recently

    Recruiter question: "Why 2 standard deviations?"
    Answer: Industry standard. 2 std deviations captures ~95%
    of normal behavior. Anything outside is statistically unusual.

    Args:
        df: DataFrame of recent pipeline events

    Returns:
        List of AnomalyResult for detected anomalies
    """
    anomalies = []

    for pipeline in df["pipeline_name"].unique():
        pipeline_df = df[df["pipeline_name"] == pipeline]

        # Method 1: Failure rate
        failure_rate = (pipeline_df["event_type"] == "FAILURE").mean()
        if failure_rate > 0.3:
            latest = pipeline_df.iloc[0]
            anomalies.append(AnomalyResult(
                pipeline_name=pipeline,
                event_type=str(latest["event_type"]),
                severity=str(latest["severity"]),
                error_message=str(latest["error_message"]),
                expected_rows=int(latest["expected_rows"]),
                actual_rows=int(latest["actual_rows"]),
                anomaly_score=round(float(failure_rate), 4),
                is_anomaly=True,
                anomaly_reason=f"High failure rate: {failure_rate:.0%} of recent events failed",
                timestamp=str(latest["timestamp"]),
            ))
            continue

        # Method 2: Volume anomaly via z-score
        row_counts = pipeline_df["actual_rows"].astype(float)
        if len(row_counts) >= 5:
            mean = row_counts.mean()
            std = row_counts.std()
            latest_count = row_counts.iloc[0]
            if std > 0:
                z_score = abs((latest_count - mean) / std)
                if z_score > 2:
                    latest = pipeline_df.iloc[0]
                    anomalies.append(AnomalyResult(
                        pipeline_name=pipeline,
                        event_type=str(latest["event_type"]),
                        severity=str(latest["severity"]),
                        error_message=str(latest["error_message"]),
                        expected_rows=int(latest["expected_rows"]),
                        actual_rows=int(latest["actual_rows"]),
                        anomaly_score=round(float(z_score), 4),
                        is_anomaly=True,
                        anomaly_reason=f"Volume anomaly: z-score of {z_score:.2f} — row count significantly outside historical range",
                        timestamp=str(latest["timestamp"]),
                    ))
                    continue

        # Method 3: SLA breach frequency
        sla_rate = (pipeline_df["event_type"] == "SLA_BREACH").mean()
        if sla_rate > 0.25:
            latest = pipeline_df.iloc[0]
            anomalies.append(AnomalyResult(
                pipeline_name=pipeline,
                event_type=str(latest["event_type"]),
                severity=str(latest["severity"]),
                error_message=str(latest["error_message"]),
                expected_rows=int(latest["expected_rows"]),
                actual_rows=int(latest["actual_rows"]),
                anomaly_score=round(float(sla_rate), 4),
                is_anomaly=True,
                anomaly_reason=f"High SLA breach rate: {sla_rate:.0%} of recent events breached SLA",
                timestamp=str(latest["timestamp"]),
            ))

    return anomalies


if __name__ == "__main__":
    df = load_recent_events()
    anomalies = detect_anomalies(df)
    print(f"Detected {len(anomalies)} anomalies")
    for a in anomalies:
        print(f"\nPipeline: {a.pipeline_name}")
        print(f"Reason:   {a.anomaly_reason}")
        print(f"Score:    {a.anomaly_score}")
