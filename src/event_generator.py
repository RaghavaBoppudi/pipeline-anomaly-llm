import uuid
import random
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import json
import os


# Recruiter question: "Why enums instead of plain strings?"
# Answer: Enums prevent typos, make valid values explicit,
# and make the code self-documenting. If someone passes
# "FAILUR" instead of "FAILURE" an enum catches it immediately.

class EventType(Enum):
    FAILURE = "FAILURE"
    SLA_BREACH = "SLA_BREACH"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    VOLUME_ANOMALY = "VOLUME_ANOMALY"
    SUCCESS = "SUCCESS"


class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Recruiter question: "Why a dataclass instead of a plain dict?"
# Answer: Dataclasses give you type hints, default values,
# and a clean repr for free. They make the schema explicit
# and self-documenting. In production you'd know exactly
# what fields every event has.

@dataclass
class PipelineEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_name: str = ""
    event_type: str = ""
    severity: str = ""
    expected_rows: int = 0
    actual_rows: int = 0
    error_message: str = ""
    duration_seconds: int = 0
    affected_table: str = ""


PIPELINES = [
    "customer_transactions_ingestion",
    "fraud_detection_feature_pipeline",
    "daily_risk_aggregation",
    "merchant_data_sync",
    "account_balance_reconciliation",
    "payment_gateway_ingestion",
    "kyc_document_processing",
    "credit_score_update_pipeline",
]

ERROR_MESSAGES = {
    EventType.FAILURE: [
        "java.lang.OutOfMemoryError: Java heap space",
        "Connection timeout after 30000ms",
        "Disk quota exceeded on worker node",
        "NullPointerException in transformation step",
        "Failed to acquire database lock after 10 retries",
    ],
    EventType.SCHEMA_DRIFT: [
        "New column detected: customer_risk_tier (VARCHAR)",
        "Column type changed: transaction_amount DECIMAL(10,2) -> DECIMAL(15,4)",
        "Column removed: legacy_account_code",
        "Nullable constraint changed on: merchant_id",
    ],
    EventType.SLA_BREACH: [
        "Pipeline did not complete by SLA window of 06:00 AM",
        "Processing time exceeded threshold: 4h 23m vs 2h limit",
        "Downstream dependency not ready within SLA",
    ],
    EventType.VOLUME_ANOMALY: [
        "Row count 45% below 30-day average",
        "Duplicate records detected: 12,453 rows",
        "Row count 312% above expected threshold",
        "Zero records ingested from source",
    ],
}


def generate_event(anomaly: bool = False, pipeline_name: str = None) -> PipelineEvent:
    """
    Generate a single pipeline event.

    Args:
        anomaly: If True generates an anomalous event
        pipeline_name: Specific pipeline or random if None

    Returns:
        PipelineEvent dataclass instance
    """
    pipeline = pipeline_name or random.choice(PIPELINES)
    base_rows = random.randint(100000, 1000000)

    if not anomaly:
        return PipelineEvent(
            pipeline_name=pipeline,
            event_type=EventType.SUCCESS.value,
            severity=Severity.LOW.value,
            expected_rows=base_rows,
            actual_rows=int(base_rows * random.uniform(0.98, 1.02)),
            error_message="",
            duration_seconds=random.randint(300, 3600),
            affected_table=f"{pipeline}_raw",
        )

    event_type = random.choice([
        EventType.FAILURE,
        EventType.SLA_BREACH,
        EventType.SCHEMA_DRIFT,
        EventType.VOLUME_ANOMALY,
    ])

    severity = random.choice([Severity.MEDIUM, Severity.MEDIUM, Severity.HIGH])

    return PipelineEvent(
        pipeline_name=pipeline,
        event_type=event_type.value,
        severity=severity.value,
        expected_rows=base_rows,
        actual_rows=0 if event_type == EventType.FAILURE else int(base_rows * random.uniform(0.1, 0.5)),
        error_message=random.choice(ERROR_MESSAGES[event_type]),
        duration_seconds=random.randint(3600, 14400),
        affected_table=f"{pipeline}_raw",
    )


def generate_events(total: int = 500, anomaly_rate: float = 0.2) -> list:
    """
    Generate a batch of pipeline events.

    Args:
        total: Total number of events to generate
        anomaly_rate: Fraction of events that are anomalous

    Returns:
        List of PipelineEvent instances
    """
    events = []
    for _ in range(total):
        is_anomaly = random.random() < anomaly_rate
        events.append(generate_event(anomaly=is_anomaly))
    return events


def save_events(events: list, output_dir: str = "data/events"):
    """Save events to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    events_dict = [vars(e) for e in events]

    json_path = os.path.join(output_dir, "events.json")
    with open(json_path, "w") as f:
        json.dump(events_dict, f, indent=2)

    csv_path = os.path.join(output_dir, "events.csv")
    pd.DataFrame(events_dict).to_csv(csv_path, index=False)

    print(f"Saved {len(events)} events to {output_dir}")
    print(f"Anomaly count: {sum(1 for e in events if e.event_type != 'SUCCESS')}")
    print(f"Success count: {sum(1 for e in events if e.event_type == 'SUCCESS')}")


if __name__ == "__main__":
    events = generate_events(total=500, anomaly_rate=0.2)
    save_events(events)
