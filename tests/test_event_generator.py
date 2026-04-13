import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.event_generator import generate_event, generate_events, EventType


def test_event_has_required_fields():
    """Every event must have all required fields populated."""
    event = generate_event(anomaly=False)
    assert event.event_id is not None
    assert event.timestamp is not None
    assert event.pipeline_name != ""
    assert event.event_type != ""


def test_normal_event_is_success():
    """Non-anomaly events should always be SUCCESS type."""
    event = generate_event(anomaly=False)
    assert event.event_type == EventType.SUCCESS.value


def test_anomaly_event_is_not_success():
    """Anomaly events should never be SUCCESS type."""
    event = generate_event(anomaly=True)
    assert event.event_type != EventType.SUCCESS.value


def test_anomaly_rate_is_roughly_correct():
    """
    Generate 1000 events at 20% anomaly rate.
    Verify anomaly count is within acceptable range.
    Allow 5% tolerance on either side.
    """
    events = generate_events(total=1000, anomaly_rate=0.2)
    anomaly_count = sum(1 for e in events if e.event_type != EventType.SUCCESS.value)
    anomaly_rate = anomaly_count / len(events)
    assert 0.15 <= anomaly_rate <= 0.25


def test_event_ids_are_unique():
    """Every event must have a unique ID."""
    events = generate_events(total=100)
    ids = [e.event_id for e in events]
    assert len(ids) == len(set(ids))
