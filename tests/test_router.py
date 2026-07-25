import pytest
from unittest.mock import patch
from agent.planner import build_plan
from agent.intent_extractor import extract_intent


class MockExtractedIntent:
    def __init__(self, intent, target_pattern=None, entity_ids=None):
        self.intent = intent
        self.target_pattern = target_pattern
        self.entity_ids = entity_ids or []


@patch("test_router.extract_intent")
def test_pattern_search_plan(mock_extract_intent):
    mock_extract_intent.return_value = MockExtractedIntent(
        intent="pattern_search",
        target_pattern="structuring"
    )
    query = "Find structuring patterns in the last 30 days"
    extracted = mock_extract_intent(query)
    plan = build_plan(extracted)

    assert extracted.intent == "pattern_search"
    assert extracted.target_pattern == "structuring"
    assert plan == ["feature_engineering_node", "anomaly_node", "risk_node", "explanation_node", "escalation_node"]


@patch("test_router.extract_intent")
def test_aggregation_query_plan(mock_extract_intent):
    mock_extract_intent.return_value = MockExtractedIntent(intent="aggregation_query")
    query = "Which customers made 10+ transactions under $10,000?"
    extracted = mock_extract_intent(query)
    plan = build_plan(extracted)

    assert extracted.intent == "aggregation_query"
    assert plan == ["aggregation_node", "risk_node", "explanation_node", "escalation_node"]


@patch("test_router.extract_intent")
def test_entity_lookup_plan(mock_extract_intent):
    mock_extract_intent.return_value = MockExtractedIntent(
        intent="entity_lookup",
        entity_ids=["4521"]
    )
    query = "Is customer ID 4521 suspicious?"
    extracted = mock_extract_intent(query)
    plan = build_plan(extracted)

    assert extracted.intent == "entity_lookup"
    assert "4521" in (extracted.entity_ids or [])
    assert plan == ["entity_lookup_node", "anomaly_node", "risk_node", "explanation_node", "escalation_node"]


def test_dataset_loads():
    from data.loader import load_dataset
    df = load_dataset()
    assert not df.empty
    assert "Amount" in df.columns
    assert "Sender_account" in df.columns