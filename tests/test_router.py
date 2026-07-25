import pytest
from agent.intent_extractor import extract_intent
from agent.planner import build_plan

def test_pattern_search_plan():
    query = "Find structuring patterns in the last 30 days"
    extracted = extract_intent(query)
    plan = build_plan(extracted)
    
    assert extracted.intent == "pattern_search"
    assert extracted.target_pattern == "structuring"
    assert plan == ["feature_engineering_node", "anomaly_node", "risk_node", "explanation_node", "escalation_node"]

def test_aggregation_query_plan():
    query = "Which customers made 10+ transactions under $10,000?"
    extracted = extract_intent(query)
    plan = build_plan(extracted)
    
    assert extracted.intent == "aggregation_query"
    assert plan == ["aggregation_node", "risk_node", "explanation_node", "escalation_node"]

def test_entity_lookup_plan():
    query = "Is customer ID 4521 suspicious?"
    extracted = extract_intent(query)
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
