from typing import List
from agent.schemas import ExtractedIntent

def build_plan(extracted: ExtractedIntent) -> List[str]:
    """
    Maps intent type to an ordered list of node names to execute.
    """
    intent = extracted.intent
    
    if intent == "entity_lookup":
        return ["entity_lookup_node", "anomaly_node", "risk_node", "explanation_node", "escalation_node"]
    elif intent == "aggregation_query":
        return ["aggregation_node", "risk_node", "explanation_node", "escalation_node"]
    elif intent == "pattern_search":
        return ["feature_engineering_node", "anomaly_node", "risk_node", "explanation_node", "escalation_node"]
    elif intent == "broad_exploration":
        return ["eda_node", "feature_engineering_node", "anomaly_node", "risk_node", "explanation_node", "escalation_node"]
    else:
        return ["eda_node", "feature_engineering_node", "anomaly_node", "risk_node", "explanation_node", "escalation_node"]
