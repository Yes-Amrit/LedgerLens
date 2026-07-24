from typing import TypedDict, Optional, List
from typing_extensions import NotRequired

class AgentState(TypedDict):
    raw_query: str
    intent: str
    date_filter: NotRequired[Optional[dict]]
    entity_ids: NotRequired[Optional[List[str]]]
    target_pattern: NotRequired[Optional[str]]
    transaction_type_filter: NotRequired[Optional[str]]
    plan: List[str]
    eda_results: NotRequired[Optional[dict]]
    feature_results: NotRequired[Optional[dict]]
    anomaly_results: NotRequired[Optional[dict]]
    risk_results: NotRequired[Optional[dict]]
    explanation: NotRequired[Optional[str]]
    escalation_action: NotRequired[Optional[str]]
    execution_summary: dict
