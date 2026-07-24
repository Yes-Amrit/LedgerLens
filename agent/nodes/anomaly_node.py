import pandas as pd
from agent.state import AgentState

def run_anomaly_detection(df: pd.DataFrame, target_pattern: str) -> dict:
    # TODO: replace with real implementation from feat/anomaly-engine — signature must not change
    return {
        "flagged_transactions": ["tx_1", "tx_2"], 
        "anomaly_scores": {"tx_1": 0.85, "tx_2": 0.6}, 
        "method_used": "stub"
    }

def anomaly_node(state: AgentState) -> dict:
    target_pattern = state.get("target_pattern", "none") or "none"
    # Stub DF
    df = pd.DataFrame()
    results = run_anomaly_detection(df, target_pattern)
    return {"anomaly_results": results}
