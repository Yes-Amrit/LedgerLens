from agent.state import AgentState

def risk_node(state: AgentState) -> dict:
    """
    Simple threshold logic converting anomaly_scores into low/medium/high
    """
    anomaly_results = state.get("anomaly_results", {})
    scores = anomaly_results.get("anomaly_scores", {})
    
    risk_results = {}
    for tx_id, score in scores.items():
        if score > 0.8:
            risk_results[tx_id] = "high"
        elif score > 0.5:
            risk_results[tx_id] = "medium"
        else:
            risk_results[tx_id] = "low"
            
    max_score = max(scores.values()) if scores else 0
    if max_score > 0.8:
        overall = "high"
    elif max_score > 0.5:
        overall = "medium"
    else:
        overall = "low"
        
    return {"risk_results": {"transaction_risks": risk_results, "overall_risk": overall}}
