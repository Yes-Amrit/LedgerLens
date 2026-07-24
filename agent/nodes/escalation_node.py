from agent.state import AgentState

def escalation_node(state: AgentState) -> dict:
    """
    Simple rule: high risk -> "report", medium -> "flag for review", low -> "monitor"
    """
    risk_results = state.get("risk_results", {})
    overall_risk = risk_results.get("overall_risk", "low")
    
    if overall_risk == "high":
        action = "report"
    elif overall_risk == "medium":
        action = "flag for review"
    else:
        action = "monitor"
        
    summary = {
        "intent": state.get("intent"),
        "plan_executed": state.get("plan"),
        "final_escalation": action,
        "risk_level": overall_risk
    }
        
    return {"escalation_action": action, "execution_summary": summary}
