import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from agent.state import AgentState

def explanation_node(state: AgentState) -> dict:
    """
    Use the LLM to generate a natural-language explanation given the flagged 
    transaction's features and risk level
    """
    risk_results = state.get("risk_results", {})
    overall_risk = risk_results.get("overall_risk", "low")
    evidence_source = risk_results.get("evidence_source", "anomaly_pipeline")
    
    if overall_risk == "low":
        return {"explanation": "No significant risk detected. Transactions appear normal."}
        
    anomaly_results = state.get("anomaly_results", {})
    flagged = anomaly_results.get("flagged_transactions", [])
    method = anomaly_results.get("method_used", "unknown")
    
    clean_flagged = []
    for f in flagged:
        clean_entry = {}
        for k, v in f.items():
            if isinstance(v, pd.Timestamp):
                clean_entry[k] = v.strftime('%Y-%m-%d %H:%M')
            elif isinstance(v, np.integer):
                clean_entry[k] = int(v)
            elif isinstance(v, np.floating):
                clean_entry[k] = round(float(v), 2)
            elif isinstance(v, float):
                clean_entry[k] = round(v, 2)
            else:
                clean_entry[k] = v
        clean_flagged.append(clean_entry)
        
    if len(clean_flagged) > 10:
        flagged_summary = clean_flagged[:10]
        flagged_summary.append({"note": f"...and {len(clean_flagged) - 10} more transactions"})
    else:
        flagged_summary = clean_flagged

    # ── Distinct prompt for aggregation-only vs anomaly-pipeline results ──
    # Aggregation-only results are single-signal (volume/threshold match
    # only) and must NOT be presented as if confirmed by multiple detectors.
    if evidence_source == "aggregation_only":
        prompt = f"""
    You are an AML investigator assistant.
    These accounts were identified by a VOLUME/THRESHOLD query only
    (e.g. counting transactions under a certain dollar amount).

    IMPORTANT CONTEXT — READ CAREFULLY:
    • This is a single-signal match based on transaction count/amount
      thresholds alone.
    • NO statistical anomaly detection, Isolation Forest ML model, or
      rule-based structuring checks have been run on these accounts.
    • The risk level is capped at MEDIUM because there is no multi-
      detector corroboration.
    • Do NOT imply that multiple detection methods confirmed this finding.

    Overall Risk Level: {overall_risk} (capped — aggregation only)
    Method Used: {method}
    Matched Accounts: {flagged_summary}

    Provide a concise explanation that:
    1. States which accounts matched the threshold criteria and why.
    2. Clearly notes this is based on transaction volume/amount patterns
       only, without independent anomaly detection corroboration.
    3. Recommends running the full anomaly detection pipeline (statistical
       + ML + rule-based) on these specific accounts for corroboration
       before escalating further.
    """
    else:
        prompt = f"""
    You are an AML investigator assistant. 
    Explain why these specific transactions were flagged.
    
    Overall Risk Level: {overall_risk}
    Method Used: {method}
    Flagged Data: {flagged_summary}
    
    Provide a concise natural language explanation referencing specific accounts or amounts if available.
    """
    
    print("EXACT PROMPT SENT TO LLM:\n", prompt)
    
    try:
        from agent.config import MODEL_NAME
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_output_tokens=1024, thinking_level="low")
        response = llm.invoke([HumanMessage(content=prompt)])
        if isinstance(response.content, list):
            explanation = "".join([block.get("text", "") for block in response.content if isinstance(block, dict) and block.get("type") == "text"])
        else:
            explanation = str(response.content)
    except Exception as e:
        explanation = f"Error generating explanation: {str(e)}"
        
    return {"explanation": explanation}

