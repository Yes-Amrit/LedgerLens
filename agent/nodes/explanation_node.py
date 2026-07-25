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
