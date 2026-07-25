import pandas as pd
from agent.state import AgentState

def entity_lookup_node(state: AgentState) -> dict:
    """
    Filter dataset to a specific entity_id, return their transaction history 
    and any existing flags.
    """
    df = state.get("dataset")
    if df is None or type(df) is not pd.DataFrame or df.empty:
        return {"feature_results": {"error": "Dataset is empty."}}
        
    entity_ids = state.get("entity_ids", [])
    if not entity_ids:
        return {"feature_results": {"error": "No entity_id provided."}}
        
    target_id = str(entity_ids[0])
    account_col = "Sender_account" if "Sender_account" in df.columns else "entity_id"
    
    if account_col in df.columns:
        df[account_col] = df[account_col].astype(str)
        history = df[df[account_col] == target_id]
    else:
        history = pd.DataFrame()
    
    flag_col = "Is_laundering" if "Is_laundering" in df.columns else ("is_flagged" if "is_flagged" in df.columns else "flagged")
    flags_count = history[flag_col].sum() if flag_col in history.columns else 0
    
    return {
        "dataset": history,
        "feature_results": {
            "entity_id": target_id,
            "transaction_count": len(history),
            "flags_count": int(flags_count)
        }
    }
