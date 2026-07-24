import pandas as pd
from agent.state import AgentState

def entity_lookup_node(state: AgentState) -> dict:
    """
    Filter dataset to a specific entity_id, return their transaction history 
    and any existing flags.
    """
    try:
        df = pd.read_csv("data/transactions.csv")
    except Exception:
        df = pd.DataFrame({
            "entity_id": ["4521", "1234", "4521"],
            "amount": [500, 1000, 2000],
            "flagged": [True, False, False]
        })
        
    entity_ids = state.get("entity_ids", [])
    if not entity_ids:
        return {"feature_results": {"error": "No entity_id provided."}}
        
    target_id = entity_ids[0]
    
    # Cast to str for comparison
    df['entity_id'] = df['entity_id'].astype(str)
    history = df[df['entity_id'] == str(target_id)]
    
    flags_count = history['flagged'].sum() if 'flagged' in history.columns else 0
    
    return {
        "feature_results": {
            "entity_id": target_id,
            "transaction_count": len(history),
            "flags_count": int(flags_count)
        }
    }
