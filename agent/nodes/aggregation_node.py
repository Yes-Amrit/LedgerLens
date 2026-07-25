import pandas as pd
from agent.state import AgentState

def aggregation_node(state: AgentState) -> dict:
    """
    Direct pandas groupby/filter for threshold-style queries 
    (e.g. count transactions under $10,000 per customer).
    """
    df = state.get("dataset")
    if df is None or type(df) is not pd.DataFrame or df.empty:
        return {"feature_results": {"aggregation_counts": {}}}
        
    amount_col = "Amount" if "Amount" in df.columns else "amount"
    account_col = "Sender_account" if "Sender_account" in df.columns else "customer_id"
    
    if amount_col in df.columns and account_col in df.columns:
        under_10k = df[df[amount_col] < 10000]
        counts = under_10k.groupby(account_col).size().to_dict()
    else:
        counts = {}
        
    return {"feature_results": {"aggregation_counts": counts}}
