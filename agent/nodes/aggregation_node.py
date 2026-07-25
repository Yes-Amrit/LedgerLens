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
        # Cast amount to numeric to avoid string comparison bugs
        numeric_amounts = pd.to_numeric(df[amount_col], errors='coerce')
        under_10k = df[numeric_amounts < 10000]
        counts_series = under_10k.groupby(account_col).size()
        print("INTERMEDIATE GROUPBY COUNTS (first 5):", counts_series.head().to_dict())
        counts = counts_series[counts_series >= 10].to_dict()
    else:
        counts = {}
        
    flagged = [
        {
            "transaction_id": f"agg_{acct}",
            "account_id": str(acct),
            "transaction_count": cnt
        } for acct, cnt in counts.items()
    ]
    scores = {str(acct): 1.0 for acct in counts.keys()}
    
    return {
        "feature_results": {"aggregation_counts": counts},
        "anomaly_results": {
            "flagged_transactions": flagged,
            "anomaly_scores": scores,
            "method_used": "aggregation"
        }
    }
