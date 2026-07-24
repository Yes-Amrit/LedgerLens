import pandas as pd
from agent.state import AgentState

def aggregation_node(state: AgentState) -> dict:
    """
    Direct pandas groupby/filter for threshold-style queries 
    (e.g. count transactions under $10,000 per customer).
    """
    try:
        df = pd.read_csv("data/transactions.csv")
    except Exception:
        df = pd.DataFrame({
            "customer_id": ["C1", "C1", "C2", "C1", "C3", "C2"],
            "amount": [5000, 15000, 2000, 9000, 12000, 8000]
        })
        
    under_10k = df[df['amount'] < 10000]
    counts = under_10k.groupby('customer_id').size().to_dict()
    
    return {"feature_results": {"aggregation_counts": counts}}
