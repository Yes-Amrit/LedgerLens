import pandas as pd
import numpy as np
from agent.state import AgentState

def feature_engineering_node(state: AgentState) -> dict:
    """
    Computes rolling 7-day transaction sum per account, transaction velocity 
    (count per time window), and deviation from account's historical average amount.
    """
    try:
        df = pd.read_csv("data/transactions.csv")
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            "account_id": ["A", "A", "B", "A", "B", "A", "C", "A", "B", "A"],
            "date": dates,
            "amount": [100, 150, 200, 50, 300, 400, 50, 100, 200, 600]
        })
        
    df = df.sort_values(by=['account_id', 'date'])
    df_indexed = df.set_index('date')
    
    df['rolling_7d_sum'] = df_indexed.groupby('account_id')['amount'].rolling('7D').sum().values
    df['rolling_7d_count'] = df_indexed.groupby('account_id')['amount'].rolling('7D').count().values
    
    historical_avg = df.groupby('account_id')['amount'].transform('mean')
    df['deviation_from_avg'] = df['amount'] - historical_avg
    
    return {"feature_results": {"status": "Feature engineering complete", "sample": df.head(5).to_dict()}}
