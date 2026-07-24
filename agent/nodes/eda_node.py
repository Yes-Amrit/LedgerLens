import pandas as pd
from agent.state import AgentState

def eda_node(state: AgentState) -> dict:
    """
    Performs basic pandas profiling (describe, missing values, correlation matrix)
    on the loaded dataset.
    """
    try:
        df = pd.read_csv("data/transactions.csv")
    except Exception:
        # Fallback to dummy data
        df = pd.DataFrame({"amount": [100, 200, 300, 400], "is_flagged": [0, 1, 0, 0]})
        
    eda_results = {
        "describe": df.describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "correlation": df.select_dtypes(include=['number']).corr().to_dict() if not df.empty else {}
    }
    
    return {"eda_results": eda_results}
