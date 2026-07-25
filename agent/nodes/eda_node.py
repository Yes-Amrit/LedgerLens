import pandas as pd
from agent.state import AgentState

def eda_node(state: AgentState) -> dict:
    """
    Performs basic pandas profiling (describe, missing values, correlation matrix)
    on the loaded dataset.
    """
    df = state.get("dataset")
    if df is None or type(df) is not pd.DataFrame or df.empty:
        df = pd.DataFrame()
        
    # Deliberate scope decision for 48-hour hackathon: 
    # Profile/downstream processing on a 10k random sample to avoid hanging on 9M rows.
    # A representative sample (10k) is sufficient for batch analysis scope and allows PyTorch to run in reasonable time.
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
        
    eda_results = {
        "describe": df.describe().to_dict() if not df.empty else {},
        "missing_values": df.isnull().sum().to_dict() if not df.empty else {},
        "correlation": df.select_dtypes(include=['number']).corr().to_dict() if not df.empty else {}
    }
    
    return {"eda_results": eda_results, "dataset": df}
