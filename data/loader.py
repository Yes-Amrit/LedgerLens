import pandas as pd
import os

def load_dataset(path: str = "data/sample_saml_d.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
        
    critical_cols = [c for c in ["Amount", "Sender_account", "Receiver_account"] if c in df.columns]
    if critical_cols:
        df = df.dropna(subset=critical_cols)
        
    return df
