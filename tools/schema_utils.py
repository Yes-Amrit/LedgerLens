import pandas as pd

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a dataframe from a specific dataset source to a standard schema expected by detectors.
    Standard schema requires: transaction_id, account_id, amount, timestamp
    """
    df_norm = df.copy()
    
    # SAML-D mappings
    if "Sender_account" in df_norm.columns:
        df_norm = df_norm.rename(columns={"Sender_account": "account_id"})
    if "Amount" in df_norm.columns:
        df_norm = df_norm.rename(columns={"Amount": "amount"})
    if "Timestamp" in df_norm.columns:
        df_norm = df_norm.rename(columns={"Timestamp": "timestamp"})
        
    # PaySim mappings
    if "nameOrig" in df_norm.columns:
        df_norm = df_norm.rename(columns={"nameOrig": "account_id"})
    if "timestamp" not in df_norm.columns and "step" in df_norm.columns:
        import datetime
        base = datetime.datetime(2023, 1, 1)
        df_norm["timestamp"] = df_norm["step"].apply(lambda s: base + datetime.timedelta(hours=int(s)))

    # Ensure transaction_id exists
    if "transaction_id" not in df_norm.columns:
        if df_norm.index.name == "transaction_id" or "transaction_id" in df_norm.index.names:
            df_norm = df_norm.reset_index(level="transaction_id")
        else:
            df_norm["transaction_id"] = df_norm.index.astype(str)

    return df_norm
