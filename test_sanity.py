import json
import numpy as np
import pandas as pd
from tools.rule_based_detector import detect_structuring
from data.loader import load_dataset

def test_sanity_accounts():
    print("Loading SAML-D...")
    df = load_dataset('data/SAML-D.csv', nrows=1500000)
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = df.index.astype(str)
    
    sanity_accounts = ["92172", "344654", "672794"]
    df_sanity = df[df["Sender_account"].isin(sanity_accounts)].copy()
    
    print(f"Testing on {len(df_sanity)} rows from sanity accounts...")
    result = detect_structuring(df_sanity)
    flagged = result.get("flagged_transactions", [])
    
    if len(flagged) == 0:
        print("SUCCESS: 0 transactions flagged in sanity accounts.")
    else:
        print(f"FAILURE: {len(flagged)} transactions flagged in sanity accounts!")
        for tx in flagged:
            print(tx)

if __name__ == "__main__":
    test_sanity_accounts()
