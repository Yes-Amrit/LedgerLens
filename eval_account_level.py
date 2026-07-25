import json
import numpy as np
import pandas as pd
from tools.rule_based_detector import detect_structuring
from data.loader import load_dataset

def main():
    df = load_dataset('data/SAML-D.csv', nrows=1500000)
    if 'transaction_id' not in df.columns: df['transaction_id'] = df.index.astype(str)
    
    with open('data/typology_mapping.json') as f: mapping = json.load(f)
    struct_types = [k for k, v in mapping.items() if v == 'structuring']
    is_struct = df['Laundering_type'].isin(struct_types)
    df_filtered = df[is_struct | (df['Is_laundering'] == 0)].copy()
    
    pos_accts = df_filtered[df_filtered['Is_laundering'] == 1]['Sender_account'].unique()
    all_accts = df_filtered['Sender_account'].unique()
    neg_accts = np.setdiff1d(all_accts, pos_accts)
    
    rng = np.random.RandomState(42)
    pos_sample = rng.choice(pos_accts, 200, replace=False)
    neg_sample = rng.choice(neg_accts, min(4800, len(neg_accts)), replace=False)
    
    # held-out accounts
    pos_held = pos_sample[int(0.7 * len(pos_sample)):]
    neg_held = neg_sample[int(0.7 * len(neg_sample)):]
    held_accts = np.concatenate([pos_held, neg_held])
    
    df_eval = df_filtered[df_filtered['Sender_account'].isin(held_accts)].copy().reset_index(drop=True)
    
    # 1. Identify True Positive Accounts
    tp_accounts = set(pos_held)
    n_accounts_total = len(held_accts)
    n_tp_accounts = len(tp_accounts)
    base_rate = n_tp_accounts / n_accounts_total if n_accounts_total > 0 else 0
    print(f"Account-Level Base Rate: {base_rate*100:.2f}% ({n_tp_accounts} / {n_accounts_total})")
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'Lift':<10} | {'Flagged Accts'}")
    
    import tools.rule_based_detector as rbd
    res = rbd.detect_structuring(df_eval)
    scores = res['anomaly_scores']
    
    for t in [0.0, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00]:
        flagged_txns = {tid for tid, sc in scores.items() if sc >= t}
        # Get accounts that have at least one flagged transaction
        flagged_accts = set(df_eval[df_eval['transaction_id'].astype(str).isin(flagged_txns)]['Sender_account'])
        
        if not flagged_accts:
            print(f"{t:<10.2f} | 0.0%       | 0.0%       | 0.0x      | 0")
            continue
            
        tp = len(flagged_accts & tp_accounts)
        p = tp / len(flagged_accts)
        r = tp / n_tp_accounts
        lift = p / base_rate if base_rate > 0 else 0
        
        print(f"{t:<10.2f} | {r*100:5.1f}%     | {p*100:5.2f}%     | {lift:6.2f}x   | {len(flagged_accts)}")

if __name__ == '__main__':
    main()
