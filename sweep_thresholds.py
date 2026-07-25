import json, numpy as np, pandas as pd
from tools.rule_based_detector import detect_structuring
from data.loader import load_dataset

def main():
    df = load_dataset("data/SAML-D.csv", nrows=1500000)
    if "transaction_id" not in df.columns:
        df["transaction_id"] = df.index.astype(str)

    with open("data/typology_mapping.json") as f:
        mapping = json.load(f)
    struct_types = [k for k, v in mapping.items() if v == "structuring"]
    is_struct = df["Laundering_type"].isin(struct_types)
    df_filtered = df[is_struct | (df["Is_laundering"] == 0)].copy()

    label_col, acct_col = "Is_laundering", "Sender_account"
    pos_accts = df_filtered[df_filtered[label_col] == 1][acct_col].unique()
    all_accts = df_filtered[acct_col].unique()
    neg_accts = np.setdiff1d(all_accts, pos_accts)

    rng = np.random.RandomState(42)
    pos_sample = rng.choice(pos_accts, 200, replace=False)
    neg_sample = rng.choice(neg_accts, min(4800, len(neg_accts)), replace=False)

    n_pos_tune = int(0.7 * len(pos_sample))
    n_neg_tune = int(0.7 * len(neg_sample))
    pos_held = pos_sample[n_pos_tune:]
    neg_held = neg_sample[n_neg_tune:]
    held_accts = np.concatenate([pos_held, neg_held])
    df_eval = df_filtered[df_filtered[acct_col].isin(held_accts)].copy().reset_index(drop=True)

    struct_ids = set(df_eval[df_eval[label_col] == 1]["transaction_id"].astype(str))
    total_txns = len(df_eval)
    n_tps = len(struct_ids)
    base_rate = n_tps / total_txns if total_txns > 0 else 0
    print(f"Total Transactions: {total_txns}, Structuring TPs: {n_tps}")
    print(f"Base Rate: {base_rate*100:.3f}%\n")
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'Lift':<10} | {'Flagged'}")
    print("-" * 65)

    for thresh in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]:
        import tools.rule_based_detector
        import importlib
        importlib.reload(tools.rule_based_detector)
        result = tools.rule_based_detector.detect_structuring(df_eval)
        scores = result["anomaly_scores"]
        
        flagged = {tid for tid, sc in scores.items() if sc >= thresh}
        tp = len(flagged & struct_ids)
        fp = len(flagged - struct_ids)
        
        precision = tp / len(flagged) if flagged else 0
        recall = tp / n_tps if n_tps else 0
        lift = precision / base_rate if base_rate > 0 else 0
        
        print(f"{thresh:<10.2f} | {recall*100:5.1f}%     | {precision*100:5.2f}%      | {lift:6.2f}x    | {len(flagged)}")

if __name__ == "__main__":
    main()
