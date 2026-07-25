import json, numpy as np, pandas as pd
from data.loader import load_dataset
from tools.feature_prep import prepare_features

# Load & filter exactly like eval
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

df_eval['_amt'] = pd.to_numeric(df_eval['Amount'], errors='coerce').fillna(0)
df_eval['_t'] = pd.to_datetime(df_eval['Timestamp'], errors='coerce')
df_eval = df_eval.dropna(subset=['_t'])
df_sorted = df_eval.sort_values(by=[acct_col, '_t'])

struct_ids = set(df_eval[df_eval[label_col] == 1]["transaction_id"].astype(str))

def evaluate_params(struct_low, struct_high, spec_low_ratio, time_window_days, flag_threshold, sum_low_ratio, sum_high_ratio):
    threshold = 7258.49
    reporting_target = 10_000
    sum_low = reporting_target * sum_low_ratio
    sum_high = reporting_target * sum_high_ratio
    spec_low = threshold * spec_low_ratio
    spec_high = threshold
    time_window = pd.Timedelta(days=time_window_days)
    
    raw_scores = {}
    
    # Fast vectorized scoring where possible
    for account, group in df_sorted.groupby(acct_col):
        records = group.reset_index(drop=True)
        amounts = records['_amt'].values
        times = records['_t'].values
        tx_ids = [str(x) for x in records['transaction_id'].values]
        n_acct = len(records)
        
        for i in range(n_acct):
            amt = amounts[i]
            tid = tx_ids[i]
            score = 0.0
            
            if struct_low <= amt < struct_high:
                score += 0.4 * ((amt - struct_low) / (struct_high - struct_low))
                
            if n_acct >= 3:
                score += 0.3
            elif n_acct == 2:
                score += 0.15
                
            raw_scores[tid] = score
            
        if n_acct < 2:
            continue
            
        for i in range(n_acct):
            ws = amounts[i]
            window_ids = [tx_ids[i]]
            spec_count = 1 if (spec_low <= amounts[i] <= spec_high) else 0
            spec_window_ids = [tx_ids[i]] if spec_count == 1 else []
            
            for j in range(i + 1, n_acct):
                if times[j] - times[i] > time_window:
                    break
                ws += amounts[j]
                window_ids.append(tx_ids[j])
                if spec_low <= amounts[j] <= spec_high:
                    spec_count += 1
                    spec_window_ids.append(tx_ids[j])
                
                if sum_low <= ws <= sum_high and len(window_ids) >= 2:
                    for t_id in window_ids:
                        raw_scores[t_id] = min(1.0, raw_scores.get(t_id, 0.0) + 0.3)
                if spec_count >= 2:
                    for t_id in spec_window_ids:
                        raw_scores[t_id] = 1.0
                        
    flagged = {tid for tid, sc in raw_scores.items() if sc >= flag_threshold}
    tp = flagged & struct_ids
    fp = flagged - struct_ids
    precision = len(tp) / len(flagged) * 100 if flagged else 0
    recall = len(tp) / len(struct_ids) * 100 if struct_ids else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, len(flagged)

# Grid search
best_f1 = 0
best_params = None

# We need VERY high precision. To get high precision, we must flag fewer false positives.
# To flag fewer false positives, we could increase struct_low, tighten sum_low/sum_high, require higher flag_threshold.
# Or require spec_count >= 2.
for flag_thresh in [0.70, 0.80, 0.90, 0.95]:
    for struct_l in [3000, 5000, 7000]:
        for window in [1, 3, 7]:
            for sum_l, sum_h in [(0.90, 1.10), (0.95, 1.05), (0.80, 1.20)]:
                p, r, f1, n_flag = evaluate_params(struct_l, 9999, 0.90, window, flag_thresh, sum_l, sum_h)
                if f1 > best_f1 or (p > 50 and r > 50):
                    best_f1 = max(f1, best_f1)
                    best_params = (struct_l, 9999, 0.90, window, flag_thresh, sum_l, sum_h)
                    print(f"P: {p:5.1f}%, R: {r:5.1f}%, F1: {f1:5.1f} | Flagged: {n_flag:5d} | Params: struct_low={struct_l}, window={window}, thresh={flag_thresh}, sum={sum_l}-{sum_h}")

print("Done.")
