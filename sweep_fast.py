import json, numpy as np, pandas as pd
from data.loader import load_dataset
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
held_accts = np.concatenate([pos_sample[int(0.7 * len(pos_sample)):], neg_sample[int(0.7 * len(neg_sample)):]])
df_eval = df_filtered[df_filtered['Sender_account'].isin(held_accts)].copy().reset_index(drop=True)
df_eval['_amt'] = pd.to_numeric(df_eval['Amount'], errors='coerce').fillna(0)
df_eval['_t'] = pd.to_datetime(df_eval['Timestamp'], errors='coerce')
df_eval = df_eval.dropna(subset=['_t'])
df_eval.sort_values(by=['Sender_account', '_t'], inplace=True)
struct_ids = set(df_eval[df_eval['Is_laundering'] == 1]['transaction_id'].astype(str))

def eval_config(struct_low, sum_low_pct, sum_high_pct, spec_min_count):
    threshold = 7258.49
    reporting_target = 10000
    sum_l = reporting_target * sum_low_pct
    sum_h = reporting_target * sum_high_pct
    spec_l = threshold * 0.90
    spec_h = threshold
    
    flagged = set()
    for acct, group in df_eval.groupby('Sender_account'):
        amounts = group['_amt'].values
        times = group['_t'].values
        tids = group['transaction_id'].astype(str).values
        n = len(amounts)
        for i in range(n):
            if amounts[i] < struct_low: continue
            ws = amounts[i]
            wids = [tids[i]]
            scount = 1 if spec_l <= amounts[i] <= spec_h else 0
            swids = [tids[i]] if scount == 1 else []
            for j in range(i+1, n):
                if (times[j] - times[i]) > np.timedelta64(7, 'D'): break
                if amounts[j] < struct_low: continue
                ws += amounts[j]
                wids.append(tids[j])
                if spec_l <= amounts[j] <= spec_h:
                    scount += 1
                    swids.append(tids[j])
                if sum_l <= ws <= sum_h and len(wids) >= 2:
                    for tid in wids: flagged.add(tid)
                if scount >= spec_min_count:
                    for tid in swids: flagged.add(tid)
    tp = len(flagged & struct_ids)
    p = tp/len(flagged) if flagged else 0
    r = tp/77
    lift = p / (77/27480)
    print(f"struct={struct_low}, sum={sum_low_pct}-{sum_high_pct}, spec={spec_min_count} -> P={p*100:.2f}%, R={r*100:.1f}%, Lift={lift:.1f}x, F={len(flagged)}, TP={tp}")

# Evaluate strict rules that do not rely on fuzzy scores
eval_config(1000, 0.95, 1.05, 3)
eval_config(3000, 0.95, 1.05, 2)
eval_config(3000, 0.98, 1.02, 2)
eval_config(5000, 0.95, 1.05, 2)
eval_config(1000, 0.99, 1.01, 3)
eval_config(3000, 0.90, 1.10, 3)
