"""
Tuning diagnostic: understand exactly WHY 13,515 normal transactions
get flagged by detect_structuring, and what signal separates TPs from FPs.
"""
import json, numpy as np, pandas as pd
from data.loader import load_dataset
from tools.feature_prep import prepare_features
from tools.rule_based_detector import detect_structuring

# ── Load & filter exactly like eval ──────────────────────────────────────────
df = load_dataset("data/SAML-D.csv", nrows=1500000)
with open("data/typology_mapping.json") as f:
    mapping = json.load(f)
struct_types = [k for k, v in mapping.items() if v == "structuring"]
is_struct = df["Laundering_type"].isin(struct_types)
df_filtered = df[is_struct | (df["Is_laundering"] == 0)].copy()

label_col, acct_col = "Is_laundering", "Sender_account"
pos_accts = df_filtered[df_filtered[label_col] == 1][acct_col].unique()
neg_accts = np.setdiff1d(df_filtered[acct_col].unique(), pos_accts)
rng = np.random.RandomState(42)
pos_sample = rng.choice(pos_accts, 200, replace=False)
neg_sample = rng.choice(neg_accts, min(4800, len(neg_accts)), replace=False)
pos_held = pos_sample[int(0.7 * len(pos_sample)):]
neg_held = neg_sample[int(0.7 * len(neg_sample)):]
held_accts = np.concatenate([pos_held, neg_held])
df_eval = df_filtered[df_filtered[acct_col].isin(held_accts)].copy().reset_index(drop=True)
df_eval = prepare_features(df_eval)

struct_ids = set(df_eval[df_eval[label_col] == 1]["transaction_id"].astype(str))
print(f"Eval: {len(df_eval)} rows, {len(struct_ids)} structuring TPs\n")

# ── Run detector and get scores ──────────────────────────────────────────────
result = detect_structuring(df_eval)
scores = result["anomaly_scores"]
flagged_ids = {t["transaction_id"] for t in result["flagged_transactions"]}

# ── Score distribution: TPs vs all ───────────────────────────────────────────
tp_scores = [scores.get(tid, 0.0) for tid in struct_ids]
all_scores = list(scores.values())
fp_flagged_scores = [scores[tid] for tid in flagged_ids if tid not in struct_ids]
tp_flagged_scores = [scores[tid] for tid in flagged_ids if tid in struct_ids]

print("=== SCORE DISTRIBUTION ===")
print(f"\nAll rows:  n={len(all_scores)}")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
all_hist = pd.cut(pd.Series(all_scores), bins=bins, labels=labels, right=False).value_counts().sort_index()
tp_hist = pd.cut(pd.Series(tp_scores), bins=bins, labels=labels, right=False).value_counts().sort_index()

print(f"\n{'Score Range':<12} {'All Rows':>10} {'Struct TPs':>12} {'TP Rate':>10}")
print("-" * 46)
for label in labels:
    a = all_hist.get(label, 0)
    t = tp_hist.get(label, 0)
    rate = f"{t/a*100:.1f}%" if a > 0 else "N/A"
    marker = " <-- FLAG_THRESHOLD" if label == "0.4-0.5" else ""
    print(f"{label:<12} {a:>10} {t:>12} {rate:>10}{marker}")

print(f"\n=== FLAGGED BREAKDOWN (score >= 0.40) ===")
print(f"Total flagged: {len(flagged_ids)}")
print(f"  TP flagged:  {len(tp_flagged_scores)}  (scores: {sorted(tp_flagged_scores)[:10]})")
print(f"  FP flagged:  {len(fp_flagged_scores)}")

# ── Per-account analysis of structuring accounts ─────────────────────────────
print(f"\n=== STRUCTURING ACCOUNT ANALYSIS ===")
struct_accts = df_eval[df_eval[label_col] == 1][acct_col].unique()
print(f"Structuring accounts in eval: {len(struct_accts)}")

for acct in struct_accts[:10]:
    acct_rows = df_eval[df_eval[acct_col] == acct]
    acct_struct = acct_rows[acct_rows[label_col] == 1]
    n_total = len(acct_rows)
    n_struct = len(acct_struct)
    amounts = acct_struct["Amount"].values
    timestamps = acct_struct["Timestamp"].values if "Timestamp" in acct_struct.columns else []
    
    acct_flagged = [tid for tid in acct_rows["transaction_id"].astype(str) if tid in flagged_ids]
    acct_tp_flagged = [tid for tid in acct_struct["transaction_id"].astype(str) if tid in flagged_ids]
    
    print(f"\n  Account {acct}:")
    print(f"    Total txns: {n_total}, Structuring txns: {n_struct}")
    print(f"    Struct amounts: {[f'${a:.2f}' for a in amounts[:5]]}")
    print(f"    Flagged: {len(acct_flagged)} total, {len(acct_tp_flagged)} are TPs")
    if timestamps is not None and len(timestamps) > 1:
        ts = pd.to_datetime(timestamps)
        span = (ts.max() - ts.min())
        print(f"    Struct time span: {span}")

# ── What makes FPs get flagged? ──────────────────────────────────────────────
print(f"\n=== FP ANALYSIS (sample) ===")
fp_flagged_tids = [tid for tid in flagged_ids if tid not in struct_ids]
fp_sample = fp_flagged_tids[:5]
for tid in fp_sample:
    row = df_eval[df_eval["transaction_id"].astype(str) == tid].iloc[0]
    acct = row[acct_col]
    acct_total = len(df_eval[df_eval[acct_col] == acct])
    print(f"  {tid}: amt=${row['Amount']:.2f}, acct={acct}, acct_txns={acct_total}, score={scores[tid]:.3f}")

# ── Key question: what threshold would separate TPs from FPs? ────────────────
print(f"\n=== THRESHOLD SWEEP ===")
for thresh in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
    flagged_at = {tid for tid, sc in scores.items() if sc >= thresh}
    tp_at = flagged_at & struct_ids
    fp_at = flagged_at - struct_ids
    prec = len(tp_at) / len(flagged_at) * 100 if flagged_at else 0
    rec = len(tp_at) / len(struct_ids) * 100 if struct_ids else 0
    print(f"  FLAG_THRESHOLD={thresh:.2f}: flagged={len(flagged_at):>6}, TP={len(tp_at):>3}, FP={len(fp_at):>6}, P={prec:>5.1f}%, R={rec:>5.1f}%")
