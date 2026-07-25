"""
Diagnostic: run each detector individually on the eval sample to see
which one is (or isn't) catching the structuring rows.
"""
import json, numpy as np, pandas as pd
from data.loader import load_dataset
from tools.feature_prep import prepare_features
from tools.rule_based_detector import detect_structuring
from tools.statistical_detector import detect_statistical_anomalies
from tools.isolation_forest_detector import detect_isolation_forest
from tools.hybrid_scorer import combine_scores

# ── Load & filter exactly like eval_anomaly_engine.py ────────────────────────
df = load_dataset("data/SAML-D.csv", nrows=1500000)
if "transaction_id" not in df.columns:
    df["transaction_id"] = df.index.astype(str)

with open("data/typology_mapping.json") as f:
    mapping = json.load(f)
struct_types = [k for k, v in mapping.items() if v == "structuring"]
is_struct = df["Laundering_type"].isin(struct_types)
df_filtered = df[is_struct | (df["Is_laundering"] == 0)].copy()

# Sample same way as eval
label_col, acct_col = "Is_laundering", "Sender_account"
pos_accts = df_filtered[df_filtered[label_col] == 1][acct_col].unique()
all_accts = df_filtered[acct_col].unique()
neg_accts = np.setdiff1d(all_accts, pos_accts)

rng = np.random.RandomState(42)
pos_sample = rng.choice(pos_accts, 200, replace=False)
neg_sample = rng.choice(neg_accts, min(4800, len(neg_accts)), replace=False)

# Use held-out split (30%)
n_pos_tune = int(0.7 * len(pos_sample))
n_neg_tune = int(0.7 * len(neg_sample))
pos_held = pos_sample[n_pos_tune:]
neg_held = neg_sample[n_neg_tune:]
held_accts = np.concatenate([pos_held, neg_held])
df_eval = df_filtered[df_filtered[acct_col].isin(held_accts)].copy().reset_index(drop=True)

print(f"Eval sample: {len(df_eval)} rows, {df_eval[label_col].sum()} true positives")
print(f"Columns: {list(df_eval.columns[:15])}...")
print(f"Has 'Timestamp': {'Timestamp' in df_eval.columns}")
print(f"Has 'timestamp': {'timestamp' in df_eval.columns}")
print(f"Has 'Time':      {'Time' in df_eval.columns}")

# Show a few structuring rows
struct_rows = df_eval[df_eval[label_col] == 1]
print(f"\nSample structuring rows (first 5):")
print(struct_rows[["transaction_id", acct_col, "Amount", "Laundering_type"]].head().to_string())
print(f"Amount range of structuring rows: {struct_rows['Amount'].min():.2f} - {struct_rows['Amount'].max():.2f}")

# ── Prepare features (like anomaly_node does) ────────────────────────────────
df_eval = prepare_features(df_eval)
print(f"\nAfter prepare_features:")
print(f"  rolling_7d_sum NaN count: {df_eval['rolling_7d_sum'].isna().sum()} / {len(df_eval)}")
print(f"  velocity_24h NaN count:   {df_eval['velocity_24h'].isna().sum()} / {len(df_eval)}")

# ── Run each detector ────────────────────────────────────────────────────────
rule_result = detect_structuring(df_eval)
print(f"\n=== RULE-BASED DETECTOR ===")
print(f"  Flagged: {len(rule_result['flagged_transactions'])}")
rule_flagged_ids = {t['transaction_id'] for t in rule_result['flagged_transactions']}
struct_ids = set(struct_rows['transaction_id'].astype(str))
rule_tp = rule_flagged_ids & struct_ids
print(f"  TP (rule alone): {len(rule_tp)}")
print(f"  FP (rule alone): {len(rule_flagged_ids) - len(rule_tp)}")

feature_cols = [c for c in ['Amount', 'rolling_7d_sum', 'velocity_24h', 'amount_deviation', 'unique_counterparties_7d'] if c in df_eval.columns]
print(f"\nFeature columns for IF/stat: {feature_cols}")

stat_result = detect_statistical_anomalies(df_eval, feature_cols)
print(f"\n=== STATISTICAL DETECTOR ===")
print(f"  Flagged: {len(stat_result['flagged_transactions'])}")
stat_flagged_ids = {t['transaction_id'] for t in stat_result['flagged_transactions']}
stat_tp = stat_flagged_ids & struct_ids
print(f"  TP (stat alone): {len(stat_tp)}")

if_result = detect_isolation_forest(df_eval, feature_cols)
print(f"\n=== ISOLATION FOREST ===")
print(f"  Flagged: {len(if_result['flagged_transactions'])}")
if_flagged_ids = {t['transaction_id'] for t in if_result['flagged_transactions']}
if_tp = if_flagged_ids & struct_ids
print(f"  TP (IF alone): {len(if_tp)}")

# ── Hybrid scorer ────────────────────────────────────────────────────────────
hybrid = combine_scores(stat_result, if_result, rule_result, target_pattern="structuring")
print(f"\n=== HYBRID SCORER ===")
print(f"  Flagged: {len(hybrid['flagged_transactions'])}")
hybrid_flagged_ids = {t['transaction_id'] for t in hybrid['flagged_transactions']}
hybrid_tp = hybrid_flagged_ids & struct_ids
print(f"  TP (hybrid): {len(hybrid_tp)}")
print(f"  FP (hybrid): {len(hybrid_flagged_ids) - len(hybrid_tp)}")

# ── Agreement analysis ───────────────────────────────────────────────────────
print(f"\n=== AGREEMENT ANALYSIS (on struct rows) ===")
for tid in list(struct_ids)[:10]:
    in_rule = tid in rule_flagged_ids
    in_stat = tid in stat_flagged_ids
    in_if   = tid in if_flagged_ids
    in_hybrid = tid in hybrid_flagged_ids
    print(f"  {tid}: rule={in_rule}, stat={in_stat}, IF={in_if} -> hybrid={in_hybrid}")
