import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import IsolationForest


def detect_isolation_forest(df: pd.DataFrame, feature_columns: List[str]) -> dict:
    """
    IsolationForest-based anomaly detection.

    Improvements over the old version:
    - OLD: contamination='auto', n_estimators=100, flat min-max normalisation.
      This made the threshold a pure artefact of the sample composition and
      produced very high false-positive rates.
    - NEW changes:
      1. contamination=0.05 — tells the model to expect ~5% anomalies (realistic
         for a stratified sample). This is the single biggest precision driver.
      2. n_estimators=200 — more trees → more stable scores on small windows.
      3. Score is min-max normalised to [0,1] but the DECISION threshold now uses
         the original IsolationForest decision_function (< 0 = anomaly) rather than
         a blind 0.5 cut on the normalised range, which is more principled.
      4. feature_columns already includes `is_full_drain` and `account_drain_ratio`
         injected by anomaly_node.py — those features carry most of the signal for
         PaySim cash_out fraud and the IF will weight them heavily.
    """
    if df.empty or not feature_columns:
        return {
            "flagged_transactions": [],
            "anomaly_scores":       {},
            "method_used":          "isolation_forest",
        }

    if 'transaction_id' not in df.columns:
        df = df.copy()
        df['transaction_id'] = [f"tx_{i}" for i in range(len(df))]

    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        return {
            "flagged_transactions": [],
            "anomaly_scores":       {str(k): 0.0 for k in df['transaction_id']},
            "method_used":          "isolation_forest",
        }

    X = df[available_features].fillna(0).values

    clf = IsolationForest(
        n_estimators=200,       
        contamination=0.005,     # extremely strict for high precision
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X)

    # decision_function: negative = anomaly, positive = inlier
    decision  = clf.decision_function(X)
    raw_scores = clf.score_samples(X)   # = decision_function + offset
    pos_scores = -raw_scores            # flip so higher = more anomalous

    # Min-max to [0,1] for the contract (anomaly_scores must be in [0,1])
    s_min, s_max = pos_scores.min(), pos_scores.max()
    norm_scores  = (pos_scores - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(pos_scores)

    # Use IF's own decision boundary (decision < 0) to determine flagged set
    # This is more principled than cutting at normalised 0.5
    is_anomaly = decision < 0

    anomaly_scores = dict(zip(df['transaction_id'], norm_scores))
    flagged_transactions = []

    for idx in df[is_anomaly].index:
        row = df.loc[idx]
        account_id = (row.get('account_id')
                      or row.get('Sender_account')
                      or row.get('nameOrig', 'unknown'))
        amount     = row.get('amount', row.get('Amount', 0.0))
        timestamp  = row.get('timestamp', row.get('Time', row.get('step', 0)))

        flagged_transactions.append({
            "transaction_id": row['transaction_id'],
            "account_id":     account_id,
            "amount":         float(amount),
            "timestamp":      timestamp,
            "reason_features": {
                "isolation_forest_score": f"{anomaly_scores[row['transaction_id']]:.2f}",
            },
        })

    return {
        "flagged_transactions": flagged_transactions,
        "anomaly_scores":       {str(k): float(v) for k, v in anomaly_scores.items()},
        "method_used":          "isolation_forest",
    }
