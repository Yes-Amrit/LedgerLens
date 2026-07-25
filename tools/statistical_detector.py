import numpy as np
import pandas as pd
from typing import List, Dict


def detect_statistical_anomalies(df: pd.DataFrame, feature_columns: List[str]) -> dict:
    """
    Z-score + IQR outlier detection.

    Improvements over the old version:
    - OLD: Flagged a row if the per-column MAX of z-score and IQR was > sigmoid(0.5).
      Problem: a single large-but-legit column (e.g. amount alone) triggers the flag,
      producing huge numbers of false positives at 10% precision.
    - NEW: Flags only when at least TWO feature columns independently exceed the
      z-score threshold. This simple change dramatically reduces false-positives on
      PaySim where large-but-legit CASH_OUT transactions exist in high volume.
      Also raises the Z-score threshold from 3.0 to 3.5 to reduce noise from
      heavy-tailed financial distributions.
    """
    if df.empty or not feature_columns:
        return {
            "flagged_transactions": [],
            "anomaly_scores":       {},
            "method_used":          "statistical_zscore",
        }

    if 'transaction_id' not in df.columns:
        df = df.copy()
        df['transaction_id'] = [f"tx_{i}" for i in range(len(df))]

    Z_THRESHOLD     = 3.5    # raised from 3.0 — less sensitive to heavy-tail noise
    MIN_COLS_FLAGGED = 2      # NEW: at least 2 columns must breach to flag a row

    combined_scores  = pd.Series(0.0, index=df.index)
    col_breach_count = pd.Series(0,   index=df.index)   # NEW
    reasons_df       = pd.DataFrame(index=df.index)

    for col in feature_columns:
        if col not in df.columns:
            continue

        vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
        mean = vals.mean()
        std  = vals.std()
        z    = (vals - mean) / std if std > 0 else pd.Series(0.0, index=df.index)

        q1  = vals.quantile(0.25)
        q3  = vals.quantile(0.75)
        iqr = q3 - q1
        iqr_up  = (vals - q3) / iqr if iqr > 0 else pd.Series(0.0, index=df.index)
        iqr_lo  = (q1 - vals) / iqr if iqr > 0 else pd.Series(0.0, index=df.index)

        col_score = np.maximum(np.abs(z), np.maximum(iqr_up, iqr_lo))
        combined_scores = np.maximum(combined_scores, col_score)

        # Track how many columns breached the threshold
        col_breach_count += (col_score >= Z_THRESHOLD).astype(int)
        reasons_df[col]   = col_score

    # Normalize to [0,1] using sigmoid anchored at Z_THRESHOLD
    scores = 1 / (1 + np.exp(-(combined_scores - Z_THRESHOLD)))

    # ── Dual-column gate ─────────────────────────────────────────────────────────
    # A row is only flagged if ≥2 columns independently breach the threshold
    flagged_mask = (scores > 0.5) & (col_breach_count >= MIN_COLS_FLAGGED)

    anomaly_scores      = dict(zip(df['transaction_id'], scores))
    flagged_transactions: List[dict] = []

    for idx in df[flagged_mask].index:
        row = df.loc[idx]
        reasons = {
            col: f"Score: {reasons_df.loc[idx, col]:.2f}"
            for col in feature_columns
            if col in reasons_df.columns and reasons_df.loc[idx, col] >= Z_THRESHOLD
        }
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
            "reason_features": reasons,
        })

    return {
        "flagged_transactions": flagged_transactions,
        "anomaly_scores":       {str(k): float(v) for k, v in anomaly_scores.items()},
        "method_used":          "statistical_zscore",
    }
