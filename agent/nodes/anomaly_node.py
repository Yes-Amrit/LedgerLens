import pandas as pd
import numpy as np
import logging
from typing import Dict, List

from tools.statistical_detector import detect_statistical_anomalies
from tools.isolation_forest_detector import detect_isolation_forest
from tools.rule_based_detector import detect_structuring
from tools.hybrid_scorer import combine_scores
from agent.state import AgentState
# from tools.lstm_autoencoder import detect_sequential_anomalies # REMOVED for hackathon


# ---------------------------------------------------------------------------
# High-precision direct rule for PaySim cash-out fraud
# Data-driven insight: 97.7% of fraud has (amount == oldbalanceOrg) AND
# the source account had money (oldbalanceOrg > 0), while 0.00% of clean
# transactions share this profile — estimated precision 100%, recall 97.7%.
# ---------------------------------------------------------------------------
def _paysim_direct_rule(df: pd.DataFrame) -> dict | None:
    """
    Returns a rule_based result dict (same contract as detect_structuring)
    if PaySim columns are present, else None.
    """
    if not {'amount', 'oldbalanceOrg', 'newbalanceOrig'}.issubset(df.columns):
        return None

    flagged: List[dict] = []
    scores:  Dict[str, float] = {}

    # Rule: amount exactly drains the source account AND the account had money
    mask = (
        (df['amount'] == df['oldbalanceOrg']) &
        (df['oldbalanceOrg'] > 0) &
        (df['newbalanceOrig'] == 0)
    )

    for idx in df[mask].index:
        row   = df.loc[idx]
        tx_id = str(row['transaction_id'])
        scores[tx_id] = 1.0
        flagged.append({
            "transaction_id": tx_id,
            "account_id":     str(row.get('nameOrig', 'unknown')),
            "amount":         float(row['amount']),
            "timestamp":      row.get('step', 0),
            "reason_features": {
                "paysim_direct_rule": (
                    f"amount ({row['amount']:.2f}) exactly drained source account "
                    f"(oldbalanceOrg={row['oldbalanceOrg']:.2f}, newbalanceOrig=0)"
                )
            },
        })

    return {
        "flagged_transactions": flagged,
        "anomaly_scores":       scores,
        "method_used":          "paysim_direct_rule",
    }


def run_anomaly_detection(df: pd.DataFrame, target_pattern: str) -> dict:
    """
    Routes dataframe to appropriate detectors based on target_pattern.
    Returns:
        {
            "flagged_transactions": List[dict],
            "anomaly_scores": Dict[str, float],
            "method_used": str
        }
    """
    if df is None or df.empty:
        return {
            "flagged_transactions": [],
            "anomaly_scores": {},
            "method_used": "none"
        }

    # Hackathon scope decision: batch analysis on a sample is explicitly acceptable
    # per the problem statement's scope guidance. We sample down large dataframes
    # to avoid performance hangs/OOM on 9M+ rows, unless they are already small.
    if len(df) > 100_000:
        acct_col = 'Sender_account' if 'Sender_account' in df.columns else ('nameOrig' if 'nameOrig' in df.columns else None)
        if acct_col:
            import numpy as np
            np.random.seed(42)
            unique_accounts = df[acct_col].unique()
            avg_tx_per_acct = len(df) / len(unique_accounts)
            target_accts = int(100_000 / avg_tx_per_acct)
            if target_accts < len(unique_accounts):
                sampled_accounts = np.random.choice(unique_accounts, size=target_accts, replace=False)
                df = df[df[acct_col].isin(sampled_accounts)].copy()
            else:
                df = df.sample(n=100_000, random_state=42).copy()
        else:
            df = df.sample(n=100_000, random_state=42).copy()

    # Ensure transaction_id exists; required by contract
    if 'transaction_id' not in df.columns:
        df = df.copy()
        df['transaction_id'] = [f"tx_{i}" for i in range(len(df))]

    # ── Feature engineering ──────────────────────────────────────────────────
    # Run the feature_prep pipeline first
    from tools.feature_prep import prepare_features
    df = prepare_features(df)
    
    # PaySim features (raw + engineered)
    paysim_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    # SAML-D features (raw + engineered)
    samld_features  = ['Amount']
    
    # Add new feature prep columns
    engineered_cols = ['rolling_7d_sum', 'velocity_24h', 'amount_deviation', 'unique_counterparties_7d']

    available_cols = set(df.columns)
    feature_cols   = [c for c in (paysim_features + samld_features + engineered_cols) if c in available_cols]

    if 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
        df = df.copy()
        eps = 1e-5

        # Feature 1: Complete account drain — 976M× lift on fraud (binary)
        df['is_full_drain'] = (
            (df['amount'] == df['oldbalanceOrg']) & (df['oldbalanceOrg'] > 0)
        ).astype(float)

        # Feature 2: Drain ratio — continuous version
        df['account_drain_ratio'] = df['amount'] / (df['oldbalanceOrg'] + eps)

        # Feature 3: Balance zeroed after — additional confirmation signal
        if 'newbalanceOrig' in df.columns:
            df['balance_zeroed_after'] = (df['newbalanceOrig'] == 0).astype(float)
            feature_cols.append('balance_zeroed_after')

        # Feature 4: Destination was empty before receipt (money-mule account)
        if 'oldbalanceDest' in df.columns:
            df['dest_was_empty'] = (df['oldbalanceDest'] == 0).astype(float)
            feature_cols.append('dest_was_empty')

        feature_cols.extend(['is_full_drain', 'account_drain_ratio'])

    # Fallback to any numeric if none matched
    if not feature_cols:
        feature_cols = list(df.select_dtypes(include=['number']).columns)

    # ── Detector routing ─────────────────────────────────────────────────────
    stat_result  = None
    if_result    = None
    rule_result  = None
    direct_result = None

    if target_pattern == "structuring":
        rule_result = detect_structuring(df)
        stat_result = detect_statistical_anomalies(df, feature_cols)
        if_result   = detect_isolation_forest(df, feature_cols)

    elif target_pattern == "layering":
        # Hackathon Fallback: Layering ground-truth recall for unsupervised models 
        # (Isolation Forest / Statistical) was very poor on SAML-D. We continue routing 
        # it through these detectors for general anomaly coverage, but we override the 
        # method_used to "general_anomaly_low_confidence" to be transparent about performance.
        if_result   = detect_isolation_forest(df, feature_cols)
        stat_result = detect_statistical_anomalies(df, feature_cols)

    elif target_pattern == "cash_out":
        # UNSUPPORTED IN THIS MILESTONE — cash_out path present but untested
        direct_result = _paysim_direct_rule(df)
        stat_result   = detect_statistical_anomalies(df, feature_cols)
        if_result     = detect_isolation_forest(df, feature_cols)

    elif target_pattern == "none":
        direct_result = _paysim_direct_rule(df)
        stat_result   = detect_statistical_anomalies(df, feature_cols)
        if_result     = detect_isolation_forest(df, feature_cols)
        rule_result   = detect_structuring(df)

    else:
        logging.warning(f"Unknown target_pattern '{target_pattern}'. Defaulting to 'none' path.")
        direct_result = _paysim_direct_rule(df)
        stat_result   = detect_statistical_anomalies(df, feature_cols)
        if_result     = detect_isolation_forest(df, feature_cols)
        rule_result   = detect_structuring(df)

    # Use hybrid scorer to combine active detectors
    result = combine_scores(
        statistical_result=stat_result,
        ml_result=if_result,
        rule_result=rule_result,
        direct_result=direct_result,
        target_pattern=target_pattern
    )
    
    if target_pattern == "layering":
        result["method_used"] = "general_anomaly_low_confidence"
        
    return result

def anomaly_node(state: AgentState) -> dict:
    target_pattern = state.get("target_pattern", "none") or "none"
    df = state.get("dataset")
    if df is None:
        df = pd.DataFrame()
    results = run_anomaly_detection(df, target_pattern)
    return {"anomaly_results": results}
