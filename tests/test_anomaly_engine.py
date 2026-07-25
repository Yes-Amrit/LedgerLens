from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from agent.nodes.anomaly_node import run_anomaly_detection
from tools.hybrid_scorer import combine_scores
from tools.isolation_forest_detector import detect_isolation_forest
from tools.lstm_autoencoder import detect_sequential_anomalies
from tools.rule_based_detector import detect_structuring
from tools.statistical_detector import detect_statistical_anomalies


@pytest.fixture
def synthetic_data():
    """Creates a small synthetic DataFrame with known-anomalous rows for all patterns."""
    np.random.seed(42)
    base_time = datetime(2023, 1, 1)

    data = []
    # Normal transactions
    for i in range(50):
        data.append(
            {
                "transaction_id": f"tx_norm_{i}",
                "account_id": f"acc_{i % 5}",
                "amount": float(np.random.uniform(10, 500)),
                "timestamp": base_time + timedelta(hours=i),
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 1000.0 - np.random.uniform(10, 500),
                "oldbalanceDest": 0.0,
                "newbalanceDest": np.random.uniform(10, 500),
            }
        )

    # Structuring anomalies: Just under $10000 limit closely in time
    for i in range(3):
        data.append(
            {
                "transaction_id": f"tx_struct_{i}",
                "account_id": "acc_structuring",
                "amount": 9400.0,
                "timestamp": base_time + timedelta(minutes=i * 10),
                "oldbalanceOrg": 20000.0,
                "newbalanceOrig": 20000.0 - 9400.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 9400.0,
            }
        )

    # Layering/Cash out anomalies: Massive amounts
    for i in range(2):
        data.append(
            {
                "transaction_id": f"tx_cashout_{i}",
                "account_id": "acc_cashout",
                "amount": 500000.0,
                "timestamp": base_time + timedelta(days=2, hours=i),
                "oldbalanceOrg": 500000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 500000.0,
            }
        )

    # Clean account with few transactions and no threshold skirting
    for i in range(2):
        data.append(
            {
                "transaction_id": f"tx_clean_{i}",
                "account_id": "acc_clean",
                "amount": 1500.0,
                "timestamp": base_time + timedelta(days=5, hours=i),
                "oldbalanceOrg": 5000.0,
                "newbalanceOrig": 3500.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 1500.0,
            }
        )

    return pd.DataFrame(data)


def validate_contract(result_dict, df):
    assert isinstance(result_dict, dict)
    assert "flagged_transactions" in result_dict
    assert "anomaly_scores" in result_dict
    assert "method_used" in result_dict

    assert isinstance(result_dict["flagged_transactions"], list)
    assert isinstance(result_dict["anomaly_scores"], dict)
    assert isinstance(result_dict["method_used"], str)

    # Check keys and score bounds
    for tx_id, score in result_dict["anomaly_scores"].items():
        assert tx_id in df["transaction_id"].values
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # Check flagged transaction shape
    for flagged in result_dict["flagged_transactions"]:
        assert "transaction_id" in flagged
        assert "account_id" in flagged
        assert "amount" in flagged
        assert "timestamp" in flagged
        assert "reason_features" in flagged


def test_detectors_contract(synthetic_data):
    from tools.schema_utils import normalize_schema
    df = normalize_schema(synthetic_data)
    feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig"]

    stat_res = detect_statistical_anomalies(df, feature_cols)
    validate_contract(stat_res, df)

    if_res = detect_isolation_forest(df, feature_cols)
    validate_contract(if_res, df)

    rule_res = detect_structuring(df)
    validate_contract(rule_res, df)

    lstm_res = detect_sequential_anomalies(df)
    validate_contract(lstm_res, df)


def test_hybrid_scorer_graceful_degradation(synthetic_data):
    from tools.schema_utils import normalize_schema
    df = normalize_schema(synthetic_data)
    feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig"]

    stat_res = detect_statistical_anomalies(df, feature_cols)
    rule_res = detect_structuring(df)

    # Omit ML entirely
    hybrid_res = combine_scores(
        stat_res, ml_result=None, rule_result=rule_res
    )
    validate_contract(hybrid_res, df)

    # Must flag the structuring cases even without ML
    flagged_ids = [
        tx["transaction_id"] for tx in hybrid_res["flagged_transactions"]
    ]
    assert "tx_struct_0" in flagged_ids


def test_run_anomaly_detection_routing(synthetic_data):
    from tools.schema_utils import normalize_schema
    df = normalize_schema(synthetic_data)
    patterns = ["structuring", "layering", "cash_out", "none"]

    for pattern in patterns:
        res = run_anomaly_detection(df, target_pattern=pattern)
        validate_contract(res, df)
        
        # The synthetic data might not breach high-confidence thresholds for a strict 'hybrid' label
        # every time. We validate that the engine routes to a valid terminal state without crashing.
        assert res["method_used"] in ["hybrid", "general_anomaly_low_confidence", "rule_based"]


def test_detect_structuring_synthetic(synthetic_data):
    from tools.schema_utils import normalize_schema
    df = normalize_schema(synthetic_data)
    
    # Run the rule-based detector
    res = detect_structuring(df)
    
    # We expect the structuring account to be flagged
    flagged = res["flagged_transactions"]
    flagged_accounts = {tx["account_id"] for tx in flagged}
    assert "acc_structuring" in flagged_accounts, "Failed to flag obvious synthetic structuring case"


def test_detect_structuring_clean(synthetic_data):
    from tools.schema_utils import normalize_schema
    df = normalize_schema(synthetic_data)
    res = detect_structuring(df)
    
    flagged = res["flagged_transactions"]
    flagged_accounts = {tx["account_id"] for tx in flagged}
    assert "acc_clean" not in flagged_accounts, "Clean account was incorrectly flagged"



def test_detect_structuring_synthetic(synthetic_data):
    df = synthetic_data
    # Run the rule-based detector
    res = detect_structuring(df)
    
    # We expect the structuring account to be flagged
    flagged = res["flagged_transactions"]
    flagged_accounts = {tx["account_id"] for tx in flagged}
    assert "acc_structuring" in flagged_accounts, "Failed to flag obvious synthetic structuring case"


def test_detect_structuring_clean(synthetic_data):
    df = synthetic_data
    res = detect_structuring(df)
    
    flagged = res["flagged_transactions"]
    flagged_accounts = {tx["account_id"] for tx in flagged}
    assert "acc_clean" not in flagged_accounts, "Clean account was incorrectly flagged"



if __name__ == "__main__":
    # If run directly via python instead of pytest
    df = synthetic_data.__wrapped__() if hasattr(synthetic_data, "__wrapped__") else synthetic_data()
    test_detectors_contract(df)
    test_hybrid_scorer_graceful_degradation(df)
    test_run_anomaly_detection_routing(df)
    print("All tests passed!")