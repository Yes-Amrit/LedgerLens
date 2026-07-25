import pandas as pd
import pytest
from tools.rule_based_detector import detect_structuring

def test_detect_structuring_empty():
    df = pd.DataFrame()
    result = detect_structuring(df)
    assert result['method_used'] == 'rule_based'
    assert len(result['flagged_transactions']) == 0

def test_detect_structuring_missing_columns():
    df = pd.DataFrame({"transaction_id": [1, 2]})
    result = detect_structuring(df)
    assert len(result['flagged_transactions']) == 0

def test_detect_structuring_single_transaction_below_threshold():
    df = pd.DataFrame({
        "transaction_id": ["tx1"],
        "Sender_account": ["acc1"],
        "Amount": [500],
        "Timestamp": ["2023-01-01 10:00:00"]
    })
    result = detect_structuring(df)
    assert len(result['flagged_transactions']) == 0

def test_detect_structuring_basic_fuzzy_score():
    # If account has >= 3 transactions and an amount is in structuring zone [1000, 9500]
    # score gets 0.3 + 0.4 * ((amt-1000)/8500)
    # If amount is 6000, score = 0.3 + 0.4*(5000/8500) = 0.3 + 0.235 = 0.535 >= 0.40 FLAG_THRESHOLD
    df = pd.DataFrame({
        "transaction_id": ["tx1", "tx2", "tx3"],
        "Sender_account": ["acc1", "acc1", "acc1"],
        "Amount": [100, 200, 6000],
        "Timestamp": [
            "2023-01-01 10:00:00",
            "2023-01-02 10:00:00",
            "2023-01-03 10:00:00"
        ]
    })
    result = detect_structuring(df)
    flagged = {t["transaction_id"] for t in result["flagged_transactions"]}
    assert "tx3" in flagged
    assert "tx1" not in flagged
    assert "tx2" not in flagged

def test_detect_structuring_rolling_sum():
    # Two transactions that sum to ~10,000 within 1 day window
    df = pd.DataFrame({
        "transaction_id": ["tx1", "tx2"],
        "Sender_account": ["acc1", "acc1"],
        "Amount": [4900, 4800], # Sum = 9700, which is between 7000 and 15000
        "Timestamp": [
            "2023-01-01 10:00:00",
            "2023-01-01 11:00:00"
        ]
    })
    # Acct has 2 transactions, so n_acct=2 -> score = 0.15
    # Amount 4900 score = 0.4*(3900/8500) = 0.183
    # Base score = 0.15 + 0.183 = 0.333
    # Rolling sum hits [7000, 15000], adds 0.3 -> 0.633 > 0.4
    result = detect_structuring(df)
    flagged = {t["transaction_id"] for t in result["flagged_transactions"]}
    assert "tx1" in flagged
    assert "tx2" in flagged

def test_detect_structuring_spec_rule():
    # 3+ transactions in [threshold*0.9, threshold]
    # threshold = 7258.49, spec_low = 6532.64
    df = pd.DataFrame({
        "transaction_id": ["tx1", "tx2", "tx3"],
        "Sender_account": ["acc1", "acc1", "acc1"],
        "Amount": [7000, 7100, 7200],
        "Timestamp": [
            "2023-01-01 10:00:00",
            "2023-01-01 11:00:00",
            "2023-01-01 12:00:00"
        ]
    })
    result = detect_structuring(df)
    flagged = {t["transaction_id"] for t in result["flagged_transactions"]}
    assert "tx1" in flagged
    assert "tx2" in flagged
    assert "tx3" in flagged
    
    # Verify score is exactly 1.0 due to spec rule
    scores = result["anomaly_scores"]
    assert scores["tx1"] == 1.0
    assert scores["tx2"] == 1.0
    assert scores["tx3"] == 1.0
