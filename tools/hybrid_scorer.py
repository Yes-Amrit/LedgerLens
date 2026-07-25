from typing import Dict, List, Optional


def combine_scores(
    statistical_result: Optional[dict] = None,
    ml_result:          Optional[dict] = None,
    rule_result:        Optional[dict] = None,
    direct_result:      Optional[dict] = None,
    target_pattern:     str            = "none",
) -> dict:
    """
    Fuses outputs from multiple detectors.
    
    Decision logic:
    - 2-detector agreement gate: a transaction is only flagged if 2+ detectors
      independently flag it. This eliminates single-detector false positives.
    - Anomaly scores are the MAX across all active detectors (for PR-AUC ranking).
    - `detector_agreement_count` is injected into each flagged entry for downstream
      explainability (e.g. explanation_node).
    - direct_result (e.g. PaySim cash_out) is still included as it is 100% precision.
    """

    all_results = {
        "statistical":       statistical_result,
        "isolation_forest":  ml_result,
        "rule_based":        rule_result,
        "direct":            direct_result,
    }
    
    active_results = {k: v for k, v in all_results.items() if v is not None}

    if not active_results:
        return {
            "flagged_transactions": [],
            "anomaly_scores":       {},
            "method_used":          "hybrid",
        }

    # Collect all tx IDs
    all_tx_ids: set = set()
    for res in active_results.values():
        all_tx_ids.update(res.get("anomaly_scores", {}).keys())

    final_scores:    Dict[str, float] = {}
    flagged_tx_data: Dict[str, dict]  = {}
    flag_counts:     Dict[str, int]   = {tx_id: 0 for tx_id in all_tx_ids}
    best_methods:    Dict[str, str]   = {tx_id: "hybrid" for tx_id in all_tx_ids}

    for tx_id in all_tx_ids:
        max_score = 0.0
        best_method = "hybrid"

        for method, res in active_results.items():
            score = res.get("anomaly_scores", {}).get(tx_id, 0.0)
            if score > max_score:
                max_score = score
                best_method = method
            elif score == max_score and best_method == "hybrid":
                # Fallback if first score is 0.0
                best_method = method

            is_flagged = any(
                tx["transaction_id"] == tx_id
                for tx in res.get("flagged_transactions", [])
            )

            if is_flagged:
                flag_counts[tx_id] += 1

                # Build merged reason metadata
                if tx_id not in flagged_tx_data:
                    tx_obj = next(
                        (tx for tx in res.get("flagged_transactions", [])
                         if tx["transaction_id"] == tx_id),
                        None,
                    )
                    if tx_obj:
                        flagged_tx_data[tx_id] = tx_obj.copy()
                        flagged_tx_data[tx_id]["reason_features"] = {}

                if tx_id in flagged_tx_data:
                    tx_obj = next(
                        (tx for tx in res.get("flagged_transactions", [])
                         if tx["transaction_id"] == tx_id),
                        None,
                    )
                    if tx_obj and "reason_features" in tx_obj:
                        for k, v in tx_obj["reason_features"].items():
                            flagged_tx_data[tx_id]["reason_features"][f"{method}_{k}"] = v

        final_scores[tx_id] = max_score
        best_methods[tx_id] = best_method

    # 2. Re-introduce 2-detector gate (from earlier robust version).
    # We require at least 2 detectors to flag a transaction to eliminate false positives.
    final_flagged: List[dict] = []
    for tx_id in all_tx_ids:
        if flag_counts[tx_id] >= 2 and tx_id in flagged_tx_data:
            # Inject the detector agreement count for downstream node logic
            entry = flagged_tx_data[tx_id]
            entry["detector_agreement_count"] = flag_counts[tx_id]
            final_flagged.append(entry)

    # We can't return a single method_used if it varies by transaction, 
    # but the contract requires a top-level method_used. We'll return "hybrid"
    return {
        "flagged_transactions": final_flagged,
        "anomaly_scores":       final_scores,
        "method_used":          "hybrid",
    }
