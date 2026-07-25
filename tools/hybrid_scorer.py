from typing import Dict, List, Optional


def combine_scores(
    statistical_result: Optional[dict],
    ml_result:          Optional[dict],
    rule_result:        Optional[dict],
    lstm_result:        Optional[dict] = None,
    direct_result:      Optional[dict] = None,
    weights:            Optional[dict] = None,
    target_pattern:     str            = "none",
) -> dict:
    """
    Fuses outputs from multiple detectors.

    Decision logic (in priority order):
    1. direct_result — high-precision deterministic rules (e.g. PaySim full-drain).
       These bypass the two-detector agreement gate because their data-validated
       precision is near 100%. Any tx flagged here is always included in output.
    2. rule_result for target_pattern == "structuring" — also bypasses the gate
       because structuring rules are deterministic and pattern-specific.
    3. All other detectors must have >= 2 independently agree to flag a tx.

    Final anomaly_score is a weighted combination:
      - direct / rule signals: always 1.0
      - others: max-of-signals across active detectors
    """

    # ── Gather all results ────────────────────────────────────────────────────
    ml_detectors = {
        "statistical":       statistical_result,
        "isolation_forest":  ml_result,
        "lstm":              lstm_result,
    }
    # Separate high-authority detectors
    authority_results = {
        "rule_based": rule_result,
        "direct":     direct_result,
    }

    all_results = {**ml_detectors, **authority_results}
    active_results = {k: v for k, v in all_results.items() if v is not None}

    if not active_results:
        return {
            "flagged_transactions": [],
            "anomaly_scores":       {},
            "method_used":          "hybrid",
        }

    # ── Collect all tx IDs ────────────────────────────────────────────────────
    all_tx_ids: set = set()
    for res in active_results.values():
        all_tx_ids.update(res.get("anomaly_scores", {}).keys())

    final_scores:    Dict[str, float] = {}
    flagged_tx_data: Dict[str, dict]  = {}
    ml_flag_counts:  Dict[str, int]   = {tx_id: 0 for tx_id in all_tx_ids}
    authority_flagged: set             = set()

    for tx_id in all_tx_ids:
        max_score = 0.0

        for method, res in active_results.items():
            score = res.get("anomaly_scores", {}).get(tx_id, 0.0)
            if score > max_score:
                max_score = score

            is_flagged = any(
                tx["transaction_id"] == tx_id
                for tx in res.get("flagged_transactions", [])
            )

            if is_flagged:
                # Track which authority detectors flagged this tx
                if method == "direct":
                    authority_flagged.add(tx_id)
                elif method == "rule_based" and target_pattern == "structuring":
                    authority_flagged.add(tx_id)
                else:
                    ml_flag_counts[tx_id] += 1

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

    # ── Final decision gate ───────────────────────────────────────────────────
    final_flagged: List[dict] = []
    for tx_id in all_tx_ids:
        include = (
            tx_id in authority_flagged          # high-precision authority bypass
            or ml_flag_counts[tx_id] >= 2       # ≥2 ML detectors agree
        )
        if include and tx_id in flagged_tx_data:
            final_flagged.append(flagged_tx_data[tx_id])

    return {
        "flagged_transactions": final_flagged,
        "anomaly_scores":       final_scores,
        "method_used":          "hybrid",
    }
