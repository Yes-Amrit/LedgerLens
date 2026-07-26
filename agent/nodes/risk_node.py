from agent.state import AgentState


def _is_aggregation_only(state: dict) -> bool:
    """
    Returns True when the result was produced by aggregation_node alone,
    with NO anomaly_node execution to corroborate the finding.

    Detection heuristic: aggregation_node writes method_used="aggregation"
    into anomaly_results. The real anomaly pipeline (anomaly_node) writes
    "hybrid", "statistical", "paysim_direct_rule", etc.  If the method is
    "aggregation" (or anomaly_results is missing/empty), no multi-detector
    corroboration has occurred.
    """
    anomaly_results = state.get("anomaly_results") or {}
    method = anomaly_results.get("method_used", "")
    return method == "aggregation" or not anomaly_results


def risk_node(state: AgentState) -> dict:
    """
    Threshold logic converting anomaly_scores into low/medium/high.

    Risk-level ceiling based on evidence source:
    ─────────────────────────────────────────────
    • Aggregation-only results (single-signal: volume/threshold match)
      are capped at MEDIUM.  A pure count-based match (e.g. "10+
      transactions under $10k") is weaker evidence — it has NOT been
      independently confirmed by statistical, ML, or rule-based
      detectors.  Allowing HIGH here would replicate the naive
      threshold-only escalation that causes excessive false positives
      in traditional AML systems.

    • Anomaly-pipeline results (multi-signal: statistical + Isolation
      Forest + rule-based, fused via hybrid_scorer) CAN reach HIGH,
      because detector agreement provides genuine corroboration.
    ─────────────────────────────────────────────
    """
    anomaly_results = state.get("anomaly_results") or {}
    scores = anomaly_results.get("anomaly_scores", {})

    aggregation_only = _is_aggregation_only(state)

    risk_results = {}
    for tx_id, score in scores.items():
        if score > 0.8:
            risk_results[tx_id] = "high"
        elif score > 0.5:
            risk_results[tx_id] = "medium"
        else:
            risk_results[tx_id] = "low"

    max_score = max(scores.values()) if scores else 0
    if max_score > 0.8:
        overall = "high"
    elif max_score > 0.5:
        overall = "medium"
    else:
        overall = "low"

    # ── Aggregation-only cap ──────────────────────────────────────────
    # Single-signal (volume/threshold) matches are capped at MEDIUM.
    # This prevents auto-escalation to HIGH/report without anomaly-
    # pipeline corroboration from the hybrid detector.
    if aggregation_only:
        if overall == "high":
            overall = "medium"
        risk_results = {
            tx_id: ("medium" if level == "high" else level)
            for tx_id, level in risk_results.items()
        }

    return {
        "risk_results": {
            "transaction_risks": risk_results,
            "overall_risk": overall,
            "evidence_source": "aggregation_only" if aggregation_only else "anomaly_pipeline",
        }
    }
