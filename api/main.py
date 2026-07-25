"""
LedgerLens FastAPI backend.
Hackathon scope: single /investigate endpoint + /health.
No auth, no versioning — this is a demo backend.
Run with:  uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional
import pandas as pd

from agent.graph import run_graph
from data.loader import load_dataset

app = FastAPI(title="LedgerLens", description="AML investigation API backed by a LangGraph agent")

# ── Dataset loaded once at startup, reused by every request ──────────────────
_dataset: Optional[pd.DataFrame] = None

@app.on_event("startup")
def _startup() -> None:
    global _dataset
    _dataset = load_dataset()


# ── Request / Response models ─────────────────────────────────────────────────
class InvestigateRequest(BaseModel):
    query: str


class InvestigateResponse(BaseModel):
    execution_summary: dict
    flagged_transactions: Any     # list[dict] or list[str] depending on anomaly-engine version
    risk_level: str
    explanation: Optional[str]
    escalation_action: str
    anomaly_method_used: Optional[str]
    # Transparency field: surfaced when anomaly engine is running in stub mode
    calibration_warning: Optional[str]


# ── POST /investigate ─────────────────────────────────────────────────────────
@app.post("/investigate", response_model=InvestigateResponse)
def investigate(req: InvestigateRequest):
    """Run the LangGraph agent pipeline against the given natural-language query."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    try:
        result = run_graph(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    summary = result.get("execution_summary", {})
    intent  = result.get("intent", "")

    # 404 if entity_lookup returned 0 rows
    if intent == "entity_lookup":
        feature_res = result.get("feature_results", {})
        if isinstance(feature_res, dict):
            if feature_res.get("error"):
                raise HTTPException(status_code=404, detail=feature_res["error"])
            if feature_res.get("transaction_count", -1) == 0:
                entity_ids = result.get("entity_ids") or []
                entity = entity_ids[0] if entity_ids else "unknown"
                raise HTTPException(
                    status_code=404,
                    detail=f"No transactions found for entity_id '{entity}'"
                )

    anomaly  = result.get("anomaly_results", {}) or {}
    risk_res = result.get("risk_results", {}) or {}

    # Detect stub mode and surface it clearly rather than silently misleading
    method_used = anomaly.get("method_used", "unknown")
    calibration_warning = None
    if method_used == "stub":
        calibration_warning = (
            "anomaly_node is running in STUB mode — scores are synthetic placeholders. "
            "risk_level and escalation_action are NOT based on real detection. "
            "Pending: merge of feat/anomaly-engine."
        )

    return InvestigateResponse(
        execution_summary=summary,
        flagged_transactions=anomaly.get("flagged_transactions", []),
        risk_level=risk_res.get("overall_risk", summary.get("risk_level", "unknown")),
        explanation=result.get("explanation"),
        escalation_action=result.get("escalation_action", "unknown"),
        anomaly_method_used=method_used,
        calibration_warning=calibration_warning,
    )


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick liveness + dataset readiness check."""
    dataset_ok = _dataset is not None and not _dataset.empty
    return {
        "status": "ok" if dataset_ok else "degraded",
        "dataset_loaded": dataset_ok,
        "dataset_rows": len(_dataset) if dataset_ok else 0,
        "dataset_columns": list(_dataset.columns) if dataset_ok else [],
    }
