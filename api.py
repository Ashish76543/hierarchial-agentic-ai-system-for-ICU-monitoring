"""
ICU Decision Support API
Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from graph import graph  # imports the compiled LangGraph graph

app = FastAPI(
    title="ICU Decision Support API",
    description="LangGraph-powered ICU clinical decision support pipeline",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# CORS — adjust origins for your frontend URL
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace * with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────
class ICURequest(BaseModel):
    # Static patient info (unchanged across time window)
    static: Dict[str, str]

    # Current readings (T=0 — most recent)
    vitals: Dict[str, float]
    labs: Dict[str, float]
    gcs: Dict[str, float]
    ventilator: Dict[str, float]
    interventions: Dict[str, int]
    outcomes: Optional[Dict[str, float]] = {}

    # Prior readings (T-1h — 1 hour ago)
    # Required by forecast_agent and prediction_agent for trend analysis
    prior_vitals: Dict[str, float]
    prior_labs: Dict[str, float]
    prior_gcs: Dict[str, float]
    prior_ventilator: Dict[str, float]
    prior_interventions: Dict[str, int]


# ─────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────
class ForecastedVitals(BaseModel):
    heart_rate: float
    sbp: float
    dbp: float
    mbp: float
    temperature: float
    sbp_ni: float
    dbp_ni: float
    mbp_ni: float
    spO2: float
    glucose: float
    sofa_24_hours: float


class PredictedOutcomes(BaseModel):
    discharge_outcome: bool
    icuouttime_outcome: bool
    death_outcome: bool
    sepsis_outcome: bool
    los_outcome: float


class ICUResponse(BaseModel):
    forecasted_vitals: Optional[ForecastedVitals]
    predicted_outcomes: Optional[PredictedOutcomes]
    clinical_summary: Optional[str]
    severity_scores: Optional[Dict[str, Any]]
    detected_syndromes: Optional[Dict[str, bool]]
    selected_protocols: Optional[Dict[str, Optional[str]]]
    recommended_interventions: Optional[Dict[str, int]]
    recommended_ventilator_changes: Optional[Dict[str, Any]]
    weaning_recommendation: Optional[Dict[str, int]]
    escalation_decision: Optional[Dict[str, int]]


# ─────────────────────────────────────────────
# Main inference endpoint
# ─────────────────────────────────────────────
@app.post("/api/analyze", response_model=ICUResponse, summary="Run ICU decision support pipeline")
async def analyze(request: ICURequest):
    """
    Accepts current + prior (T-1h) patient ICU data and returns:
    - Forecasted vitals (next interval)
    - Predicted outcomes (discharge, mortality, sepsis, LOS)
    - Clinical summary (trend + current + forecast + outcomes)
    - Severity scores & detected syndromes
    - Recommended interventions & ventilator settings
    - Weaning / escalation decisions
    """
    state = {
        "messages": [],

        # Static
        "original_static": request.static,

        # Current (T=0)
        "original_vitals": request.vitals,
        "original_labs": request.labs,
        "original_gcs": request.gcs,
        "original_ventilator": request.ventilator,
        "original_interventions": request.interventions,
        "original_outcomes": request.outcomes or {},

        # Prior (T-1h)
        "prior_vitals": request.prior_vitals,
        "prior_labs": request.prior_labs,
        "prior_gcs": request.prior_gcs,
        "prior_ventilator": request.prior_ventilator,
        "prior_interventions": request.prior_interventions,

        # Previous agent outputs (empty on first call)
        "previous_forecasted_vitals": None,
        "previous_clinical_summary": None,
        "previous_outcomes": None,

        # Graph outputs (all start as None)
        "forecasted_vitals": None,
        "outcomes": None,
        "clinical_summary": None,
        "severity_scores": None,
        "detected_syndromes": None,
        "selected_protocols": None,
        "escalation_decision": None,
        "recommended_ventilator_changes": None,
        "recommended_interventions": None,
        "improvement_detected": None,
        "weaning_recommendation": None,
    }

    try:
        result = graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    forecasted = result.get("forecasted_vitals")
    outcomes = result.get("outcomes")

    return ICUResponse(
        forecasted_vitals=ForecastedVitals(**forecasted.model_dump()) if forecasted else None,
        predicted_outcomes=PredictedOutcomes(**outcomes.model_dump()) if outcomes else None,
        clinical_summary=result.get("clinical_summary"),
        severity_scores=result.get("severity_scores"),
        detected_syndromes=result.get("detected_syndromes"),
        selected_protocols=result.get("selected_protocols"),
        recommended_interventions=result.get("recommended_interventions"),
        recommended_ventilator_changes=result.get("recommended_ventilator_changes"),
        weaning_recommendation=result.get("weaning_recommendation"),
        escalation_decision=result.get("escalation_decision"),
    )


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}