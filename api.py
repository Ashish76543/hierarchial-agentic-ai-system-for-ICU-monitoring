"""
ICU Decision Support API

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph import graph


app = FastAPI(
    title="ICU Decision Support API",
    description="LangGraph-powered ICU clinical decision support pipeline",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class ICURequest(BaseModel):
    """
    One current ICU snapshot plus the immediately preceding
    one-hour snapshot used by the forecasting/prediction agents.
    """

    # Static patient information.
    # Values may be numeric strings in the existing graph.
    static: Dict[str, str]

    # Current snapshot (T=0).
    vitals: Dict[str, float]
    labs: Dict[str, float]
    gcs: Dict[str, float]
    ventilator: Dict[str, float]
    interventions: Dict[str, int]
    outcomes: Optional[Dict[str, float]] = None

    # Previous snapshot (T-1h).
    prior_vitals: Dict[str, float]
    prior_labs: Dict[str, float]
    prior_gcs: Dict[str, float]
    prior_ventilator: Dict[str, float]
    prior_interventions: Dict[str, int]


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

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
    forecasted_vitals: Optional[ForecastedVitals] = None
    predicted_outcomes: Optional[PredictedOutcomes] = None
    clinical_summary: Optional[str] = None

    # Fuzzy outputs: membership / recommendation strength.
    severity_scores: Optional[Dict[str, Any]] = None
    detected_syndromes: Optional[Dict[str, float]] = None
    selected_protocols: Optional[Dict[str, float]] = None
    escalation_decision: Optional[Dict[str, float]] = None
    weaning_recommendation: Optional[Dict[str, float]] = None
    recommended_interventions: Optional[Dict[str, float]] = None

    # Numeric ventilator settings and mode information.
    recommended_ventilator_changes: Optional[Dict[str, Any]] = None


# ============================================================
# HELPERS
# ============================================================

def model_to_dict(value: Any) -> Optional[Dict[str, Any]]:
    """
    Convert a Pydantic model to a dictionary while also accepting
    dictionaries returned directly by LangGraph.
    """
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return value

    raise TypeError(
        f"Expected a Pydantic model or dict, got {type(value).__name__}"
    )


# ============================================================
# MAIN INFERENCE ENDPOINT
# ============================================================

@app.post(
    "/api/analyze",
    response_model=ICUResponse,
    summary="Run ICU decision support pipeline",
)
async def analyze(request: ICURequest):
    """
    Accept current + prior ICU data and run the complete graph.

    Returns:
      - Forecasted vitals
      - Predicted outcomes
      - Clinical summary
      - Fuzzy severity scores
      - Fuzzy syndrome memberships
      - Fuzzy protocol memberships
      - Fuzzy escalation scores
      - Fuzzy weaning/readiness scores
      - Fuzzy treatment recommendations
      - Ventilator recommendations
    """

    state = {
        "messages": [],

        # ----------------------------------------------------
        # Static information
        # ----------------------------------------------------
        "original_static": request.static,

        # ----------------------------------------------------
        # Current snapshot
        # ----------------------------------------------------
        "original_vitals": request.vitals,
        "original_labs": request.labs,
        "original_gcs": request.gcs,
        "original_ventilator": request.ventilator,
        "original_interventions": request.interventions,
        "original_outcomes": request.outcomes or {},

        # ----------------------------------------------------
        # Previous snapshot
        # ----------------------------------------------------
        "prior_vitals": request.prior_vitals,
        "prior_labs": request.prior_labs,
        "prior_gcs": request.prior_gcs,
        "prior_ventilator": request.prior_ventilator,
        "prior_interventions": request.prior_interventions,

        # ----------------------------------------------------
        # Previous graph outputs
        # ----------------------------------------------------
        "previous_forecasted_vitals": None,
        "previous_clinical_summary": None,
        "previous_outcomes": None,

        # ----------------------------------------------------
        # Current graph outputs
        # ----------------------------------------------------
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

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {type(exc).__name__}: {exc}",
        ) from exc

    # --------------------------------------------------------
    # Convert graph outputs
    # --------------------------------------------------------

    forecasted = model_to_dict(
        result.get("forecasted_vitals")
    )

    outcomes = model_to_dict(
        result.get("outcomes")
    )

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    try:
        return ICUResponse(
            forecasted_vitals=(
                ForecastedVitals(**forecasted)
                if forecasted is not None
                else None
            ),

            predicted_outcomes=(
                PredictedOutcomes(**outcomes)
                if outcomes is not None
                else None
            ),

            clinical_summary=result.get(
                "clinical_summary"
            ),

            severity_scores=result.get(
                "severity_scores"
            ),

            detected_syndromes=result.get(
                "detected_syndromes"
            ),

            selected_protocols=result.get(
                "selected_protocols"
            ),

            escalation_decision=result.get(
                "escalation_decision"
            ),

            weaning_recommendation=result.get(
                "weaning_recommendation"
            ),

            recommended_interventions=result.get(
                "recommended_interventions"
            ),

            recommended_ventilator_changes=result.get(
                "recommended_ventilator_changes"
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Graph returned data that does not match the API "
                f"response schema: {type(exc).__name__}: {exc}"
            ),
        ) from exc


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    summary="Health check",
)
def health():
    return {
        "status": "ok"
    }