"""
ICU Decision Support API

Run:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import graph


# ============================================================
# APP
# ============================================================

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

    # --------------------------------------------------------
    # Static patient information
    # --------------------------------------------------------

    static: Dict[str, str]

    # --------------------------------------------------------
    # Current snapshot (T=0)
    # --------------------------------------------------------

    vitals: Dict[str, float]

    labs: Dict[str, float]

    gcs: Dict[str, float]

    ventilator: Dict[str, float]

    interventions: Dict[str, int]

    outcomes: Optional[Dict[str, float]] = None

    # --------------------------------------------------------
    # Previous snapshot (T-1h)
    # --------------------------------------------------------

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

    # ========================================================
    # Prediction outputs
    # ========================================================

    forecasted_vitals: Optional[ForecastedVitals] = None

    predicted_outcomes: Optional[PredictedOutcomes] = None

    clinical_summary: Optional[str] = None

    # ========================================================
    # Fuzzy severity
    # ========================================================

    severity_scores: Optional[Dict[str, Any]] = None

    # ========================================================
    # Syndrome detection
    # ========================================================

    detected_syndromes: Optional[Dict[str, float]] = None

    # ========================================================
    # Protocol selection
    # ========================================================

    selected_protocols: Optional[Dict[str, float]] = None

    # ========================================================
    # Escalation
    # ========================================================

    escalation_decision: Optional[Dict[str, float]] = None

    # ========================================================
    # Weaning
    # ========================================================

    weaning_recommendation: Optional[Dict[str, float]] = None

    liberation_scores: Optional[Dict[str, float]] = None

    # ========================================================
    # Treatment
    # ========================================================

    recommended_interventions: Optional[Dict[str, float]] = None

    # ========================================================
    # Ventilator
    # ========================================================

    recommended_ventilator_changes: Optional[Dict[str, Any]] = None

    # ========================================================
    # FINAL ACTION / TRACKING
    # ========================================================

    final_action: Optional[Dict[str, Any]] = None

    treatment_change_detected: Optional[bool] = None

    ventilation_change_detected: Optional[bool] = None

    decision_trace: Optional[list] = None


# ============================================================
# HELPERS
# ============================================================

def model_to_dict(value: Any) -> Optional[Dict[str, Any]]:
    """
    Convert a Pydantic model to a dictionary.

    Also accepts dictionaries returned directly by LangGraph.
    """

    if value is None:
        return None

    # Pydantic v2
    if hasattr(value, "model_dump"):
        return value.model_dump()

    # Already a dictionary
    if isinstance(value, dict):
        return value

    raise TypeError(
        f"Expected a Pydantic model or dict, "
        f"got {type(value).__name__}"
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
    Run the complete ICU decision-support pipeline.

    Pipeline:

        Input
          ↓
        Forecast
          ↓
        Outcome Prediction
          ↓
        Clinical Summary
          ↓
        Severity Scoring
          ↓
        Syndrome Detection
          ↓
        Protocol Selection
          ↓
        Weaning Assessment
          ↓
        Escalation
          ↓
        Treatment Selection
          ↓
        Ventilator Settings
          ↓
        Final Action
    """

    # ========================================================
    # BUILD INITIAL LANGGRAPH STATE
    # ========================================================

    state = {

        # ----------------------------------------------------
        # Messages
        # ----------------------------------------------------

        "messages": [],

        # ----------------------------------------------------
        # Static patient information
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

        "weaning_recommendation": None,

        "liberation_scores": None,

        "recommended_interventions": None,

        "recommended_ventilator_changes": None,

        "improvement_detected": None,

        # ----------------------------------------------------
        # Final action tracking
        # ----------------------------------------------------

        "final_action": None,

        "treatment_change_detected": False,

        "ventilation_change_detected": False,

        "decision_trace": None,
    }

    # ========================================================
    # RUN LANGGRAPH
    # ========================================================

    try:

        result = graph.invoke(state)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Pipeline error: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    # ========================================================
    # CONVERT PREDICTION MODELS
    # ========================================================

    try:

        forecasted = model_to_dict(
            result.get("forecasted_vitals")
        )

        outcomes = model_to_dict(
            result.get("outcomes")
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to convert graph prediction output: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    # ========================================================
    # BUILD API RESPONSE
    # ========================================================

    try:

        response = ICUResponse(

            # ------------------------------------------------
            # Forecast
            # ------------------------------------------------

            forecasted_vitals=(
                ForecastedVitals(**forecasted)
                if forecasted is not None
                else None
            ),

            # ------------------------------------------------
            # Outcomes
            # ------------------------------------------------

            predicted_outcomes=(
                PredictedOutcomes(**outcomes)
                if outcomes is not None
                else None
            ),

            # ------------------------------------------------
            # Clinical summary
            # ------------------------------------------------

            clinical_summary=result.get(
                "clinical_summary"
            ),

            # ------------------------------------------------
            # Severity
            # ------------------------------------------------

            severity_scores=result.get(
                "severity_scores"
            ),

            # ------------------------------------------------
            # Syndromes
            # ------------------------------------------------

            detected_syndromes=result.get(
                "detected_syndromes"
            ),

            # ------------------------------------------------
            # Protocols
            # ------------------------------------------------

            selected_protocols=result.get(
                "selected_protocols"
            ),

            # ------------------------------------------------
            # Escalation
            # ------------------------------------------------

            escalation_decision=result.get(
                "escalation_decision"
            ),

            # ------------------------------------------------
            # Weaning
            # ------------------------------------------------

            weaning_recommendation=result.get(
                "weaning_recommendation"
            ),

            liberation_scores=result.get(
                "liberation_scores"
            ),

            # ------------------------------------------------
            # Treatment
            # ------------------------------------------------

            recommended_interventions=result.get(
                "recommended_interventions"
            ),

            # ------------------------------------------------
            # Ventilator
            # ------------------------------------------------

            recommended_ventilator_changes=result.get(
                "recommended_ventilator_changes"
            ),

            # ------------------------------------------------
            # Final action
            # ------------------------------------------------

            final_action=result.get(
                "final_action"
            ),

            # ------------------------------------------------
            # Change detection
            # ------------------------------------------------

            treatment_change_detected=result.get(
                "treatment_change_detected",
                False,
            ),

            ventilation_change_detected=result.get(
                "ventilation_change_detected",
                False,
            ),

            # ------------------------------------------------
            # Complete decision trace
            # ------------------------------------------------

            decision_trace=result.get(
                "decision_trace"
            ),
        )

        return response

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Graph returned data that does not match "
                "the API response schema: "
                f"{type(exc).__name__}: {exc}"
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