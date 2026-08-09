from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Optional, Annotated, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatOllama(model="hf.co/BioMistral/BioMistral-7B-GGUF:latest")


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────
class ForecastSt(BaseModel):
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
    model_config = {"extra": "forbid"}


class ForecastOutput(BaseModel):
    forecasted_vitals: ForecastSt
    model_config = {"extra": "forbid"}


class OutcomesSt(BaseModel):
    discharge_outcome: bool
    icuouttime_outcome: bool
    death_outcome: bool
    sepsis_outcome: bool
    los_outcome: float
    model_config = {"extra": "forbid"}


class OutcomeOutput(BaseModel):
    outcome: OutcomesSt
    model_config = {"extra": "forbid"}


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
class OverallState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # Static (unchanged across time window)
    original_static: Dict[str, str]

    # Current readings (T=0)
    original_vitals: Dict[str, float]
    original_labs: Dict[str, float]
    original_gcs: Dict[str, float]
    original_ventilator: Dict[str, float]
    original_interventions: Dict[str, int]
    original_outcomes: Dict[str, float]

    # Prior readings (T-1h)
    prior_vitals: Dict[str, float]
    prior_labs: Dict[str, float]
    prior_gcs: Dict[str, float]
    prior_ventilator: Dict[str, float]
    prior_interventions: Dict[str, int]

    # Previous agent outputs (from last graph run)
    previous_forecasted_vitals: Optional[ForecastSt]
    previous_clinical_summary: Optional[str]
    previous_outcomes: Optional[OutcomesSt]

    # Graph outputs
    forecasted_vitals: Optional[ForecastSt]
    clinical_summary: Optional[str]
    outcomes: Optional[OutcomesSt]
    recommended_ventilator_changes: Optional[Dict[str, object]]
    recommended_interventions: Optional[Dict[str, float]]
    improvement_detected: Optional[bool]
    severity_scores: Optional[Dict[str, float]]
    detected_syndromes: Optional[Dict[str, float]]
    selected_protocols: Optional[Dict[str, float]]
    escalation_decision: Optional[Dict[str, float]]
    weaning_recommendation: Optional[Dict[str, float]]


# ─────────────────────────────────────────────
# Structured output models
# ─────────────────────────────────────────────
outcome_model = llm.with_structured_output(OutcomeOutput)
forecast_model = llm.with_structured_output(ForecastOutput)


# ─────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────
def start_agent(state: OverallState):
    return {}


def forecast_agent(state: OverallState):
    previous_forecast = state.get("forecasted_vitals")

    system_prompt = SystemMessage(content=(
        "You are an ICU physiological forecasting agent. "
        "You are given TWO snapshots of patient data separated by 1 hour: "
        "'Prior (T-1h)' is the reading from 1 hour ago, and 'Current (T=0)' is the most recent reading. "
        "Use the trend between these two time points to predict the patient's future vital signs "
        "for the next monitoring interval. "
        "Forecast realistic ICU physiological values. "
        "Do not hallucinate parameters. Only use defined vital fields."
    ))

    human_prompt = HumanMessage(content=f"""
        Static variables: {state['original_static']}

        --- Prior vitals (T-1h) ---
        Vitals:        {state['prior_vitals']}
        Labs:          {state['prior_labs']}
        GCS:           {state['prior_gcs']}
        Ventilator:    {state['prior_ventilator']}
        Interventions: {state['prior_interventions']}

        --- Current vitals (T=0) ---
        Vitals:        {state['original_vitals']}
        Labs:          {state['original_labs']}
        GCS:           {state['original_gcs']}
        Ventilator:    {state['original_ventilator']}
        Interventions: {state['original_interventions']}
    """)

    structured = forecast_model.invoke([system_prompt, human_prompt])
    return {"previous_forecasted_vitals": previous_forecast, "forecasted_vitals": structured.forecasted_vitals}


def prediction_agent(state: OverallState):
    previous_outcomes = state.get("outcomes")

    system_prompt = SystemMessage(content=(
        "You are an ICU clinical prediction agent. "
        "You are given TWO snapshots of patient data separated by 1 hour: "
        "'Prior (T-1h)' is the reading from 1 hour ago, and 'Current (T=0)' is the most recent reading. "
        "Analyze the trend between the two time points along with current vitals, labs, "
        "ventilator settings, and interventions. "
        "Predict: discharge_outcome, icuouttime_outcome, death_outcome, sepsis_outcome, los_outcome. "
        "Be medically realistic and conservative."
    ))

    human_prompt = HumanMessage(content=f"""
        Static variables: {state['original_static']}

        --- Prior data (T-1h) ---
        Vitals:        {state['prior_vitals']}
        Labs:          {state['prior_labs']}
        GCS:           {state['prior_gcs']}
        Ventilator:    {state['prior_ventilator']}
        Interventions: {state['prior_interventions']}

        --- Current data (T=0) ---
        Vitals:        {state['original_vitals']}
        Labs:          {state['original_labs']}
        GCS:           {state['original_gcs']}
        Ventilator:    {state['original_ventilator']}
        Interventions: {state['original_interventions']}
    """)

    structured = outcome_model.invoke([system_prompt, human_prompt])
    return {"previous_outcomes": previous_outcomes, "outcomes": structured.outcome}


# ─────────────────────────────────────────────────────────────────────────────
# summarization_agent fans-in AFTER forecast_agent and prediction_agent,
# so forecasted_vitals and outcomes are guaranteed populated in state.
# Prompt is kept compact/flat to avoid throttling BioMistral-7B-GGUF.
# ─────────────────────────────────────────────────────────────────────────────
def summarization_agent(state: OverallState):
    previous_summary = state.get("clinical_summary")
    forecasted_vitals = state.get("forecasted_vitals")
    outcomes = state.get("outcomes")

    fv = forecasted_vitals.model_dump() if forecasted_vitals else {}
    oc = outcomes

    system_prompt = SystemMessage(content=(
        "You are an ICU physician. Write a short clinical summary with exactly four paragraphs:\n"
        "Paragraph 1 - TREND: compare T-1h vs current vitals and state if the patient is improving or deteriorating.\n"
        "Paragraph 2 - CURRENT STATUS: describe the patient's condition right now based on vitals, labs, GCS, and ventilator.\n"
        "Paragraph 3 - FORECAST: describe what the predicted vitals suggest will happen next.\n"
        "Paragraph 4 - OUTCOMES: state each predicted outcome and what it means clinically, flagging any high-risk findings.\n"
        "Be brief and use plain medical language. No bullet points, no extra headings."
    ))

    human_prompt = HumanMessage(content=(
        f"Patient: {state['original_static'].get('anchor_age','?')}yo "
        f"{state['original_static'].get('gender','?')}, "
        f"unit: {state['original_static'].get('first_careunit','?')}.\n\n"

        f"PRIOR VITALS (T-1h): HR={state['prior_vitals'].get('heart_rate')}, "
        f"SBP={state['prior_vitals'].get('sbp')}, MBP={state['prior_vitals'].get('mbp')}, "
        f"SpO2={state['prior_vitals'].get('spO2')}%, Temp={state['prior_vitals'].get('temperature')}C. "
        f"pH={state['prior_labs'].get('pH')}, pCO2={state['prior_labs'].get('pCO2')}, "
        f"GCS={state['prior_gcs'].get('gcs')}, SOFA={state['prior_gcs'].get('sofa_24_hours')}.\n"

        f"CURRENT VITALS (T=0): HR={state['original_vitals'].get('heart_rate')}, "
        f"SBP={state['original_vitals'].get('sbp')}, MBP={state['original_vitals'].get('mbp')}, "
        f"SpO2={state['original_vitals'].get('spO2')}%, Temp={state['original_vitals'].get('temperature')}C. "
        f"pH={state['original_labs'].get('pH')}, pCO2={state['original_labs'].get('pCO2')}, "
        f"pO2={state['original_labs'].get('pO2')}, GCS={state['original_gcs'].get('gcs')}, "
        f"SOFA={state['original_gcs'].get('sofa_24_hours')}. "
        f"FiO2={state['original_ventilator'].get('set_fio21')}%, "
        f"PEEP={state['original_ventilator'].get('set_peep1')}, "
        f"RR={state['original_ventilator'].get('set_rr1')}. "
        f"invasive={state['original_interventions'].get('invasive')}, "
        f"NIV={state['original_interventions'].get('noninvasive')}, "
        f"vasopressor={state['original_interventions'].get('vasopressor')}.\n\n"

        f"FORECASTED VITALS: HR={fv.get('heart_rate','?')}, SBP={fv.get('sbp','?')}, "
        f"MBP={fv.get('mbp','?')}, SpO2={fv.get('spO2','?')}%, "
        f"SOFA={fv.get('sofa_24_hours','?')}.\n\n"

        f"PREDICTED OUTCOMES: discharge={oc.discharge_outcome if oc else 'N/A'}, "
        f"ICU_exit={oc.icuouttime_outcome if oc else 'N/A'}, "
        f"death_risk={oc.death_outcome if oc else 'N/A'}, "
        f"sepsis={oc.sepsis_outcome if oc else 'N/A'}, "
        f"LOS={round(oc.los_outcome, 1) if oc else 'N/A'} hrs.\n\n"

        "Write the four-paragraph clinical summary now."
    ))

    response = llm.invoke([system_prompt, human_prompt])
    return {"previous_clinical_summary": previous_summary, "clinical_summary": response.content}




# ============================================================
# FUZZY MEMBERSHIP FUNCTIONS
# ============================================================

def trapezoid_membership(x, a, b, c, d):
    """
    General trapezoidal membership function.

    a ---- b ===== c ---- d
    0      1       1      0

    Handles:
        normal trapezoids
        left-shoulder functions
        right-shoulder functions
    """

    if x is None:
        return 0.0

    # Left shoulder
    if a == b:

        if x <= b:
            return 1.0

        if x >= d:
            return 0.0

        if x <= c:
            return 1.0

        return (d - x) / (d - c)

    # Right shoulder
    if c == d:

        if x <= a:
            return 0.0

        if x >= c:
            return 1.0

        if x <= b:
            return (x - a) / (b - a)

        return 1.0

    # Outside
    if x <= a or x >= d:
        return 0.0

    # Rising edge
    if a < x < b:
        return (x - a) / (b - a)

    # Plateau
    if b <= x <= c:
        return 1.0

    # Falling edge
    return (d - x) / (d - c)


# ============================================================
# HEMODYNAMIC MEMBERSHIP
# ============================================================

def low_map_membership(mbp):

    return trapezoid_membership(
        mbp,
        50,
        60,
        60,
        65
    )


def high_shock_index_membership(shock_index):

    return trapezoid_membership(
        shock_index,
        0.7,
        0.8,
        1.0,
        1.2
    )


# ============================================================
# OXYGENATION MEMBERSHIP
# ============================================================

def low_spo2_membership(spo2):

    return trapezoid_membership(
        spo2,
        85,
        90,
        90,
        93
    )


# ============================================================
# ACID-BASE MEMBERSHIP
# ============================================================

def acidemia_membership(ph):

    return trapezoid_membership(
        ph,
        7.20,
        7.25,
        7.25,
        7.35
    )


def hypercapnia_membership(pco2):

    return trapezoid_membership(
        pco2,
        45,
        50,
        55,
        60
    )


# ============================================================
# VENTILATOR SUPPORT MEMBERSHIP
# ============================================================

def high_fio2_membership(fio2):

    return trapezoid_membership(
        fio2,
        40,
        50,
        60,
        80
    )


def high_peep_membership(peep):

    return trapezoid_membership(
        peep,
        5,
        8,
        10,
        14
    )


# ============================================================
# P/F RATIO MEMBERSHIPS
# ============================================================

def pf_mild_membership(pf_ratio):

    return trapezoid_membership(
        pf_ratio,
        200,
        225,
        300,
        350
    )


def pf_moderate_membership(pf_ratio):

    return trapezoid_membership(
        pf_ratio,
        100,
        125,
        200,
        225
    )


def pf_severe_membership(pf_ratio):

    return trapezoid_membership(
        pf_ratio,
        50,
        50,
        80,
        100
    )


# ============================================================
# GCS MEMBERSHIPS
# ============================================================

def gcs_mild_membership(gcs):

    return trapezoid_membership(
        gcs,
        12,
        13,
        15,
        15
    )


def gcs_moderate_membership(gcs):

    return trapezoid_membership(
        gcs,
        8,
        9,
        12,
        13
    )


def gcs_severe_membership(gcs):

    return trapezoid_membership(
        gcs,
        3,
        3,
        6,
        8
    )

# ============================================================
# FUZZY SEVERITY AGENT
# ============================================================

def severity_scoring_agent(state: OverallState):

    forecast = state.get("forecasted_vitals")

    labs = state.get("original_labs") or {}
    gcs = state.get("original_gcs") or {}
    vent = state.get("original_ventilator") or {}

    if forecast is None:
        return {}

    scores = {}

    # ========================================================
    # 1. HEMODYNAMIC STATUS
    # ========================================================

    # Shock Index
    #
    # SI = Heart Rate / Systolic Blood Pressure

    if forecast.sbp > 0:

        shock_index = (
            forecast.heart_rate /
            forecast.sbp
        )

        scores["shock_index"] = shock_index

        scores["high_shock_index"] = (
            high_shock_index_membership(
                shock_index
            )
        )

    else:

        scores["shock_index"] = None
        scores["high_shock_index"] = 0.0


    # Low MAP membership

    scores["low_map"] = low_map_membership(
        forecast.mbp
    )


    # ========================================================
    # 2. OXYGENATION
    # ========================================================

    scores["low_spo2"] = (
        low_spo2_membership(
            forecast.spO2
        )
    )


    # ========================================================
    # 3. ARTERIAL BLOOD GAS
    # ========================================================

    pH = labs.get("pH")
    pco2 = labs.get("pCO2")
    po2 = labs.get("pO2")


    # -------------------------
    # Acidemia
    # -------------------------

    if pH is not None:

        scores["acidemia"] = (
            acidemia_membership(pH)
        )

    else:

        scores["acidemia"] = 0.0


    # -------------------------
    # Hypercapnia
    # -------------------------

    if pco2 is not None:

        scores["hypercapnia"] = (
            hypercapnia_membership(pco2)
        )

    else:

        scores["hypercapnia"] = 0.0


    # Raw PaO2

    scores["pao2"] = po2


    # ========================================================
    # 4. VENTILATOR SUPPORT
    # ========================================================

    fio2 = vent.get("set_fio21")
    peep = vent.get("set_peep1")


    scores["fio2"] = fio2
    scores["peep"] = peep


    # -------------------------
    # FiO2
    # -------------------------

    if fio2 is not None:

        scores["fio2_support"] = (
            high_fio2_membership(fio2)
        )

    else:

        scores["fio2_support"] = 0.0


    # -------------------------
    # PEEP
    # -------------------------

    if peep is not None:

        scores["peep_support"] = (
            high_peep_membership(peep)
        )

    else:

        scores["peep_support"] = 0.0


    # ========================================================
    # 5. P/F RATIO
    # ========================================================

    if (
        po2 is not None
        and fio2 is not None
        and fio2 > 0
    ):

        fio2_fraction = fio2 / 100.0

        pf_ratio = (
            po2 /
            fio2_fraction
        )

        scores["pf_ratio"] = pf_ratio

        # Fuzzy memberships

        scores["pf_mild"] = (
            pf_mild_membership(
                pf_ratio
            )
        )

        scores["pf_moderate"] = (
            pf_moderate_membership(
                pf_ratio
            )
        )

        scores["pf_severe"] = (
            pf_severe_membership(
                pf_ratio
            )
        )

    else:

        scores["pf_ratio"] = None

        scores["pf_mild"] = 0.0
        scores["pf_moderate"] = 0.0
        scores["pf_severe"] = 0.0


    # ========================================================
    # 6. NEUROLOGICAL STATUS
    # ========================================================

    gcs_score = gcs.get("gcs")

    if gcs_score is not None:

        scores["gcs_mild"] = (
            gcs_mild_membership(
                gcs_score
            )
        )

        scores["gcs_moderate"] = (
            gcs_moderate_membership(
                gcs_score
            )
        )

        scores["gcs_severe"] = (
            gcs_severe_membership(
                gcs_score
            )
        )

    else:

        scores["gcs_mild"] = 0.0
        scores["gcs_moderate"] = 0.0
        scores["gcs_severe"] = 0.0


    # ========================================================
    # 7. RETURN FUZZY MEMBERSHIPS
    # ========================================================

    return {
        "severity_scores": scores
    }






# ============================================================
# FUZZY SYNDROME DETECTION AGENT
# ============================================================

def syndrome_detection_agent(state: OverallState):

    scores = state.get("severity_scores")

    if scores is None:
        return {}

    # ========================================================
    # HELPER VALUES
    # ========================================================

    low_spo2 = scores.get(
        "low_spo2",
        0.0
    )

    pf_mild = scores.get(
        "pf_mild",
        0.0
    )

    pf_moderate = scores.get(
        "pf_moderate",
        0.0
    )

    pf_severe = scores.get(
        "pf_severe",
        0.0
    )

    fio2_support = scores.get(
        "fio2_support",
        0.0
    )

    peep_support = scores.get(
        "peep_support",
        0.0
    )

    hypercapnia = scores.get(
        "hypercapnia",
        0.0
    )

    acidemia = scores.get(
        "acidemia",
        0.0
    )

    low_map = scores.get(
        "low_map",
        0.0
    )

    high_shock_index = scores.get(
        "high_shock_index",
        0.0
    )

    gcs_moderate = scores.get(
        "gcs_moderate",
        0.0
    )

    gcs_severe = scores.get(
        "gcs_severe",
        0.0
    )


    # ========================================================
    # FUZZY OR HELPER
    # ========================================================

    def fuzzy_or(*values):
        return max(values)


    # ========================================================
    # 1. RESPIRATORY FAILURE
    # ========================================================

    # Rule:
    #
    # IF low SpO2
    # AND moderate/severe oxygenation impairment
    # THEN respiratory failure

    oxygenation_impairment = fuzzy_or(
        pf_moderate,
        pf_severe
    )

    respiratory_rule_1 = min(
        low_spo2,
        oxygenation_impairment
    )


    # --------------------------------------------------------
    # Rule 2
    #
    # IF low SpO2
    # AND high oxygen requirement
    # THEN respiratory failure
    # --------------------------------------------------------

    respiratory_rule_2 = min(
        low_spo2,
        fio2_support
    )


    # --------------------------------------------------------
    # Rule 3
    #
    # IF hypercapnia
    # AND acidemia
    # THEN respiratory failure
    # --------------------------------------------------------

    respiratory_rule_3 = min(
        hypercapnia,
        acidemia
    )


    # --------------------------------------------------------
    # Rule 4
    #
    # IF poor oxygenation
    # AND high PEEP
    # THEN respiratory failure
    # --------------------------------------------------------

    respiratory_rule_4 = min(
        oxygenation_impairment,
        peep_support
    )


    # Aggregate all respiratory rules

    respiratory_failure = fuzzy_or(
        respiratory_rule_1,
        respiratory_rule_2,
        respiratory_rule_3,
        respiratory_rule_4
    )


    # ========================================================
    # 2. SEVERE RESPIRATORY FAILURE
    # ========================================================

    # Rule:
    #
    # IF severe P/F impairment
    # THEN severe respiratory failure

    severe_respiratory_rule_1 = (
        pf_severe
    )


    # --------------------------------------------------------
    # Rule 2
    #
    # IF low SpO2
    # AND moderate P/F impairment
    # AND high oxygen support
    # THEN severe respiratory failure
    # --------------------------------------------------------

    severe_respiratory_rule_2 = min(
        low_spo2,
        pf_moderate,
        fio2_support
    )


    # --------------------------------------------------------
    # Rule 3
    #
    # IF hypercapnia
    # AND acidemia
    # AND low SpO2
    # THEN severe respiratory failure
    # --------------------------------------------------------

    severe_respiratory_rule_3 = min(
        hypercapnia,
        acidemia,
        low_spo2
    )


    # Aggregate

    severe_respiratory_failure = fuzzy_or(
        severe_respiratory_rule_1,
        severe_respiratory_rule_2,
        severe_respiratory_rule_3
    )


    # ========================================================
    # 3. CIRCULATORY SHOCK
    # ========================================================

    # Rule:
    #
    # IF MAP is low
    # AND shock index is high
    # THEN circulatory shock

    circulatory_rule_1 = min(
        low_map,
        high_shock_index
    )


    # --------------------------------------------------------
    # Rule 2
    #
    # Strong low MAP itself contributes to circulatory
    # compromise.
    # --------------------------------------------------------

    circulatory_rule_2 = low_map


    # Aggregate

    circulatory_shock = fuzzy_or(
        circulatory_rule_1,
        circulatory_rule_2
    )


    # ========================================================
    # 4. NEUROLOGICAL FAILURE
    # ========================================================

    # Moderate/severe neurological impairment

    neurological_failure = fuzzy_or(
        gcs_moderate,
        gcs_severe
    )


    # ========================================================
    # 5. SEPSIS
    # ========================================================

    # IMPORTANT:
    #
    # We cannot create a genuinely fuzzy sepsis membership
    # from a Boolean `outcomes.sepsis_outcome`.
    #
    # Therefore this should NOT be mixed into the fuzzy
    # physiological calculations.
    #
    # Keep it separate until you provide a fuzzy sepsis
    # membership/probability from the upstream model.

    outcomes = state.get("outcomes")

    if outcomes is not None:

        sepsis_membership = (
            1.0
            if outcomes.sepsis_outcome
            else 0.0
        )

    else:

        sepsis_membership = 0.0


    # ========================================================
    # 6. RETURN FUZZY SYNDROME MEMBERSHIPS
    # ========================================================

    syndromes = {

        # These are FUZZY VALUES.
        #
        # They are NOT booleans.

        "respiratory_failure":
            respiratory_failure,

        "severe_respiratory_failure":
            severe_respiratory_failure,

        "circulatory_shock":
            circulatory_shock,

        "neurological_failure":
            neurological_failure,

        "sepsis":
            sepsis_membership,


        # Keep the explicit names too

        "respiratory_severity":
            respiratory_failure,

        "severe_respiratory_severity":
            severe_respiratory_failure,

        "circulatory_severity":
            circulatory_shock,

        "neurological_severity":
            neurological_failure,
    }


    return {
        "detected_syndromes": syndromes
    }











# ============================================================
# FUZZY PROTOCOL SELECTION AGENT
# ============================================================

def protocol_selection_agent(state: OverallState):

    syndromes = state.get("detected_syndromes")

    if syndromes is None:
        return {}

    # ========================================================
    # FUZZY INPUTS
    # ========================================================

    respiratory = syndromes.get(
        "respiratory_failure",
        0.0
    )

    severe_respiratory = syndromes.get(
        "severe_respiratory_failure",
        0.0
    )

    circulatory = syndromes.get(
        "circulatory_shock",
        0.0
    )

    sepsis = syndromes.get(
        "sepsis",
        0.0
    )


    # ========================================================
    # ADDITIONAL FUZZY EVIDENCE
    #
    # These come from the severity layer.
    # ========================================================

    scores = state.get(
        "severity_scores"
    ) or {}

    low_spo2 = scores.get(
        "low_spo2",
        0.0
    )

    pf_moderate = scores.get(
        "pf_moderate",
        0.0
    )

    pf_severe = scores.get(
        "pf_severe",
        0.0
    )

    fio2_support = scores.get(
        "fio2_support",
        0.0
    )

    peep_support = scores.get(
        "peep_support",
        0.0
    )

    hypercapnia = scores.get(
        "hypercapnia",
        0.0
    )

    acidemia = scores.get(
        "acidemia",
        0.0
    )


    # ========================================================
    # FUZZY HELPERS
    # ========================================================

    def fuzzy_and(*values):
        """
        Mamdani AND using minimum.
        """
        return min(values)


    def fuzzy_or(*values):
        """
        Mamdani OR using maximum.
        """
        return max(values)


    # ========================================================
    # 1. RESPIRATORY PROTOCOLS
    # ========================================================

    # --------------------------------------------------------
    # INVASIVE SUPPORT
    # --------------------------------------------------------
    #
    # Rule 1:
    #
    # IF severe respiratory failure
    # THEN invasive-support suitability is high
    #

    invasive_rule_1 = severe_respiratory


    # Rule 2:
    #
    # IF severe P/F impairment
    # AND high oxygen support
    # THEN invasive-support suitability is high
    #

    invasive_rule_2 = fuzzy_and(
        pf_severe,
        fio2_support
    )


    # Rule 3:
    #
    # IF severe respiratory failure
    # AND high ventilatory support
    # THEN invasive-support suitability is high
    #

    invasive_rule_3 = fuzzy_and(
        severe_respiratory,
        peep_support
    )


    invasive_support = fuzzy_or(
        invasive_rule_1,
        invasive_rule_2,
        invasive_rule_3
    )


    # --------------------------------------------------------
    # ADVANCED NON-INVASIVE SUPPORT
    # --------------------------------------------------------
    #
    # Rule 1:
    #
    # IF respiratory failure
    # AND moderate oxygenation impairment
    # THEN advanced respiratory support is appropriate
    #

    noninvasive_rule_1 = fuzzy_and(
        respiratory,
        pf_moderate
    )


    # Rule 2:
    #
    # IF low SpO2
    # AND elevated oxygen requirement
    # THEN advanced support is appropriate
    #

    noninvasive_rule_2 = fuzzy_and(
        low_spo2,
        fio2_support
    )


    # Rule 3:
    #
    # IF hypercapnia
    # AND acidemia
    # THEN advanced respiratory support is appropriate
    #

    noninvasive_rule_3 = fuzzy_and(
        hypercapnia,
        acidemia
    )


    advanced_respiratory_support = fuzzy_or(
        noninvasive_rule_1,
        noninvasive_rule_2,
        noninvasive_rule_3
    )


    # --------------------------------------------------------
    # HIGH-FLOW OXYGEN SUPPORT
    # --------------------------------------------------------
    #
    # This is represented separately rather than hiding it
    # inside "advanced_respiratory_support".
    #

    highflow_rule_1 = fuzzy_and(
        respiratory,
        low_spo2
    )


    highflow_rule_2 = fuzzy_and(
        pf_moderate,
        fio2_support
    )


    highflow_support = fuzzy_or(
        highflow_rule_1,
        highflow_rule_2
    )


    # ========================================================
    # 2. CIRCULATION PROTOCOL
    # ========================================================

    # --------------------------------------------------------
    # VASOPRESSOR SUPPORT
    # --------------------------------------------------------
    #
    # IF circulatory shock is high
    # THEN vasopressor suitability increases
    #

    vasopressor_rule_1 = circulatory


    # Additional rule:
    #
    # Strong circulatory abnormality itself reinforces
    # vasopressor consideration.
    #

    low_map = scores.get(
        "low_map",
        0.0
    )

    high_shock_index = scores.get(
        "high_shock_index",
        0.0
    )


    vasopressor_rule_2 = fuzzy_and(
        low_map,
        high_shock_index
    )


    vasopressor_suitability = fuzzy_or(
        vasopressor_rule_1,
        vasopressor_rule_2
    )


    # ========================================================
    # 3. RENAL PROTOCOL
    # ========================================================

    # IMPORTANT:
    #
    # We should not claim that sepsis + shock alone proves
    # CRRT is indicated.
    #
    # Therefore this is a fuzzy "CRRT consideration"
    # membership, not a direct treatment decision.
    #

    # The current state does not contain renal-specific evidence
    # such as potassium, urine output, creatinine trend, severe
    # metabolic acidosis, or fluid overload.
    #
    # Therefore CRRT suitability remains neutral rather than
    # incorrectly inferring a renal indication from sepsis + shock.
    crrt_suitability = 0.0


    # ========================================================
    # 4. RETURN FUZZY PROTOCOL MEMBERSHIPS
    # ========================================================

    protocols = {

        # Respiratory protocol memberships
        "invasive_support":
            invasive_support,

        "noninvasive_support":
            advanced_respiratory_support,

        "highflow_support":
            highflow_support,

        # Circulation
        "vasopressor_support":
            vasopressor_suitability,

        # Renal
        "crrt_support":
            crrt_suitability,
    }


    return {
        "selected_protocols": protocols
    }



# ============================================================
# FUZZY ESCALATION AGENT
# ============================================================

def escalation_agent(state: OverallState):

    protocols = state.get(
        "selected_protocols"
    )

    current = state.get(
        "original_interventions"
    )

    if protocols is None or current is None:
        return {}


    # ========================================================
    # FUZZY HELPERS
    # ========================================================

    def fuzzy_and(*values):
        """
        Fuzzy AND = minimum.
        """
        return min(values)


    def fuzzy_or(*values):
        """
        Fuzzy OR = maximum.
        """
        return max(values)


    def fuzzy_not(value):
        """
        Fuzzy NOT / complement.
        """
        return 1.0 - value


    # ========================================================
    # CURRENT INTERVENTIONS
    # ========================================================

    current_invasive = float(
        current.get("invasive", 0)
    )

    current_noninvasive = float(
        current.get("noninvasive", 0)
    )

    current_highflow = float(
        current.get("highflow", 0)
    )

    current_vasopressor = float(
        current.get("vasopressor", 0)
    )

    current_crrt = float(
        current.get("crrt", 0)
    )


    # ========================================================
    # 1. RESPIRATORY ESCALATION
    # ========================================================

    invasive_support = protocols.get(
        "invasive_support",
        0.0
    )

    noninvasive_support = protocols.get(
        "noninvasive_support",
        0.0
    )

    highflow_support = protocols.get(
        "highflow_support",
        0.0
    )


    # --------------------------------------------------------
    # INVASIVE ESCALATION
    # --------------------------------------------------------
    #
    # IF invasive support suitability is high
    # AND invasive ventilation is not already active
    # THEN invasive escalation need is high
    #

    invasive_escalation = fuzzy_and(
        invasive_support,
        fuzzy_not(current_invasive)
    )


    # --------------------------------------------------------
    # NON-INVASIVE ESCALATION
    # --------------------------------------------------------
    #
    # IF non-invasive support is suitable
    # AND invasive ventilation is absent
    # AND NIV is not already active
    #

    noninvasive_escalation = fuzzy_and(
        noninvasive_support,
        fuzzy_not(current_invasive),
        fuzzy_not(current_noninvasive)
    )


    # --------------------------------------------------------
    # HIGH-FLOW ESCALATION
    # --------------------------------------------------------
    #
    # IF high-flow support is suitable
    # AND invasive ventilation is absent
    # AND high-flow is not already active
    #

    highflow_escalation = fuzzy_and(
        highflow_support,
        fuzzy_not(current_invasive),
        fuzzy_not(current_highflow)
    )


    # ========================================================
    # 2. CIRCULATORY ESCALATION
    # ========================================================

    vasopressor_support = protocols.get(
        "vasopressor_support",
        0.0
    )


    vasopressor_escalation = fuzzy_and(
        vasopressor_support,
        fuzzy_not(current_vasopressor)
    )


    # ========================================================
    # 3. RENAL ESCALATION
    # ========================================================

    crrt_support = protocols.get(
        "crrt_support",
        0.0
    )


    crrt_escalation = fuzzy_and(
        crrt_support,
        fuzzy_not(current_crrt)
    )


    # ========================================================
    # 4. RETURN FUZZY ESCALATION DEGREES
    # ========================================================

    escalation = {

        "invasive":
            invasive_escalation,

        "noninvasive":
            noninvasive_escalation,

        "highflow":
            highflow_escalation,

        "vasopressor":
            vasopressor_escalation,

        "crrt":
            crrt_escalation,
    }


    return {
        "escalation_decision": escalation
    }



# ============================================================
# FUZZY CONTROL UTILITIES
# ============================================================

def fuzzy_and(*values):
    """
    Mamdani fuzzy AND.
    Uses minimum operator.
    """
    values = [
        max(0.0, min(1.0, float(v)))
        for v in values
    ]

    return min(values)


def fuzzy_or(*values):
    """
    Mamdani fuzzy OR.
    Uses maximum operator.
    """
    values = [
        max(0.0, min(1.0, float(v)))
        for v in values
    ]

    return max(values)


def fuzzy_not(value):
    """
    Fuzzy complement.
    """
    value = max(0.0, min(1.0, float(value)))

    return 1.0 - value


def clamp01(value):
    """
    Keep value inside [0, 1].
    """
    return max(0.0, min(1.0, float(value)))






# ============================================================
# FUZZY WEANING AGENT
# ============================================================

def weaning_agent(state: OverallState):

    forecast = state.get("forecasted_vitals")
    labs = state.get("original_labs") or {}
    gcs = state.get("original_gcs") or {}
    vent = state.get("original_ventilator") or {}
    current = state.get("original_interventions") or {}
    scores = state.get("severity_scores") or {}

    if forecast is None:
        return {}

    # ========================================================
    # CURRENT INTERVENTIONS
    # ========================================================

    current_invasive = float(
        current.get("invasive", 0)
    )

    current_noninvasive = float(
        current.get("noninvasive", 0)
    )

    current_highflow = float(
        current.get("highflow", 0)
    )

    current_vasopressor = float(
        current.get("vasopressor", 0)
    )

    current_crrt = float(
        current.get("crrt", 0)
    )

    # If absolutely no support is active, there is nothing
    # to wean.

    total_support = fuzzy_or(
        current_invasive,
        current_noninvasive,
        current_highflow,
        current_vasopressor,
        current_crrt
    )

    if total_support == 0.0:
        return {
            "weaning_recommendation": None
        }

    # ========================================================
    # PHYSIOLOGICAL MEMBERSHIPS
    # ========================================================

    low_spo2 = scores.get(
        "low_spo2",
        0.0
    )

    low_map = scores.get(
        "low_map",
        0.0
    )

    hypercapnia = scores.get(
        "hypercapnia",
        0.0
    )

    acidemia = scores.get(
        "acidemia",
        0.0
    )

    # Convert abnormality membership into stability membership.

    oxygenation_stability = fuzzy_not(
        low_spo2
    )

    hemodynamic_stability = fuzzy_not(
        low_map
    )

    ventilation_stability = fuzzy_not(
        hypercapnia
    )

    acid_base_stability = fuzzy_not(
        acidemia
    )

    # ========================================================
    # RAW VALUES
    # ========================================================

    spo2 = forecast.spO2
    mbp = forecast.mbp

    pco2 = labs.get("pCO2")
    ph = labs.get("pH")

    gcs_score = gcs.get("gcs")

    fio2 = vent.get("set_fio21")
    peep = vent.get("set_peep1")

    # ========================================================
    # ADDITIONAL MEMBERSHIPS
    # ========================================================

    # Oxygen requirement:
    #
    # Lower FiO2 = greater readiness.
    #
    # This is represented continuously rather than
    # using "FiO2 <= 40".

    if fio2 is not None:

        fio2_readiness = trapezoid_membership(
            fio2,
            21,
            21,
            30,
            40
        )

    else:

        fio2_readiness = 0.0


    # PEEP readiness

    if peep is not None:

        peep_readiness = trapezoid_membership(
            peep,
            3,
            3,
            5,
            8
        )

    else:

        peep_readiness = 0.0


    # GCS readiness

    if gcs_score is not None:

        neurological_readiness = trapezoid_membership(
            gcs_score,
            10,
            12,
            15,
            15
        )

    else:

        neurological_readiness = 0.0


    # ========================================================
    # 1. INVASIVE VENTILATION → SBT READINESS
    # ========================================================

    # The patient needs a combination of:
    #
    # oxygenation stability
    # hemodynamic stability
    # acid/base stability
    # neurological readiness
    # reasonable oxygen/PEEP requirement

    sbt_rule_1 = fuzzy_and(
        oxygenation_stability,
        hemodynamic_stability,
        acid_base_stability,
        neurological_readiness
    )

    sbt_rule_2 = fuzzy_and(
        oxygenation_stability,
        fio2_readiness,
        peep_readiness
    )

    sbt_rule_3 = fuzzy_and(
        ventilation_stability,
        acid_base_stability,
        hemodynamic_stability
    )

    sbt_readiness = fuzzy_or(
        sbt_rule_1,
        sbt_rule_2,
        sbt_rule_3
    )


    # ========================================================
    # 2. EXTUBATION READINESS
    # ========================================================

    # IMPORTANT:
    #
    # We do NOT automatically say:
    #
    # invasive = 0
    # NIV = 1
    #
    # because an SBT result is not available in the state.
    #
    # We instead produce a fuzzy "extubation readiness"
    # score.

    extubation_readiness = fuzzy_and(
        sbt_readiness,
        neurological_readiness,
        hemodynamic_stability,
        oxygenation_stability
    )


    # ========================================================
    # 3. NIV WEANING
    # ========================================================

    niv_weaning_rule_1 = fuzzy_and(
        oxygenation_stability,
        ventilation_stability,
        hemodynamic_stability
    )

    niv_weaning_rule_2 = fuzzy_and(
        fio2_readiness,
        acid_base_stability
    )

    niv_weaning_readiness = fuzzy_or(
        niv_weaning_rule_1,
        niv_weaning_rule_2
    )


    # ========================================================
    # 4. HFNC WEANING
    # ========================================================

    highflow_weaning_rule_1 = fuzzy_and(
        oxygenation_stability,
        hemodynamic_stability
    )

    highflow_weaning_rule_2 = fuzzy_and(
        fio2_readiness,
        ventilation_stability
    )

    highflow_weaning_readiness = fuzzy_or(
        highflow_weaning_rule_1,
        highflow_weaning_rule_2
    )


    # ========================================================
    # 5. VASOPRESSOR WEANING
    # ========================================================

    # Better circulation = lower MAP abnormality.
    #
    # We don't use "MAP >= 65".

    vasopressor_weaning = hemodynamic_stability


    # ========================================================
    # 6. CRRT WEANING
    # ========================================================

    # DO NOT infer CRRT discontinuation from sepsis resolution.
    #
    # Your current state lacks sufficient renal recovery
    # information.
    #
    # Therefore expose a fuzzy readiness value only if
    # renal evidence exists.

    renal_recovery = scores.get(
        "renal_recovery",
        0.0
    )

    crrt_weaning = renal_recovery


    # ========================================================
    # 7. RETURN
    # ========================================================

    recommendation = {

        # Invasive liberation
        "sbt_readiness":
            sbt_readiness,

        "extubation_readiness":
            extubation_readiness,

        # NIV
        "niv_weaning_readiness":
            niv_weaning_readiness,

        # HFNC
        "highflow_weaning_readiness":
            highflow_weaning_readiness,

        # Vasopressor
        "vasopressor_weaning_readiness":
            vasopressor_weaning,

        # CRRT
        "crrt_weaning_readiness": crrt_weaning
    }

    return {
        "weaning_recommendation": recommendation
    }




# ============================================================
# FUZZY TREATMENT AGENT
# ============================================================

def treatment_agent(state: OverallState):

    escalation = state.get(
        "escalation_decision"
    ) or {}

    weaning = state.get(
        "weaning_recommendation"
    ) or {}

    current = state.get(
        "original_interventions"
    ) or {}

    # ========================================================
    # CURRENT SUPPORT
    # ========================================================

    current_invasive = float(
        current.get("invasive", 0)
    )

    current_noninvasive = float(
        current.get("noninvasive", 0)
    )

    current_highflow = float(
        current.get("highflow", 0)
    )

    current_vasopressor = float(
        current.get("vasopressor", 0)
    )

    current_crrt = float(
        current.get("crrt", 0)
    )


    # ========================================================
    # ESCALATION MEMBERSHIPS
    # ========================================================

    invasive = escalation.get(
        "invasive",
        0.0
    )

    noninvasive = escalation.get(
        "noninvasive",
        0.0
    )

    highflow = escalation.get(
        "highflow",
        0.0
    )

    vasopressor = escalation.get(
        "vasopressor",
        0.0
    )

    crrt = escalation.get(
        "crrt",
        0.0
    )


    # ========================================================
    # WEANING MEMBERSHIPS
    # ========================================================

    sbt_readiness = weaning.get(
        "sbt_readiness",
        0.0
    )

    extubation_readiness = weaning.get(
        "extubation_readiness",
        0.0
    )

    niv_weaning = weaning.get(
        "niv_weaning_readiness",
        0.0
    )

    highflow_weaning = weaning.get(
        "highflow_weaning_readiness",
        0.0
    )

    vasopressor_weaning = weaning.get(
        "vasopressor_weaning_readiness",
        0.0
    )

    crrt_weaning = weaning.get(
        "crrt_weaning_readiness",
        0.0
    )


    # ========================================================
    # FUZZY TREATMENT OUTPUTS
    # ========================================================

    # Invasive support is reduced by extubation readiness.

    invasive_recommendation = fuzzy_and(
        invasive,
        fuzzy_not(extubation_readiness)
    )


    # NIV support is reduced if invasive support is strongly
    # required or if the patient is ready to wean.

    noninvasive_recommendation = fuzzy_and(
        noninvasive,
        fuzzy_not(invasive),
        fuzzy_not(niv_weaning)
    )


    # HFNC is reduced if invasive support is strongly required.

    highflow_recommendation = fuzzy_and(
        highflow,
        fuzzy_not(invasive),
        fuzzy_not(highflow_weaning)
    )


    # Vasopressor continuation/escalation.

    vasopressor_recommendation = fuzzy_and(
        vasopressor,
        fuzzy_not(vasopressor_weaning)
    )


    # CRRT.

    crrt_recommendation = fuzzy_and(
        crrt,
        fuzzy_not(crrt_weaning)
    )


    # ========================================================
    # LIBERATION / CONTINUATION SIGNALS
    # ========================================================

    liberation = {

        "sbt_readiness":
            sbt_readiness,

        "extubation_readiness":
            extubation_readiness,

        "niv_weaning":
            niv_weaning,

        "highflow_weaning":
            highflow_weaning,

        "vasopressor_weaning":
            vasopressor_weaning,

        "crrt_weaning":
            crrt_weaning,
    }


    # ========================================================
    # FINAL FUZZY RECOMMENDATIONS
    # ========================================================

    recommended = {

        "invasive":
            invasive_recommendation,

        "noninvasive":
            noninvasive_recommendation,

        "highflow":
            highflow_recommendation,

        "vasopressor":
            vasopressor_recommendation,

        "crrt":
            crrt_recommendation,
    }


    return {
        "recommended_interventions": recommended,
        "liberation_scores": liberation
    }







# ============================================================
# FUZZY VENTILATOR SETTING AGENT
# ============================================================

def ventilator_setting_agent(state: OverallState):

    static = state.get(
        "original_static"
    ) or {}

    labs = state.get(
        "original_labs"
    ) or {}

    forecast = state.get(
        "forecasted_vitals"
    )

    vent = state.get(
        "original_ventilator"
    ) or {}

    interventions = state.get(
        "recommended_interventions"
    ) or {}

    if forecast is None:
        return {}

    # ========================================================
    # FUZZY INTERVENTION STRENGTHS
    # ========================================================

    invasive = interventions.get(
        "invasive",
        0.0
    )

    noninvasive = interventions.get(
        "noninvasive",
        0.0
    )

    highflow = interventions.get(
        "highflow",
        0.0
    )


    # ========================================================
    # PATIENT DATA
    # ========================================================

    pbw = float(
        static.get("pbw_kg", 0)
    )

    spo2 = forecast.spO2

    pco2 = labs.get(
        "pCO2"
    )

    ph = labs.get(
        "pH"
    )

    current_fio2 = float(
        vent.get("set_fio21", 21)
    )

    current_peep = float(
        vent.get("set_peep1", 5)
    )

    current_rr = float(
        vent.get("set_rr1", 16)
    )


    # ========================================================
    # SPO2 FUZZY MEMBERSHIPS
    # ========================================================

    spo2_very_low = trapezoid_membership(
        spo2,
        80,
        85,
        88,
        90
    )

    spo2_low = trapezoid_membership(
        spo2,
        88,
        90,
        92,
        94
    )

    spo2_target = trapezoid_membership(
        spo2,
        90,
        92,
        96,
        97
    )

    spo2_high = trapezoid_membership(
        spo2,
        96,
        97,
        100,
        100
    )


    # ========================================================
    # FIO2 CHANGE FUZZY RULES
    # ========================================================

    # Increase strongly when SpO2 is very low.

    fio2_increase_strong = spo2_very_low

    # Increase moderately when SpO2 is low.

    fio2_increase_moderate = spo2_low

    # Maintain when SpO2 is in target region.

    fio2_maintain = spo2_target

    # Decrease when SpO2 is high.

    fio2_decrease = spo2_high


    # ========================================================
    # FUZZY DEFUZZIFICATION FOR FIO2 CHANGE
    # ========================================================

    numerator = (
        fio2_increase_strong * 15
        + fio2_increase_moderate * 8
        + fio2_maintain * 0
        + fio2_decrease * -5
    )

    denominator = (
        fio2_increase_strong
        + fio2_increase_moderate
        + fio2_maintain
        + fio2_decrease
    )

    if denominator > 0:

        fio2_change = (
            numerator /
            denominator
        )

    else:

        fio2_change = 0.0


    # Keep the resulting physical setting within
    # the ventilator's valid percentage range.

    recommended_fio2 = max(
        21.0,
        min(
            100.0,
            current_fio2 + fio2_change
        )
    )


    # ========================================================
    # PEEP FUZZY CONTROL
    # ========================================================

    # Low P/F increases the fuzzy need for PEEP.
    #
    # We derive this from the existing severity memberships.

    scores = state.get(
        "severity_scores"
    ) or {}

    pf_severe = scores.get(
        "pf_severe",
        0.0
    )

    pf_moderate = scores.get(
        "pf_moderate",
        0.0
    )

    high_peep_need = max(
        pf_severe,
        pf_moderate
    )


    # Defuzzified PEEP change.

    peep_change = (
        high_peep_need * 4.0
    )


    recommended_peep = max(
        5.0,
        min(
            20.0,
            current_peep + peep_change
        )
    )


    # ========================================================
    # RESPIRATORY RATE FUZZY CONTROL
    # ========================================================

    if pco2 is not None:

        pco2_high = trapezoid_membership(
            pco2,
            45,
            50,
            55,
            60
        )

        pco2_normal = trapezoid_membership(
            pco2,
            35,
            38,
            42,
            45
        )

        pco2_low = trapezoid_membership(
            pco2,
            25,
            30,
            35,
            38
        )

    else:

        pco2_high = 0.0
        pco2_normal = 0.0
        pco2_low = 0.0


    # Higher PaCO2 → greater RR increase.
    #
    # Lower PaCO2 → RR reduction.

    rr_numerator = (
        pco2_high * 4
        + pco2_normal * 0
        + pco2_low * -2
    )

    rr_denominator = (
        pco2_high
        + pco2_normal
        + pco2_low
    )

    if rr_denominator > 0:

        rr_change = (
            rr_numerator /
            rr_denominator
        )

    else:

        rr_change = 0.0


    recommended_rr = max(
        10.0,
        min(
            35.0,
            current_rr + rr_change
        )
    )


    # ========================================================
    # TIDAL VOLUME
    # ========================================================

    # For invasive ventilation:
    #
    # Start around 6 mL/kg PBW.
    #
    # Current SSC guidance:
    # 6 mL/kg for ARDS.
    # 6-8 mL/kg IBW for sepsis-associated hypoxemic
    # respiratory failure without ARDS.
    #
    # We use 6 mL/kg as the conservative fuzzy target.

    target_tv_per_kg = 6.0

    target_tv = (
        pbw *
        target_tv_per_kg
    )


    # ========================================================
    # PRESSURE SUPPORT
    # ========================================================

    # For NIV, pressure support is treated as a fuzzy
    # support requirement rather than a fixed "10".

    pressure_support_need = max(
        pf_moderate,
        pf_severe,
        pco2_high if pco2 is not None else 0.0
    )

    pressure_support = (
        5.0
        +
        pressure_support_need * 10.0
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    # --------------------------------------------------------
    # SELECT MODE BY FUZZY MEMBERSHIP
    # --------------------------------------------------------
    #
    # Select the respiratory mode with the strongest fuzzy
    # suitability rather than using code-order priority.

    mode_scores = {
        "HFNC": clamp01(highflow),
        "NIV": clamp01(noninvasive),
        "INVASIVE": clamp01(invasive),
    }

    selected_mode, selected_strength = max(
        mode_scores.items(),
        key=lambda item: item[1]
    )

    if selected_strength <= 0.0:
        return {
            "recommended_ventilator_changes": None
        }


    # --------------------------------------------------------
    # HFNC
    # --------------------------------------------------------

    if selected_mode == "HFNC":
        return {
            "recommended_ventilator_changes": {

                "mode": "HFNC",

                "hfnc_flow_rate":
                    40.0 + (
                        selected_strength * 20.0
                    ),

                "set_fio21":
                    recommended_fio2,

                "fuzzy_control_strength":
                    selected_strength
            }
        }


    # --------------------------------------------------------
    # NIV
    # --------------------------------------------------------

    if selected_mode == "NIV":
        return {
            "recommended_ventilator_changes": {

                "mode": "NIV",

                "set_fio21":
                    recommended_fio2,

                "set_peep1":
                    recommended_peep,

                "pressure_support":
                    pressure_support,

                "fuzzy_control_strength":
                    selected_strength
            }
        }


    # --------------------------------------------------------
    # INVASIVE
    # --------------------------------------------------------

    # Conservative pressure-control target derived from the
    # fuzzy PEEP and pressure-support requirements.
    #
    # This is a recommendation signal, NOT a prediction of
    # measured peak/plateau pressure.

    pressure_control_level = (
        recommended_peep +
        pressure_support
    )

    return {
        "recommended_ventilator_changes": {

            "mode":
                "INVASIVE",

            "set_tv1":
                target_tv,

            "total_tv":
                target_tv,

            "set_fio21":
                recommended_fio2,

            "set_peep1":
                recommended_peep,

            "total_peep":
                recommended_peep,

            "set_rr1":
                recommended_rr,

            "total_rr":
                recommended_rr,

            "set_ie_ratio1":
                2.0,

            "set_pc1":
                pressure_control_level,

            "rr":
                recommended_rr,

            "fuzzy_control_strength":
                selected_strength
        }
    }





def action_agent(state: OverallState):
    return {}


# ─────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────
def build_graph():
    builder = StateGraph(OverallState)
    builder.add_node("start_agent", start_agent)
    builder.add_node("forecast_agent", forecast_agent)
    builder.add_node("prediction_agent", prediction_agent)
    builder.add_node("summarization_agent", summarization_agent)
    builder.add_node("action_agent", action_agent)
    builder.add_node("severity_scoring_agent", severity_scoring_agent)
    builder.add_node("syndrome_detection_agent", syndrome_detection_agent)
    builder.add_node("protocol_selection_agent", protocol_selection_agent)
    builder.add_node("escalation_agent", escalation_agent)
    builder.add_node("treatment_agent", treatment_agent)
    builder.add_node("ventilator_setting_agent", ventilator_setting_agent)
    builder.add_node("weaning_agent", weaning_agent)

    builder.add_edge(START, "start_agent")
    builder.add_edge("start_agent", "forecast_agent")
    builder.add_edge("start_agent", "prediction_agent")

    # Fan-in: summarization waits for both forecast and prediction to complete
    # so forecasted_vitals and outcomes are populated before it runs
    builder.add_edge("forecast_agent", "summarization_agent")
    builder.add_edge("prediction_agent", "summarization_agent")
    builder.add_edge("summarization_agent", "action_agent")

    builder.add_edge("action_agent", "severity_scoring_agent")
    builder.add_edge("severity_scoring_agent", "syndrome_detection_agent")
    builder.add_edge("syndrome_detection_agent", "protocol_selection_agent")
    builder.add_edge("syndrome_detection_agent", "weaning_agent")
    builder.add_edge("protocol_selection_agent", "escalation_agent")
    builder.add_edge("weaning_agent", "escalation_agent")
    builder.add_edge("escalation_agent", "treatment_agent")
    builder.add_edge("treatment_agent", "ventilator_setting_agent")
    builder.add_edge("ventilator_setting_agent", END)

    return builder.compile()


graph = build_graph()