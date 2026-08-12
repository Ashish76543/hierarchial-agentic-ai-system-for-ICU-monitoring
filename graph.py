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
    liberation_scores: Optional[Dict[str, float]]
    final_action: Optional[Dict[str, object]]
    decision_trace: Optional[list]
    treatment_change_detected: Optional[bool]
    ventilation_change_detected: Optional[bool]


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

    Normal trapezoid:
        a ---- b ===== c ---- d
        0      1       1      0

    Left shoulder (a == b):
        x <= b -> 1
        x >= d -> 0

    Right shoulder (c == d):
        x <= a -> 0
        x >= c -> 1

    All returned values are clamped to [0, 1].
    """

    if x is None:
        return 0.0

    x = float(x)

    if a > b or b > c or c > d:
        raise ValueError(
            f"Invalid trapezoid parameters: {(a, b, c, d)}"
        )

    # Left shoulder
    if a == b:
        if x <= b:
            return 1.0
        if x >= d:
            return 0.0
        if x <= c:
            return 1.0
        return clamp01((d - x) / (d - c))

    # Right shoulder
    if c == d:
        if x <= a:
            return 0.0
        if x >= c:
            return 1.0
        if x <= b:
            return clamp01((x - a) / (b - a))
        return 1.0

    # Normal trapezoid
    if x <= a or x >= d:
        return 0.0

    if a < x < b:
        return clamp01((x - a) / (b - a))

    if b <= x <= c:
        return 1.0

    return clamp01((d - x) / (d - c))


# ============================================================
# HEMODYNAMIC MEMBERSHIP
# ============================================================

def low_map_membership(mbp):
    """
    Low MAP.

    65 mmHg is the important clinical anchor used here.
    The lower transition points are fuzzy-model parameters.
    """

    return trapezoid_membership(
        mbp,
        55,
        55,
        60,
        65
    )


def high_shock_index_membership(shock_index):
    """
    High Shock Index.

    Uses a right shoulder so an extremely high Shock Index
    cannot incorrectly return to zero.
    """

    return trapezoid_membership(
        shock_index,
        0.7,
        0.9,
        1.0,
        1.0
    )


# ============================================================
# OXYGENATION MEMBERSHIP
# ============================================================

def low_spo2_membership(spo2):
    """
    Low SpO2.

    Uses a left shoulder so very low SpO2 remains strongly
    abnormal rather than falling back toward zero.
    """

    return trapezoid_membership(
        spo2,
        88,
        88,
        92,
        94
    )


# ============================================================
# ACID-BASE MEMBERSHIP
# ============================================================

def acidemia_membership(ph):
    """
    Acidemia.

    pH < 7.35 is represented as increasing acidemia membership.
    """

    return trapezoid_membership(
        ph,
        7.20,
        7.20,
        7.30,
        7.35
    )


def hypercapnia_membership(pco2):
    """
    Hypercapnia.

    Uses a right shoulder so very high PaCO2 remains highly
    abnormal instead of returning to zero.
    """

    return trapezoid_membership(
        pco2,
        45,
        50,
        60,
        60
    )


# ============================================================
# VENTILATOR SUPPORT MEMBERSHIP
# ============================================================

def high_fio2_membership(fio2):
    """
    High FiO2 requirement.

    Uses a right shoulder because higher FiO2 should not reduce
    the membership of "high oxygen requirement".
    """

    return trapezoid_membership(
        fio2,
        40,
        60,
        60,
        60
    )


def high_peep_membership(peep):
    """
    High PEEP requirement.

    Uses a right shoulder so very high PEEP remains high.
    """

    return trapezoid_membership(
        peep,
        5,
        8,
        10,
        10
    )


# ============================================================
# P/F RATIO MEMBERSHIPS
# ============================================================

def pf_mild_membership(pf_ratio):
    """
    Mild oxygenation impairment.

    Fuzzy overlap is intentional around the clinical category
    boundaries.
    """

    return trapezoid_membership(
        pf_ratio,
        200,
        225,
        300,
        350
    )


def pf_moderate_membership(pf_ratio):
    """
    Moderate oxygenation impairment.
    """

    return trapezoid_membership(
        pf_ratio,
        100,
        125,
        200,
        225
    )


def pf_severe_membership(pf_ratio):
    """
    Severe oxygenation impairment.

    Left shoulder prevents severe membership from becoming zero
    at extremely low P/F values.
    """

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
    """
    Mild neurological impairment: approximately GCS 13-15.
    """

    return trapezoid_membership(
        gcs,
        12,
        13,
        15,
        15
    )


def gcs_moderate_membership(gcs):
    """
    Moderate neurological impairment: approximately GCS 9-12.
    """

    return trapezoid_membership(
        gcs,
        8,
        9,
        12,
        13
    )


def gcs_severe_membership(gcs):
    """
    Severe neurological impairment: approximately GCS 3-8.

    The upper transition is kept beyond 8 so GCS=8 still has
    full severe membership rather than incorrectly becoming zero.
    """

    return trapezoid_membership(
        gcs,
        3,
        3,
        8,
        9
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

    scores["low_map"] = low_map_membership(
        forecast.mbp
    )

    # ========================================================
    # 2. OXYGENATION
    # ========================================================

    scores["low_spo2"] = low_spo2_membership(
        forecast.spO2
    )

    # ========================================================
    # 3. ARTERIAL BLOOD GAS
    # ========================================================

    pH = labs.get("pH")
    pco2 = labs.get("pCO2")
    po2 = labs.get("pO2")

    scores["acidemia"] = (
        acidemia_membership(pH)
        if pH is not None
        else 0.0
    )

    scores["hypercapnia"] = (
        hypercapnia_membership(pco2)
        if pco2 is not None
        else 0.0
    )

    scores["pao2"] = po2

    # ========================================================
    # 4. VENTILATOR SUPPORT
    # ========================================================

    fio2 = normalized_fio2(get_numeric(vent, "set_fio21", "fio2", "FiO2"))
    peep = get_numeric(vent, "set_peep1", "peep", "PEEP")

    scores["fio2"] = fio2
    scores["peep"] = peep

    scores["fio2_support"] = (
        high_fio2_membership(fio2)
        if fio2 is not None
        else 0.0
    )

    scores["peep_support"] = (
        high_peep_membership(peep)
        if peep is not None
        else 0.0
    )

    # ========================================================
    # 5. P/F RATIO
    # ========================================================

    if (
        po2 is not None
        and fio2 is not None
        and fio2 > 0
    ):

        fio2_fraction = fio2 / 100.0

        pf_ratio = po2 / fio2_fraction

        scores["pf_ratio"] = pf_ratio

        scores["pf_mild"] = pf_mild_membership(
            pf_ratio
        )

        scores["pf_moderate"] = pf_moderate_membership(
            pf_ratio
        )

        scores["pf_severe"] = pf_severe_membership(
            pf_ratio
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

        scores["gcs_mild"] = gcs_mild_membership(
            gcs_score
        )

        scores["gcs_moderate"] = gcs_moderate_membership(
            gcs_score
        )

        scores["gcs_severe"] = gcs_severe_membership(
            gcs_score
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

    low_spo2 = clamp01(scores.get("low_spo2", 0.0))
    pf_moderate = clamp01(scores.get("pf_moderate", 0.0))
    pf_severe = clamp01(scores.get("pf_severe", 0.0))
    fio2_support = clamp01(scores.get("fio2_support", 0.0))
    peep_support = clamp01(scores.get("peep_support", 0.0))
    hypercapnia = clamp01(scores.get("hypercapnia", 0.0))
    acidemia = clamp01(scores.get("acidemia", 0.0))
    low_map = clamp01(scores.get("low_map", 0.0))
    high_shock_index = clamp01(scores.get("high_shock_index", 0.0))
    gcs_moderate = clamp01(scores.get("gcs_moderate", 0.0))
    gcs_severe = clamp01(scores.get("gcs_severe", 0.0))

    # --------------------------------------------------------
    # Respiratory failure
    # --------------------------------------------------------
    oxygenation_impairment = fuzzy_or(
        pf_moderate,
        pf_severe
    )

    respiratory_rule_1 = fuzzy_and(
        low_spo2,
        oxygenation_impairment
    )

    respiratory_rule_2 = fuzzy_and(
        low_spo2,
        fio2_support
    )

    respiratory_rule_3 = fuzzy_and(
        hypercapnia,
        acidemia
    )

    respiratory_rule_4 = fuzzy_and(
        oxygenation_impairment,
        peep_support
    )

    respiratory_failure = fuzzy_or(
        respiratory_rule_1,
        respiratory_rule_2,
        respiratory_rule_3,
        respiratory_rule_4
    )

    # --------------------------------------------------------
    # Severe respiratory failure
    #
    # A severe P/F pattern is not used completely by itself.
    # It is combined with evidence of respiratory support/
    # oxygenation compromise.
    # --------------------------------------------------------
    severe_rule_1 = fuzzy_and(
        pf_severe,
        fuzzy_or(
            low_spo2,
            fio2_support,
            peep_support
        )
    )

    severe_rule_2 = fuzzy_and(
        low_spo2,
        pf_moderate,
        fio2_support
    )

    severe_rule_3 = fuzzy_and(
        hypercapnia,
        acidemia,
        low_spo2
    )

    severe_respiratory_failure = fuzzy_or(
        severe_rule_1,
        severe_rule_2,
        severe_rule_3
    )

    # --------------------------------------------------------
    # Circulatory compromise
    # --------------------------------------------------------
    circulatory_rule_1 = fuzzy_and(
        low_map,
        high_shock_index
    )

    circulatory_rule_2 = low_map

    circulatory_shock = fuzzy_or(
        circulatory_rule_1,
        circulatory_rule_2
    )

    # --------------------------------------------------------
    # Neurological impairment
    # --------------------------------------------------------
    neurological_failure = fuzzy_or(
        gcs_moderate,
        gcs_severe
    )

    # --------------------------------------------------------
    # Sepsis
    #
    # The current outcome model only supplies a Boolean
    # prediction, so this remains a 0/1 value. It is not
    # pretending that a Boolean prediction is a physiological
    # fuzzy membership.
    # --------------------------------------------------------
    outcomes = state.get("outcomes")

    if outcomes is not None:
        sepsis_membership = (
            1.0 if outcomes.sepsis_outcome else 0.0
        )
    else:
        sepsis_membership = 0.0

    syndromes = {
        "respiratory_failure": clamp01(respiratory_failure),
        "severe_respiratory_failure": clamp01(
            severe_respiratory_failure
        ),
        "circulatory_shock": clamp01(circulatory_shock),
        "neurological_failure": clamp01(neurological_failure),
        "sepsis": clamp01(sepsis_membership),

        "respiratory_severity": clamp01(respiratory_failure),
        "severe_respiratory_severity": clamp01(
            severe_respiratory_failure
        ),
        "circulatory_severity": clamp01(circulatory_shock),
        "neurological_severity": clamp01(neurological_failure),
    }

    return {
        "detected_syndromes": syndromes
    }

def protocol_selection_agent(state: OverallState):
    syndromes = state.get("detected_syndromes")
    if syndromes is None:
        return {}

    respiratory = clamp01(
        syndromes.get("respiratory_failure", 0.0)
    )
    severe_respiratory = clamp01(
        syndromes.get("severe_respiratory_failure", 0.0)
    )
    circulatory = clamp01(
        syndromes.get("circulatory_shock", 0.0)
    )

    scores = state.get("severity_scores") or {}

    low_spo2 = clamp01(scores.get("low_spo2", 0.0))
    pf_moderate = clamp01(scores.get("pf_moderate", 0.0))
    pf_severe = clamp01(scores.get("pf_severe", 0.0))
    fio2_support = clamp01(scores.get("fio2_support", 0.0))
    peep_support = clamp01(scores.get("peep_support", 0.0))
    hypercapnia = clamp01(scores.get("hypercapnia", 0.0))
    acidemia = clamp01(scores.get("acidemia", 0.0))
    low_map = clamp01(scores.get("low_map", 0.0))
    high_shock_index = clamp01(
        scores.get("high_shock_index", 0.0)
    )

    # --------------------------------------------------------
    # Respiratory protocol suitability
    # --------------------------------------------------------
    invasive_support = fuzzy_or(
        severe_respiratory,
        fuzzy_and(pf_severe, fio2_support),
        fuzzy_and(severe_respiratory, peep_support)
    )

    noninvasive_support = fuzzy_or(
        fuzzy_and(respiratory, pf_moderate),
        fuzzy_and(low_spo2, fio2_support),
        fuzzy_and(hypercapnia, acidemia)
    )

    highflow_support = fuzzy_or(
        fuzzy_and(respiratory, low_spo2),
        fuzzy_and(pf_moderate, fio2_support)
    )

    # These are fuzzy suitability scores, not simultaneous orders.
    # The escalation agent resolves the competing respiratory modes.
    # --------------------------------------------------------
    # Circulation
    # --------------------------------------------------------
    vasopressor_support = fuzzy_or(
        circulatory,
        fuzzy_and(low_map, high_shock_index)
    )

    # --------------------------------------------------------
    # Renal
    #
    # No renal-specific continuous inputs exist in this state.
    # Therefore CRRT must remain zero rather than being inferred
    # from sepsis or shock alone.
    # --------------------------------------------------------
    crrt_support = 0.0

    return {
        "selected_protocols": {
            "invasive_support": clamp01(invasive_support),
            "noninvasive_support": clamp01(noninvasive_support),
            "highflow_support": clamp01(highflow_support),
            "vasopressor_support": clamp01(vasopressor_support),
            "crrt_support": clamp01(crrt_support),
        }
    }

def escalation_agent(state: OverallState):
    protocols = state.get("selected_protocols")
    current = state.get("original_interventions")

    if protocols is None or current is None:
        return {}

    current_invasive = clamp01(current.get("invasive", 0))
    current_noninvasive = clamp01(current.get("noninvasive", 0))
    current_highflow = clamp01(current.get("highflow", 0))
    current_vasopressor = clamp01(current.get("vasopressor", 0))
    current_crrt = clamp01(current.get("crrt", 0))

    invasive_support = clamp01(
        protocols.get("invasive_support", 0.0)
    )
    noninvasive_support = clamp01(
        protocols.get("noninvasive_support", 0.0)
    )
    highflow_support = clamp01(
        protocols.get("highflow_support", 0.0)
    )

    # --------------------------------------------------------
    # Respiratory escalation
    #
    # Respiratory supports are alternatives. We therefore
    # defuzzify the competing fuzzy suitability scores by
    # selecting the strongest candidate rather than returning
    # simultaneous escalation of multiple modes.
    # --------------------------------------------------------
    respiratory_candidates = {
        "invasive": invasive_support,
        "noninvasive": noninvasive_support,
        "highflow": highflow_support,
    }

    if current_invasive > 0.0:
        # Already invasively supported: do not recommend NIV/HFNC
        # as a simultaneous escalation.
        respiratory_escalation = {
            "invasive": 0.0,
            "noninvasive": 0.0,
            "highflow": 0.0,
        }
    else:
        best_mode, best_strength = max(
            respiratory_candidates.items(),
            key=lambda item: item[1]
        )

        respiratory_escalation = {
            "invasive": 0.0,
            "noninvasive": 0.0,
            "highflow": 0.0,
        }

        # Do not escalate to a mode already active.
        current_mode = {
            "invasive": current_invasive,
            "noninvasive": current_noninvasive,
            "highflow": current_highflow,
        }

        if current_mode[best_mode] <= 0.0:
            respiratory_escalation[best_mode] = clamp01(
                best_strength
            )

    # --------------------------------------------------------
    # Circulation
    # --------------------------------------------------------
    vasopressor_support = clamp01(
        protocols.get("vasopressor_support", 0.0)
    )

    vasopressor_escalation = fuzzy_and(
        vasopressor_support,
        fuzzy_not(current_vasopressor)
    )

    # --------------------------------------------------------
    # Renal
    # --------------------------------------------------------
    crrt_support = clamp01(
        protocols.get("crrt_support", 0.0)
    )

    crrt_escalation = fuzzy_and(
        crrt_support,
        fuzzy_not(current_crrt)
    )

    return {
        "escalation_decision": {
            "invasive": clamp01(
                respiratory_escalation["invasive"]
            ),
            "noninvasive": clamp01(
                respiratory_escalation["noninvasive"]
            ),
            "highflow": clamp01(
                respiratory_escalation["highflow"]
            ),
            "vasopressor": clamp01(vasopressor_escalation),
            "crrt": clamp01(crrt_escalation),
        }
    }

def fuzzy_and(*values):
    """
    Mamdani fuzzy AND = minimum.
    """

    if not values:
        return 0.0

    values = [
        clamp01(v)
        for v in values
    ]

    return min(values)


def fuzzy_or(*values):
    """
    Mamdani fuzzy OR = maximum.
    """

    if not values:
        return 0.0

    values = [
        clamp01(v)
        for v in values
    ]

    return max(values)


def fuzzy_not(value):
    """
    Fuzzy complement.
    """

    return 1.0 - clamp01(value)


def clamp01(value):
    """
    Keep a fuzzy value inside [0, 1].
    """

    if value is None:
        return 0.0

    return max(
        0.0,
        min(1.0, float(value))
    )


def get_numeric(data: Optional[Dict], *keys, default=None):
    """Return the first usable numeric value from a mapping."""
    data = data or {}
    for key in keys:
        value = data.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def normalized_fio2(value):
    """Accept FiO2 as 80 or 0.80 and normalize it to percent."""
    if value is None:
        return None
    value = float(value)
    if 0 < value <= 1:
        value *= 100.0
    return max(0.0, min(100.0, value))


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

    current_invasive = clamp01(current.get("invasive", 0))
    current_noninvasive = clamp01(current.get("noninvasive", 0))
    current_highflow = clamp01(current.get("highflow", 0))
    current_vasopressor = clamp01(current.get("vasopressor", 0))
    current_crrt = clamp01(current.get("crrt", 0))

    if fuzzy_or(
        current_invasive,
        current_noninvasive,
        current_highflow,
        current_vasopressor,
        current_crrt
    ) == 0.0:
        return {"weaning_recommendation": None}

    low_spo2 = clamp01(scores.get("low_spo2", 0.0))
    low_map = clamp01(scores.get("low_map", 0.0))
    high_shock_index = clamp01(
        scores.get("high_shock_index", 0.0)
    )
    hypercapnia = clamp01(scores.get("hypercapnia", 0.0))
    acidemia = clamp01(scores.get("acidemia", 0.0))

    oxygenation_stability = fuzzy_not(low_spo2)
    hemodynamic_stability = fuzzy_and(
        fuzzy_not(low_map),
        fuzzy_not(high_shock_index)
    )
    ventilation_stability = fuzzy_not(hypercapnia)
    acid_base_stability = fuzzy_not(acidemia)

    gcs_score = gcs.get("gcs")
    fio2 = normalized_fio2(get_numeric(vent, "set_fio21", "fio2", "FiO2"))
    peep = get_numeric(vent, "set_peep1", "peep", "PEEP")

    # Missing information is not treated as "normal".
    fio2_readiness = (
        trapezoid_membership(
            fio2,
            21,
            21,
            30,
            40
        )
        if fio2 is not None else 0.0
    )

    peep_readiness = (
        trapezoid_membership(
            peep,
            3,
            3,
            5,
            8
        )
        if peep is not None else 0.0
    )

    neurological_readiness = (
        trapezoid_membership(
            gcs_score,
            10,
            12,
            15,
            15
        )
        if gcs_score is not None else 0.0
    )

    # --------------------------------------------------------
    # SBT readiness
    # --------------------------------------------------------
    sbt_readiness = fuzzy_and(
        oxygenation_stability,
        hemodynamic_stability,
        ventilation_stability,
        acid_base_stability,
        neurological_readiness,
        fio2_readiness,
        peep_readiness
    )

    # --------------------------------------------------------
    # Extubation readiness
    # --------------------------------------------------------
    extubation_readiness = fuzzy_and(
        sbt_readiness,
        neurological_readiness,
        hemodynamic_stability,
        oxygenation_stability,
        ventilation_stability
    )

    # --------------------------------------------------------
    # NIV weaning
    # --------------------------------------------------------
    niv_weaning_readiness = fuzzy_and(
        oxygenation_stability,
        ventilation_stability,
        acid_base_stability,
        hemodynamic_stability,
        fio2_readiness
    )

    # --------------------------------------------------------
    # HFNC weaning
    # --------------------------------------------------------
    highflow_weaning_readiness = fuzzy_and(
        oxygenation_stability,
        hemodynamic_stability,
        fio2_readiness
    )

    # --------------------------------------------------------
    # Vasopressor weaning
    # --------------------------------------------------------
    vasopressor_weaning = hemodynamic_stability

    # --------------------------------------------------------
    # CRRT weaning
    #
    # No renal recovery variables exist, so do not manufacture
    # a CRRT-weaning score.
    # --------------------------------------------------------
    crrt_weaning = 0.0

    return {
        "weaning_recommendation": {
            "sbt_readiness": clamp01(sbt_readiness),
            "extubation_readiness": clamp01(extubation_readiness),
            "niv_weaning_readiness": clamp01(
                niv_weaning_readiness
            ),
            "highflow_weaning_readiness": clamp01(
                highflow_weaning_readiness
            ),
            "vasopressor_weaning_readiness": clamp01(
                vasopressor_weaning
            ),
            "crrt_weaning_readiness": clamp01(crrt_weaning),
        }
    }

def treatment_agent(state: OverallState):
    escalation = state.get("escalation_decision") or {}
    weaning = state.get("weaning_recommendation") or {}
    current = state.get("original_interventions") or {}

    # Current intervention strengths
    current_invasive = clamp01(current.get("invasive", 0))
    current_noninvasive = clamp01(current.get("noninvasive", 0))
    current_highflow = clamp01(current.get("highflow", 0))
    current_vasopressor = clamp01(current.get("vasopressor", 0))
    current_crrt = clamp01(current.get("crrt", 0))

    # Escalation strengths
    invasive_escalation = clamp01(
        escalation.get("invasive", 0.0)
    )
    noninvasive_escalation = clamp01(
        escalation.get("noninvasive", 0.0)
    )
    highflow_escalation = clamp01(
        escalation.get("highflow", 0.0)
    )
    vasopressor_escalation = clamp01(
        escalation.get("vasopressor", 0.0)
    )
    crrt_escalation = clamp01(
        escalation.get("crrt", 0.0)
    )

    # Weaning strengths
    extubation_readiness = clamp01(
        weaning.get("extubation_readiness", 0.0)
    )
    niv_weaning = clamp01(
        weaning.get("niv_weaning_readiness", 0.0)
    )
    highflow_weaning = clamp01(
        weaning.get("highflow_weaning_readiness", 0.0)
    )
    vasopressor_weaning = clamp01(
        weaning.get("vasopressor_weaning_readiness", 0.0)
    )
    crrt_weaning = clamp01(
        weaning.get("crrt_weaning_readiness", 0.0)
    )

    # --------------------------------------------------------
    # Existing support
    # --------------------------------------------------------
    invasive_continuation = fuzzy_and(
        current_invasive,
        fuzzy_not(extubation_readiness)
    )

    noninvasive_continuation = fuzzy_and(
        current_noninvasive,
        fuzzy_not(niv_weaning)
    )

    highflow_continuation = fuzzy_and(
        current_highflow,
        fuzzy_not(highflow_weaning)
    )

    vasopressor_continuation = fuzzy_and(
        current_vasopressor,
        fuzzy_not(vasopressor_weaning)
    )

    crrt_continuation = fuzzy_and(
        current_crrt,
        fuzzy_not(crrt_weaning)
    )

    # --------------------------------------------------------
    # Combine continuation + escalation.
    # Respiratory modes are mutually exclusive.
    # --------------------------------------------------------
    respiratory_candidates = {
        "invasive": fuzzy_or(
            invasive_continuation,
            invasive_escalation
        ),
        "noninvasive": fuzzy_or(
            noninvasive_continuation,
            noninvasive_escalation
        ),
        "highflow": fuzzy_or(
            highflow_continuation,
            highflow_escalation
        ),
    }

    best_mode, best_strength = max(
        respiratory_candidates.items(),
        key=lambda item: item[1]
    )

    recommended = {
        "invasive": 0.0,
        "noninvasive": 0.0,
        "highflow": 0.0,
        "vasopressor": clamp01(
            fuzzy_or(
                vasopressor_continuation,
                vasopressor_escalation
            )
        ),
        "crrt": clamp01(
            fuzzy_or(
                crrt_continuation,
                crrt_escalation
            )
        ),
    }

    # Do not convert a weak/zero respiratory signal into support.
    if best_strength > 0.0:
        recommended[best_mode] = clamp01(best_strength)

    liberation = {
        "sbt_readiness": clamp01(
            weaning.get("sbt_readiness", 0.0)
        ),
        "extubation_readiness": extubation_readiness,
        "niv_weaning": niv_weaning,
        "highflow_weaning": highflow_weaning,
        "vasopressor_weaning": vasopressor_weaning,
        "crrt_weaning": crrt_weaning,
    }

    return {
        "recommended_interventions": recommended,
        "liberation_scores": liberation
    }

def ventilator_setting_agent(state: OverallState):
    static = state.get("original_static") or {}
    labs = state.get("original_labs") or {}
    forecast = state.get("forecasted_vitals")
    vent = state.get("original_ventilator") or {}
    interventions = state.get("recommended_interventions") or {}
    scores = state.get("severity_scores") or {}

    if forecast is None:
        return {}

    invasive = clamp01(interventions.get("invasive", 0.0))
    noninvasive = clamp01(interventions.get("noninvasive", 0.0))
    highflow = clamp01(interventions.get("highflow", 0.0))

    # No respiratory intervention recommended.
    if max(invasive, noninvasive, highflow) <= 0.0:
        return {
            "recommended_ventilator_changes": {
                "status": "no_respiratory_change",
                "fuzzy_control_strength": 0.0,
            }
        }

    pbw = get_numeric(static, "pbw_kg", "pbw", "predicted_body_weight", default=0.0)
    spo2 = float(forecast.spO2)

    pco2 = labs.get("pCO2")

    current_fio2 = normalized_fio2(
        get_numeric(vent, "set_fio21", "fio2", "FiO2", default=21)
    )
    current_peep = get_numeric(
        vent, "set_peep1", "peep", "PEEP", default=5
    )
    current_rr = get_numeric(
        vent, "set_rr1", "rr", "respiratory_rate", "RR", default=16
    )

    current_fio2 = max(21.0, min(100.0, current_fio2))
    current_peep = max(0.0, min(20.0, current_peep))
    current_rr = max(5.0, min(40.0, current_rr))

    # --------------------------------------------------------
    # SpO2 fuzzy sets
    # --------------------------------------------------------
    spo2_very_low = trapezoid_membership(
        spo2, 80, 80, 85, 90
    )
    spo2_low = trapezoid_membership(
        spo2, 88, 90, 92, 94
    )
    spo2_target = trapezoid_membership(
        spo2, 90, 92, 96, 97
    )
    spo2_high = trapezoid_membership(
        spo2, 96, 97, 100, 100
    )

    # --------------------------------------------------------
    # FiO2 fuzzy control
    # --------------------------------------------------------
    numerator = (
        spo2_very_low * 15.0
        + spo2_low * 8.0
        + spo2_target * 0.0
        + spo2_high * -5.0
    )

    denominator = (
        spo2_very_low
        + spo2_low
        + spo2_target
        + spo2_high
    )

    fio2_change = (
        numerator / denominator
        if denominator > 0.0
        else 0.0
    )

    recommended_fio2 = max(
        21.0,
        min(100.0, current_fio2 + fio2_change)
    )

    # --------------------------------------------------------
    # P/F fuzzy PEEP control
    # --------------------------------------------------------
    pf_severe = clamp01(scores.get("pf_severe", 0.0))
    pf_moderate = clamp01(scores.get("pf_moderate", 0.0))

    oxygenation_need = fuzzy_or(
        pf_severe,
        pf_moderate
    )

    # Bounded adjustment: fuzzy controller modifies the current
    # setting; it does not jump directly to an unbounded value.
    peep_change = oxygenation_need * 2.0

    recommended_peep = max(
        5.0,
        min(20.0, current_peep + peep_change)
    )

    # --------------------------------------------------------
    # PaCO2 fuzzy RR control
    # --------------------------------------------------------
    if pco2 is not None:
        pco2_high = trapezoid_membership(
            pco2, 45, 50, 60, 60
        )
        pco2_normal = trapezoid_membership(
            pco2, 35, 38, 42, 45
        )
        pco2_low = trapezoid_membership(
            pco2, 25, 25, 30, 35
        )

        rr_numerator = (
            pco2_high * 4.0
            + pco2_normal * 0.0
            + pco2_low * -2.0
        )

        rr_denominator = (
            pco2_high
            + pco2_normal
            + pco2_low
        )

        rr_change = (
            rr_numerator / rr_denominator
            if rr_denominator > 0.0
            else 0.0
        )
    else:
        rr_change = 0.0

    recommended_rr = max(
        10.0,
        min(35.0, current_rr + rr_change)
    )

    # --------------------------------------------------------
    # Tidal volume
    # --------------------------------------------------------
    target_tv = (
        pbw * 6.0
        if pbw > 0.0
        else None
    )

    # --------------------------------------------------------
    # NIV pressure support
    # --------------------------------------------------------
    pressure_support_need = fuzzy_or(
        pf_moderate,
        pf_severe,
        trapezoid_membership(
            pco2, 45, 50, 60, 60
        ) if pco2 is not None else 0.0
    )

    pressure_support = max(
        5.0,
        min(
            20.0,
            5.0 + pressure_support_need * 10.0
        )
    )

    # --------------------------------------------------------
    # Defuzzify competing respiratory intervention strengths.
    # --------------------------------------------------------
    mode_scores = {
        "HFNC": highflow,
        "NIV": noninvasive,
        "INVASIVE": invasive,
    }

    selected_mode, selected_strength = max(
        mode_scores.items(),
        key=lambda item: item[1]
    )

    selected_strength = clamp01(selected_strength)

    if selected_strength <= 0.0:
        return {
            "recommended_ventilator_changes": {
                "status": "no_respiratory_change",
                "fuzzy_control_strength": 0.0,
            }
        }

    # --------------------------------------------------------
    # HFNC
    # --------------------------------------------------------
    if selected_mode == "HFNC":
        return {
            "recommended_ventilator_changes": {
                "mode": "HFNC",
                "hfnc_flow_rate": (
                    40.0 + selected_strength * 20.0
                ),
                "set_fio21": recommended_fio2,
                "fuzzy_control_strength": selected_strength,
            }
        }

    # --------------------------------------------------------
    # NIV
    # --------------------------------------------------------
    if selected_mode == "NIV":
        return {
            "recommended_ventilator_changes": {
                "mode": "NIV",
                "set_fio21": recommended_fio2,
                "set_peep1": recommended_peep,
                "pressure_support": pressure_support,
                "fuzzy_control_strength": selected_strength,
            }
        }

    # --------------------------------------------------------
    # Invasive ventilation
    # --------------------------------------------------------
    if target_tv is None:
        return {
            "recommended_ventilator_changes": {
                "mode": "INVASIVE",
                "set_fio21": recommended_fio2,
                "set_peep1": recommended_peep,
                "set_rr1": recommended_rr,
                "fuzzy_control_strength": selected_strength,
            }
        }

    # Use PBW-based low tidal volume target.
    # Keep pressure-control output separate from NIV PS.
    driving_pressure = (
        10.0
        + pf_severe * 5.0
    )

    pressure_control_level = (
        recommended_peep + driving_pressure
    )

    return {
        "recommended_ventilator_changes": {
            "mode": "INVASIVE",
            "set_tv1": target_tv,
            "total_tv": target_tv,
            "set_fio21": recommended_fio2,
            "set_peep1": recommended_peep,
            "total_peep": recommended_peep,
            "set_rr1": recommended_rr,
            "total_rr": recommended_rr,
            "set_ie_ratio1": 2.0,
            "set_pc1": pressure_control_level,
            "rr": recommended_rr,
            "fuzzy_control_strength": selected_strength,
        }
    }

def action_agent(state: OverallState):
    """Create one stable final action payload for the API/UI."""
    syndromes = state.get("detected_syndromes") or {}
    protocols = state.get("selected_protocols") or {}
    escalation = state.get("escalation_decision") or {}
    treatment = state.get("recommended_interventions") or {}
    vent_changes = state.get("recommended_ventilator_changes") or {}
    current = state.get("original_interventions") or {}
    current_vent = state.get("original_ventilator") or {}

    treatment_change = any(
        abs(float(treatment.get(k, 0.0) or 0.0) - float(current.get(k, 0.0) or 0.0)) > 0.05
        for k in ("invasive", "noninvasive", "highflow", "vasopressor", "crrt")
    )

    ventilation_change = False
    if vent_changes and vent_changes.get("status") != "no_respiratory_change":
        for key in ("set_fio21", "set_peep1", "set_rr1", "set_tv1", "set_pc1", "set_ie_ratio1"):
            if key not in vent_changes:
                continue
            old_value = current_vent.get(key)
            new_value = vent_changes.get(key)
            if old_value is None:
                ventilation_change = True
                break
            try:
                if abs(float(new_value) - float(old_value)) > 0.1:
                    ventilation_change = True
                    break
            except (TypeError, ValueError):
                if str(new_value) != str(old_value):
                    ventilation_change = True
                    break

    trace = [
        {"stage": "syndrome_detection", "status": "completed", "output": syndromes},
        {"stage": "protocol_selection", "status": "completed", "output": protocols},
        {"stage": "escalation", "status": "completed", "output": escalation},
        {"stage": "treatment_selection", "status": "completed", "output": treatment},
        {"stage": "ventilator_settings", "status": "completed", "output": vent_changes},
    ]

    return {
        "final_action": {
            "status": "completed",
            "treatment_selected": treatment,
            "ventilator_changes": vent_changes,
            "treatment_change_detected": treatment_change,
            "ventilation_change_detected": ventilation_change,
        },
        "decision_trace": trace,
        "treatment_change_detected": treatment_change,
        "ventilation_change_detected": ventilation_change,
    }


# ─────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────
def build_graph():
    """
    Run the decision pipeline sequentially.

    Every downstream clinical-decision agent now executes only after the
    preceding agent has written its required state. This removes the
    fan-out/fan-in dependency ambiguity around syndrome -> treatment.
    """
    builder = StateGraph(OverallState)

    builder.add_node("start_agent", start_agent)
    builder.add_node("forecast_agent", forecast_agent)
    builder.add_node("prediction_agent", prediction_agent)
    builder.add_node("summarization_agent", summarization_agent)
    builder.add_node("severity_scoring_agent", severity_scoring_agent)
    builder.add_node("syndrome_detection_agent", syndrome_detection_agent)
    builder.add_node("protocol_selection_agent", protocol_selection_agent)
    builder.add_node("weaning_agent", weaning_agent)
    builder.add_node("escalation_agent", escalation_agent)
    builder.add_node("treatment_agent", treatment_agent)
    builder.add_node("ventilator_setting_agent", ventilator_setting_agent)
    builder.add_node("action_agent", action_agent)

    builder.add_edge(START, "start_agent")
    builder.add_edge("start_agent", "forecast_agent")
    builder.add_edge("forecast_agent", "prediction_agent")
    builder.add_edge("prediction_agent", "summarization_agent")
    builder.add_edge("summarization_agent", "severity_scoring_agent")
    builder.add_edge("severity_scoring_agent", "syndrome_detection_agent")
    builder.add_edge("syndrome_detection_agent", "protocol_selection_agent")
    builder.add_edge("protocol_selection_agent", "weaning_agent")
    builder.add_edge("weaning_agent", "escalation_agent")
    builder.add_edge("escalation_agent", "treatment_agent")
    builder.add_edge("treatment_agent", "ventilator_setting_agent")
    builder.add_edge("ventilator_setting_agent", "action_agent")
    builder.add_edge("action_agent", END)

    return builder.compile()


graph = build_graph()
