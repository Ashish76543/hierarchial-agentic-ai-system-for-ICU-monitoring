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
    recommended_ventilator_changes: Optional[Dict[str, float]]
    recommended_interventions: Optional[Dict[str, int]]
    improvement_detected: Optional[bool]
    severity_scores: Optional[Dict[str, float]]
    detected_syndromes: Optional[Dict[str, bool]]
    selected_protocols: Optional[Dict[str, str]]
    escalation_decision: Optional[Dict[str, int]]
    weaning_recommendation: Optional[Dict[str, int]]


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


def severity_scoring_agent(state: OverallState):
    forecast = state.get("forecasted_vitals")
    labs = state.get("original_labs")
    gcs = state.get("original_gcs")
    vent = state.get("original_ventilator")
    if forecast is None:
        return {}
    scores = {}
    scores["shock_index"] = forecast.heart_rate / forecast.sbp if forecast.sbp > 0 else 0
    scores["hypotension"] = forecast.mbp < 65
    scores["hypoxia"] = forecast.spO2 < 92
    scores["severe_hypoxia"] = forecast.spO2 < 88
    if labs:
        scores["acidosis"] = labs["pH"] < 7.35
        scores["hypercapnia"] = labs["pCO2"] > 45
        scores["hypoxemia_lab"] = labs["pO2"] < 80
    else:
        scores["acidosis"] = scores["hypercapnia"] = scores["hypoxemia_lab"] = False
    scores["neurological_impairment"] = gcs.get("gcs", 15) < 13 if gcs else False
    if vent:
        scores["high_fio2"] = vent["set_fio21"] > 40
        scores["high_peep"] = vent["set_peep1"] > 8
    else:
        scores["high_fio2"] = scores["high_peep"] = False
    return {"severity_scores": scores}


def syndrome_detection_agent(state: OverallState):
    scores = state.get("severity_scores")
    outcomes = state.get("outcomes")
    if scores is None:
        return {}
    syndromes = {
        "respiratory_failure": scores["hypoxia"] or scores["hypoxemia_lab"] or scores["high_fio2"] or scores["hypercapnia"],
        "severe_respiratory_failure": scores["severe_hypoxia"] or (scores["hypoxia"] and scores["hypercapnia"]),
        "circulatory_shock": scores["hypotension"] or scores["shock_index"] > 0.9,
        "neurological_failure": scores["neurological_impairment"],
        "sepsis": outcomes.sepsis_outcome if outcomes else False,
    }
    return {"detected_syndromes": syndromes}


def protocol_selection_agent(state: OverallState):
    syndromes = state.get("detected_syndromes")
    if syndromes is None:
        return {}
    protocols = {"oxygen_protocol": None, "circulation_protocol": None, "renal_protocol": None}
    if syndromes["severe_respiratory_failure"]:
        protocols["oxygen_protocol"] = "invasive"
    elif syndromes["respiratory_failure"]:
        protocols["oxygen_protocol"] = "noninvasive"
    if syndromes["circulatory_shock"]:
        protocols["circulation_protocol"] = "vasopressor"
    if syndromes["sepsis"] and syndromes["circulatory_shock"]:
        protocols["renal_protocol"] = "crrt"
    return {"selected_protocols": protocols}


def escalation_agent(state: OverallState):
    protocols = state.get("selected_protocols")
    current = state.get("original_interventions")
    if protocols is None or current is None:
        return {}
    escalation = {"invasive": 0, "noninvasive": 0, "highflow": 0, "vasopressor": 0, "crrt": 0}
    if protocols["oxygen_protocol"] == "invasive":
        escalation["invasive"] = 1
    elif protocols["oxygen_protocol"] == "noninvasive" and current["invasive"] == 0:
        escalation["noninvasive"] = 1
    if protocols["circulation_protocol"] == "vasopressor":
        escalation["vasopressor"] = 1
    if protocols["renal_protocol"] == "crrt":
        escalation["crrt"] = 1
    return {"escalation_decision": escalation}


def weaning_agent(state: OverallState):
    forecast = state.get("forecasted_vitals")
    labs = state.get("original_labs")
    gcs = state.get("original_gcs")
    vent = state.get("original_ventilator")
    current = state.get("original_interventions")
    scores = state.get("severity_scores")
    if forecast is None or current is None:
        return {}
    on_invasive = current.get("invasive", 0) == 1
    on_niv = current.get("noninvasive", 0) == 1
    on_highflow = current.get("highflow", 0) == 1
    on_vasopressor = current.get("vasopressor", 0) == 1
    on_crrt = current.get("crrt", 0) == 1
    if not any([on_invasive, on_niv, on_highflow, on_vasopressor, on_crrt]):
        return {"weaning_recommendation": None}
    weaning = {k: current.get(k, 0) for k in ["invasive", "noninvasive", "highflow", "vasopressor", "crrt"]}
    spo2 = forecast.spO2
    pco2 = labs.get("pCO2", 40) if labs else 40
    ph = labs.get("pH", 7.40) if labs else 7.40
    gcs_score = gcs.get("gcs", 15) if gcs else 15
    fio2 = vent.get("set_fio21", 21) if vent else 21
    peep = vent.get("set_peep1", 5) if vent else 5
    mbp = forecast.mbp
    if on_invasive and all([spo2 >= 92, fio2 <= 40, peep <= 8, ph >= 7.35, gcs_score >= 13, mbp >= 65, pco2 <= 55]):
        weaning.update({"invasive": 0, "noninvasive": 1, "highflow": 0})
    elif on_niv and all([spo2 >= 94, fio2 <= 40, peep <= 5, ph >= 7.35, pco2 <= 50, gcs_score >= 13, mbp >= 65]):
        weaning.update({"noninvasive": 0, "highflow": 1})
    elif on_highflow and all([spo2 >= 95, fio2 <= 30, ph >= 7.35, pco2 <= 45, mbp >= 65]):
        weaning["highflow"] = 0
    if on_vasopressor and mbp >= 70 and not (scores.get("hypotension", False) if scores else False) and not (scores.get("shock_index", 1) > 0.9 if scores else False):
        weaning["vasopressor"] = 0
    if on_crrt:
        outcomes = state.get("outcomes")
        if outcomes and not outcomes.sepsis_outcome and mbp >= 65 and not (scores.get("hypotension", False) if scores else False):
            weaning["crrt"] = 0
    if weaning == current:
        return {"weaning_recommendation": None}
    return {"weaning_recommendation": weaning}


def treatment_agent(state: OverallState):
    escalation = state.get("escalation_decision")
    weaning = state.get("weaning_recommendation")
    current = state.get("original_interventions")
    if escalation is None:
        return {}
    recommended = current.copy() if current else {}
    if weaning:
        recommended.update(weaning)
    for key, value in escalation.items():
        if value == 1:
            recommended[key] = 1
    if recommended.get("invasive", 0) == 1:
        recommended["noninvasive"] = 0
        recommended["highflow"] = 0
    elif recommended.get("noninvasive", 0) == 1:
        recommended["highflow"] = 0
    return {"recommended_interventions": recommended}


def ventilator_setting_agent(state: OverallState):
    static = state["original_static"]
    labs = state["original_labs"]
    forecast = state["forecasted_vitals"]
    vent = state["original_ventilator"]
    interventions = state["recommended_interventions"]
    invasive = interventions.get("invasive", 0)
    noninvasive = interventions.get("noninvasive", 0)
    highflow = interventions.get("highflow", 0)
    pbw = float(static["pbw_kg"])
    spo2 = forecast.spO2
    pco2 = labs["pCO2"]
    recommended = {}
    if invasive == 0 and noninvasive == 0 and highflow == 0:
        return {"recommended_ventilator_changes": None}
    if highflow == 1:
        current_fio2 = vent["set_fio21"]
        fio2 = min(current_fio2 + 10, 100) if spo2 < 88 else max(current_fio2 - 10, 30) if spo2 > 96 else current_fio2
        return {"recommended_ventilator_changes": {"mode": "HFNC", "hfnc_flow_rate": 50, "set_fio21": fio2}}
    if noninvasive == 1:
        current_fio2 = vent["set_fio21"]
        fio2 = min(current_fio2 + 10, 100) if spo2 < 88 else max(current_fio2 - 10, 30) if spo2 > 96 else current_fio2
        peep = max(5, vent["set_peep1"])
        return {"recommended_ventilator_changes": {"mode": "NIV", "set_fio21": fio2, "set_peep1": peep, "pressure_support": 10}}
    if invasive == 1:
        set_tv1 = pbw * 6
        current_fio2 = vent["set_fio21"]
        fio2 = min(current_fio2 + 10, 100) if spo2 < 88 else max(current_fio2 - 10, 30) if spo2 > 96 else current_fio2
        peep_map = [(40, 5), (50, 8), (60, 10), (70, 12), (80, 14), (90, 16)]
        peep = 18
        for threshold, p in peep_map:
            if fio2 <= threshold:
                peep = p
                break
        current_rr = vent["set_rr1"]
        rr = min(current_rr + 2, 35) if pco2 > 45 else max(current_rr - 2, 10) if pco2 < 35 else current_rr
        set_pc1 = peep + 14
        return {"recommended_ventilator_changes": {
            "mode": "INVASIVE", "set_tv1": set_tv1, "total_tv": set_tv1,
            "set_fio21": fio2, "set_peep1": peep, "total_peep": peep,
            "set_rr1": rr, "total_rr": rr, "set_ie_ratio1": 2,
            "set_pc1": set_pc1, "_pinsp_draeger2": set_pc1,
            "_pinsp_hamilton2": set_pc1, "_pcv_level2": set_pc1,
            "_set_pc_draeger": set_pc1, "ppeak": set_pc1, "rr": rr,
        }}


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