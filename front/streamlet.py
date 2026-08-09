import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://localhost:8000/api/analyze"

st.set_page_config(
    page_title="ICU Decision Support",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 ICU Decision Support Dashboard")
st.caption("Forecasting • Fuzzy severity • Syndrome detection • Treatment support")


def d(obj, key):
    value = obj.get(key)
    return value if isinstance(value, dict) else {}


def bar_chart(values, title):
    numeric = {
        k: float(v) for k, v in values.items()
        if isinstance(v, (int, float))
    }
    if not numeric:
        st.info("No numeric data available.")
        return

    df = pd.DataFrame({
        "Metric": list(numeric.keys()),
        "Value": list(numeric.values())
    })

    fig = go.Figure(go.Bar(
        x=df["Metric"],
        y=df["Value"],
        text=[f"{v:.2f}" for v in df["Value"]],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        height=380,
        yaxis_title="Membership / Strength",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)


def trend_chart(payload, result):
    forecast = d(result, "forecasted_vitals")
    current = payload["vitals"]
    prior = payload["prior_vitals"]

    names = {
        "heart_rate": "Heart Rate",
        "sbp": "SBP",
        "mbp": "MAP",
        "spO2": "SpO₂",
        "temperature": "Temperature",
    }

    key = st.selectbox(
        "Vital",
        list(names),
        format_func=lambda x: names[x],
    )

    values = [
        prior.get(key),
        current.get(key),
        forecast.get(key),
    ]

    if any(v is None for v in values):
        st.warning("Data missing for this vital.")
        return

    fig = go.Figure(go.Scatter(
        x=["T-1h", "Current", "Forecast"],
        y=values,
        mode="lines+markers+text",
        text=[f"{v:.1f}" for v in values],
        textposition="top center",
    ))
    fig.update_layout(
        title=f"{names[key]} Trend",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def intervention_cards(values):
    names = {
        "invasive": "Invasive",
        "noninvasive": "NIV",
        "highflow": "HFNC",
        "vasopressor": "Vasopressor",
        "crrt": "CRRT",
    }

    cols = st.columns(5)
    for col, key in zip(cols, names):
        col.metric(names[key], f"{float(values.get(key, 0)):.2f}")


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

with st.sidebar:
    st.header("Patient Input")

    st.subheader("Static")
    age = st.number_input("Age", 0, 120, 60)
    gender = st.selectbox("Gender", ["M", "F"])
    unit = st.text_input("Care Unit", "MICU")
    pbw = st.number_input("PBW (kg)", 20.0, 200.0, 70.0)

    st.divider()

    st.subheader("Current — T=0")
    hr = st.number_input("Heart Rate", 20.0, 250.0, 110.0)
    sbp = st.number_input("SBP", 40.0, 250.0, 95.0)
    dbp = st.number_input("DBP", 20.0, 150.0, 60.0)
    mbp = st.number_input("MAP", 30.0, 180.0, 70.0)
    spo2 = st.number_input("SpO₂ (%)", 50.0, 100.0, 92.0)
    temp = st.number_input("Temperature", 25.0, 45.0, 37.5)
    glucose = st.number_input("Glucose", 20.0, 1000.0, 120.0)

    st.subheader("Current Labs")
    ph = st.number_input("pH", 6.5, 8.0, 7.40)
    pco2 = st.number_input("pCO₂", 10.0, 150.0, 40.0)
    po2 = st.number_input("pO₂", 20.0, 500.0, 100.0)

    st.subheader("Current GCS")
    gcs = st.number_input("GCS", 3.0, 15.0, 15.0)
    sofa = st.number_input("SOFA 24h", 0.0, 24.0, 5.0)

    st.subheader("Current Ventilator")
    fio2 = st.number_input("FiO₂ (%)", 21.0, 100.0, 40.0)
    peep = st.number_input("PEEP", 0.0, 20.0, 5.0)
    rr = st.number_input("Respiratory Rate", 5.0, 40.0, 16.0)

    st.subheader("Current Interventions")
    invasive = st.checkbox("Invasive")
    niv = st.checkbox("NIV")
    highflow = st.checkbox("HFNC")
    vasopressor = st.checkbox("Vasopressor")
    crrt = st.checkbox("CRRT")

    st.divider()

    st.subheader("Previous — T-1h")
    p_hr = st.number_input("Previous HR", 20.0, 250.0, 105.0)
    p_sbp = st.number_input("Previous SBP", 40.0, 250.0, 100.0)
    p_dbp = st.number_input("Previous DBP", 20.0, 150.0, 65.0)
    p_mbp = st.number_input("Previous MAP", 30.0, 180.0, 75.0)
    p_spo2 = st.number_input("Previous SpO₂ (%)", 50.0, 100.0, 94.0)
    p_temp = st.number_input("Previous Temperature", 25.0, 45.0, 37.3)
    p_glucose = st.number_input("Previous Glucose", 20.0, 1000.0, 115.0)

    st.subheader("Previous Labs")
    p_ph = st.number_input("Previous pH", 6.5, 8.0, 7.42)
    p_pco2 = st.number_input("Previous pCO₂", 10.0, 150.0, 38.0)
    p_po2 = st.number_input("Previous pO₂", 20.0, 500.0, 110.0)

    st.subheader("Previous GCS")
    p_gcs = st.number_input("Previous GCS", 3.0, 15.0, 15.0)
    p_sofa = st.number_input("Previous SOFA", 0.0, 24.0, 4.0)

    st.subheader("Previous Ventilator")
    p_fio2 = st.number_input("Previous FiO₂ (%)", 21.0, 100.0, 40.0)
    p_peep = st.number_input("Previous PEEP", 0.0, 20.0, 5.0)
    p_rr = st.number_input("Previous RR", 5.0, 40.0, 16.0)

    st.subheader("Previous Interventions")
    p_invasive = st.checkbox("Previous Invasive")
    p_niv = st.checkbox("Previous NIV")
    p_highflow = st.checkbox("Previous HFNC")
    p_vasopressor = st.checkbox("Previous Vasopressor")
    p_crrt = st.checkbox("Previous CRRT")

    analyze = st.button("🔍 Analyze Patient", type="primary", use_container_width=True)


# ------------------------------------------------------------
# BUILD REQUEST
# ------------------------------------------------------------

payload = {
    "static": {
        "anchor_age": str(age),
        "gender": gender,
        "first_careunit": unit,
        "pbw_kg": str(pbw),
    },
    "vitals": {
        "heart_rate": hr, "sbp": sbp, "dbp": dbp, "mbp": mbp,
        "spO2": spo2, "temperature": temp, "glucose": glucose,
    },
    "labs": {"pH": ph, "pCO2": pco2, "pO2": po2},
    "gcs": {"gcs": gcs, "sofa_24_hours": sofa},
    "ventilator": {
        "set_fio21": fio2, "set_peep1": peep, "set_rr1": rr,
    },
    "interventions": {
        "invasive": int(invasive), "noninvasive": int(niv),
        "highflow": int(highflow), "vasopressor": int(vasopressor),
        "crrt": int(crrt),
    },
    "outcomes": None,
    "prior_vitals": {
        "heart_rate": p_hr, "sbp": p_sbp, "dbp": p_dbp, "mbp": p_mbp,
        "spO2": p_spo2, "temperature": p_temp, "glucose": p_glucose,
    },
    "prior_labs": {"pH": p_ph, "pCO2": p_pco2, "pO2": p_po2},
    "prior_gcs": {"gcs": p_gcs, "sofa_24_hours": p_sofa},
    "prior_ventilator": {
        "set_fio21": p_fio2, "set_peep1": p_peep, "set_rr1": p_rr,
    },
    "prior_interventions": {
        "invasive": int(p_invasive), "noninvasive": int(p_niv),
        "highflow": int(p_highflow), "vasopressor": int(p_vasopressor),
        "crrt": int(p_crrt),
    },
}


# ------------------------------------------------------------
# CALL FASTAPI
# ------------------------------------------------------------

if analyze:
    try:
        with st.spinner("Running ICU pipeline..."):
            response = requests.post(API_URL, json=payload, timeout=180)
            response.raise_for_status()
            st.session_state.result = response.json()
            st.session_state.payload = payload
    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI is not running. Start it first:\n\n"
            "uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
        )
        st.stop()
    except requests.exceptions.HTTPError:
        try:
            st.error(response.json())
        except Exception:
            st.error(response.text)
        st.stop()
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()


if "result" not in st.session_state:
    st.info("Enter patient data in the sidebar and click Analyze Patient.")
    st.stop()

result = st.session_state.result
payload = st.session_state.payload

forecast = d(result, "forecasted_vitals")
outcomes = d(result, "predicted_outcomes")
severity = d(result, "severity_scores")
syndromes = d(result, "detected_syndromes")
protocols = d(result, "selected_protocols")
escalation = d(result, "escalation_decision")
weaning = d(result, "weaning_recommendation")
recommended = d(result, "recommended_interventions")
vent_changes = d(result, "recommended_ventilator_changes")


# ------------------------------------------------------------
# FORECAST
# ------------------------------------------------------------

st.header("📈 Current → Forecast")

cols = st.columns(5)
for col, label, key, unit in [
    (cols[0], "Heart Rate", "heart_rate", "bpm"),
    (cols[1], "SBP", "sbp", "mmHg"),
    (cols[2], "MAP", "mbp", "mmHg"),
    (cols[3], "SpO₂", "spO2", "%"),
    (cols[4], "Temperature", "temperature", "°C"),
]:
    current_value = payload["vitals"].get(key)
    forecast_value = forecast.get(key)
    if current_value is not None and forecast_value is not None:
        col.metric(
            label,
            f"{forecast_value:.1f} {unit}",
            f"{forecast_value-current_value:+.1f}",
        )


st.subheader("Vital Trend")
trend_chart(payload, result)


# ------------------------------------------------------------
# OUTCOMES
# ------------------------------------------------------------

st.header("🎯 Predicted Outcomes")

cols = st.columns(5)
for col, label, key in [
    (cols[0], "Discharge", "discharge_outcome"),
    (cols[1], "ICU Exit", "icuouttime_outcome"),
    (cols[2], "Death", "death_outcome"),
    (cols[3], "Sepsis", "sepsis_outcome"),
]:
    value = outcomes.get(key)
    col.metric(label, "YES" if value else "NO" if value is not None else "—")

los = outcomes.get("los_outcome")
cols[4].metric("LOS", f"{float(los):.1f} hrs" if los is not None else "—")


# ------------------------------------------------------------
# FUZZY RESULTS
# ------------------------------------------------------------

st.header("🧠 Fuzzy Severity")
severity_plot = {
    k: v for k, v in severity.items()
    if k not in {"shock_index", "pao2", "pf_ratio", "fio2", "peep"}
}
bar_chart(severity_plot, "Severity / Membership Strength")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Detected Syndromes")
    bar_chart(syndromes, "Syndrome Membership")

with col2:
    st.subheader("Protocol Suitability")
    bar_chart(protocols, "Protocol Suitability")


# ------------------------------------------------------------
# DECISIONS
# ------------------------------------------------------------

st.header("⬆️ Escalation")
intervention_cards(escalation)

st.header("⬇️ Weaning / Liberation Readiness")
bar_chart(weaning, "Weaning Readiness")

st.header("💊 Recommended Interventions")
intervention_cards(recommended)


# ------------------------------------------------------------
# VENTILATOR
# ------------------------------------------------------------

st.header("🫁 Recommended Ventilator Settings")

if vent_changes:
    st.info(f"Selected mode: **{vent_changes.get('mode', 'N/A')}**")

    available = [
        ("FiO₂", "set_fio21", "%"),
        ("PEEP", "set_peep1", "cmH₂O"),
        ("RR", "set_rr1", "/min"),
        ("Tidal Volume", "set_tv1", "mL"),
        ("Pressure Support", "pressure_support", "cmH₂O"),
        ("HFNC Flow", "hfnc_flow_rate", "L/min"),
    ]

    existing = [x for x in available if x[1] in vent_changes]
    cols = st.columns(max(1, min(4, len(existing))))

    for col, (label, key, unit) in zip(cols, existing):
        value = vent_changes[key]
        col.metric(label, f"{float(value):.1f} {unit}")

    with st.expander("All ventilator output"):
        st.json(vent_changes)
else:
    st.info("No respiratory support recommendation returned.")


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

st.header("📋 Clinical Summary")
summary = result.get("clinical_summary")
st.write(summary if summary else "No clinical summary returned.")


with st.expander("Complete API Response"):
    st.json(result)