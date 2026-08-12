import json
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = "http://localhost:8000/api/analyze"


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="ICU Clinical Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HUMAN-CENTERED UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ink: #24313f;
        --muted: #687786;
        --paper: #fbfaf7;
        --panel: #ffffff;
        --border: #e2e7eb;
        --blue: #4f718f;
        --blue-soft: #edf4f8;
        --teal: #4e807d;
        --teal-soft: #edf6f4;
        --warm: #a56b4e;
        --warm-soft: #faf0eb;
        --green: #5f826d;
        --green-soft: #eef5f0;
        --red: #a65d5d;
        --red-soft: #faeeee;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
    }

    section[data-testid="stSidebar"] {
        background: #f5f3ee;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    h1 {
        font-weight: 650;
        margin-bottom: 0.2rem;
    }

    h2, h3 {
        font-weight: 600;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
    }

    .section-note {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: -0.45rem;
        margin-bottom: 0.75rem;
    }

    .status-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 0.3rem 0 0.8rem 0;
    }

    .status-label {
        color: var(--muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.15rem;
    }

    .status-value {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 600;
    }

    .status-value.good {
        color: var(--green);
    }

    .status-value.alert {
        color: var(--red);
    }

    .status-value.neutral {
        color: var(--blue);
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.7rem 0.85rem;
        box-shadow: 0 1px 2px rgba(30, 42, 52, 0.03);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        overflow: visible !important;
        white-space: nowrap !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink);
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
        font-size: 1.7rem !important;
        line-height: 1.15 !important;
    }

    div[data-testid="stMetricValue"] > div {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
    }

    .decision-box {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 5px solid var(--blue);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.35rem 0 0.8rem 0;
    }

    .decision-box.alert {
        border-left-color: var(--red);
        background: var(--red-soft);
    }

    .decision-box.good {
        border-left-color: var(--green);
        background: var(--green-soft);
    }

    .decision-title {
        font-weight: 650;
        color: var(--ink);
        margin-bottom: 0.18rem;
    }

    .decision-text {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .small-note {
        font-size: 0.78rem;
        color: var(--muted);
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 2.65rem;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        border-radius: 9px 9px 0 0;
    }

    .stExpander {
        border-color: var(--border) !important;
        border-radius: 12px !important;
    }

    @media (max-width: 1200px) {
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "payload" not in st.session_state:
    st.session_state.payload = None

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Form"


# ============================================================
# HELPERS
# ============================================================

def as_dict(obj: Any) -> Dict[str, Any]:
    """Return a dict when possible; otherwise return an empty dict."""
    return obj if isinstance(obj, dict) else {}


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert a value to float without crashing the UI."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_value(value: Any, decimals: int = 1, suffix: str = "") -> str:
    """Human-friendly value formatting."""
    number = safe_float(value)
    if number is None:
        return "—"
    return f"{number:.{decimals}f}{suffix}"


def normalize_payload(payload: Any) -> Dict[str, Any]:
    """
    Validate the basic shape of an API request.

    This intentionally does not force every nested field to exist,
    because the backend is responsible for the clinical pipeline
    validation. The UI only checks that the top-level object exists.
    """
    if not isinstance(payload, dict):
        raise ValueError("JSON input must contain a single JSON object.")

    required = {
        "static",
        "vitals",
        "labs",
        "gcs",
        "ventilator",
        "interventions",
        "prior_vitals",
        "prior_labs",
        "prior_gcs",
        "prior_ventilator",
        "prior_interventions",
    }

    missing = sorted(required - set(payload.keys()))

    if missing:
        raise ValueError(
            "JSON is missing required sections: "
            + ", ".join(missing)
        )

    return payload


def post_payload(payload: Dict[str, Any]) -> None:
    """Send a request to FastAPI and store the result."""
    try:
        with st.spinner("Reviewing the patient data…"):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=180,
            )

        if response.ok:
            st.session_state.result = response.json()
            st.session_state.payload = payload
            st.success("Analysis completed.")
            return

        try:
            error_body = response.json()
        except ValueError:
            error_body = response.text

        st.error(
            f"FastAPI returned HTTP {response.status_code}.\n\n"
            f"{error_body}"
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "The decision-support API could not be reached.\n\n"
            "Start FastAPI with:\n"
            "`uvicorn api:app --host 0.0.0.0 --port 8000 --reload`"
        )

    except requests.exceptions.Timeout:
        st.error(
            "The API took too long to respond. "
            "The model may still be loading or processing the request."
        )

    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {type(exc).__name__}: {exc}")

    except ValueError as exc:
        st.error(f"The API returned invalid JSON: {exc}")

    except Exception as exc:
        st.error(f"Unexpected error: {type(exc).__name__}: {exc}")


def bar_chart(
    values: Dict[str, Any],
    title: str,
    y_title: str = "Strength",
    normalized: bool = True,
) -> None:
    """Render a readable horizontal bar chart."""
    numeric = {}

    for key, value in values.items():
        number = safe_float(value)
        if number is not None:
            numeric[key] = number

    if not numeric:
        st.info("No numeric data available.")
        return

    df = pd.DataFrame(
        {
            "Metric": list(numeric.keys()),
            "Value": list(numeric.values()),
        }
    )

    fig = go.Figure(
        go.Bar(
            x=df["Value"],
            y=df["Metric"],
            orientation="h",
            text=[f"{v:.2f}" for v in df["Value"]],
            textposition="outside",
            marker_color="#6f8fa8",
        )
    )

    fig.update_layout(
        title=title,
        height=max(280, 52 * len(df)),
        margin=dict(l=10, r=70, t=60, b=20),
        xaxis_title=y_title,
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#24313f"),
        showlegend=False,
    )

    if normalized:
        lower = min(0, min(numeric.values()))
        upper = max(1, max(numeric.values()) * 1.15)
        fig.update_xaxes(range=[lower, upper])

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def trend_chart(payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    forecast = as_dict(result.get("forecasted_vitals"))
    current = as_dict(payload.get("vitals"))
    prior = as_dict(payload.get("prior_vitals"))

    names = {
        "heart_rate": "Heart rate",
        "sbp": "Systolic BP",
        "mbp": "MAP",
        "spO2": "SpO₂",
        "temperature": "Temperature",
    }

    key = st.selectbox(
        "Vital",
        list(names.keys()),
        format_func=lambda x: names[x],
        key="trend_vital",
    )

    values = [
        safe_float(prior.get(key)),
        safe_float(current.get(key)),
        safe_float(forecast.get(key)),
    ]

    if any(value is None for value in values):
        st.warning(
            f"Not enough data was returned to draw the {names[key].lower()} trend."
        )
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=["1 hour ago", "Current", "Forecast"],
            y=values,
            mode="lines+markers+text",
            text=[f"{value:.1f}" for value in values],
            textposition="top center",
            line=dict(color="#5f7f98", width=3),
            marker=dict(size=8, color="#5f7f98"),
        )
    )

    fig.update_layout(
        title=f"{names[key]} over time",
        height=390,
        margin=dict(l=10, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#24313f"),
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#e7ebee"),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def intervention_cards(
    values: Dict[str, Any],
    title_map: Optional[Dict[str, str]] = None,
) -> None:
    """Display intervention strengths without failing on missing keys."""
    names = title_map or {
        "invasive": "Invasive ventilation",
        "noninvasive": "NIV",
        "highflow": "HFNC",
        "vasopressor": "Vasopressor",
        "crrt": "CRRT",
    }

    cols = st.columns(len(names))

    for col, key in zip(cols, names):
        value = safe_float(values.get(key), 0.0)

        if value is None:
            display = "—"
        else:
            display = f"{value:.2f}"

        col.metric(names[key], display)


def render_boolean_outcome(label: str, value: Any) -> None:
    """Render predicted boolean outcomes consistently."""
    if value is None:
        st.metric(label, "—")
        return

    is_positive = bool(value)
    st.metric(label, "Yes" if is_positive else "No")


def render_status(
    label: str,
    value: str,
    tone: str = "neutral",
) -> None:
    """Small human-readable status card."""
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value {tone}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_box(
    title: str,
    text: str,
    tone: str = "neutral",
) -> None:
    st.markdown(
        f"""
        <div class="decision-box {tone}">
            <div class="decision-title">{title}</div>
            <div class="decision-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.title("ICU Clinical Support")

st.markdown(
    '<div class="app-subtitle">'
    "A clinician-oriented view of physiological trends, fuzzy severity, "
    "syndromes, treatment support, and ventilator recommendations."
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Decision-support output only. Recommendations should be reviewed by "
    "a qualified clinician before any patient-care action."
)


# ============================================================
# SIDEBAR INPUT
# ============================================================

with st.sidebar:
    st.header("Patient data")

    input_mode = st.radio(
        "Input method",
        ["Form", "JSON"],
        horizontal=True,
        key="input_mode_radio",
    )

    st.session_state.input_mode = input_mode

    if input_mode == "JSON":

        st.subheader("Load JSON")

        uploaded_json = st.file_uploader(
            "Upload a patient JSON file",
            type=["json"],
            help="Upload a JSON object using the same structure expected by /api/analyze.",
        )

        json_text = st.text_area(
            "Or paste JSON",
            height=260,
            placeholder='{"static": {...}, "vitals": {...}, ...}',
        )

        use_uploaded = uploaded_json is not None

        analyze_json = st.button(
            "Analyze JSON",
            type="primary",
            use_container_width=True,
        )

        if uploaded_json is not None:
            try:
                uploaded_payload = json.load(uploaded_json)

                st.success("JSON file loaded.")

                with st.expander("Preview loaded JSON"):
                    st.json(uploaded_payload)

            except Exception as exc:
                uploaded_payload = None
                st.error(
                    f"Could not read the uploaded JSON: "
                    f"{type(exc).__name__}: {exc}"
                )

        else:
            uploaded_payload = None

        if analyze_json:

            try:
                if use_uploaded:
                    candidate = uploaded_payload
                else:
                    if not json_text.strip():
                        raise ValueError(
                            "Paste JSON or upload a JSON file first."
                        )

                    candidate = json.loads(json_text)

                normalized = normalize_payload(candidate)
                post_payload(normalized)

            except json.JSONDecodeError as exc:
                st.error(
                    f"Invalid JSON near line {exc.lineno}, "
                    f"column {exc.colno}: {exc.msg}"
                )

            except ValueError as exc:
                st.error(str(exc))

    else:

        st.subheader("Patient details")

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=60,
            step=1,
        )

        gender = st.selectbox(
            "Sex",
            ["M", "F"],
        )

        unit = st.text_input(
            "Care unit",
            "MICU",
        )

        pbw = st.number_input(
            "Predicted body weight (kg)",
            min_value=20.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
        )

        st.divider()

        st.subheader("Current — T=0")

        hr = st.number_input(
            "Heart rate (bpm)",
            20.0, 250.0, 110.0,
        )

        sbp = st.number_input(
            "SBP (mmHg)",
            40.0, 250.0, 95.0,
        )

        dbp = st.number_input(
            "DBP (mmHg)",
            20.0, 150.0, 60.0,
        )

        mbp = st.number_input(
            "MAP (mmHg)",
            30.0, 180.0, 70.0,
        )

        spo2 = st.number_input(
            "SpO₂ (%)",
            50.0, 100.0, 92.0,
        )

        temp = st.number_input(
            "Temperature (°C)",
            25.0, 45.0, 37.5,
        )

        glucose = st.number_input(
            "Glucose",
            20.0, 1000.0, 120.0,
        )

        st.subheader("Current labs")

        ph = st.number_input(
            "pH",
            6.5, 8.0, 7.40,
        )

        pco2 = st.number_input(
            "pCO₂",
            10.0, 150.0, 40.0,
        )

        po2 = st.number_input(
            "pO₂",
            20.0, 500.0, 100.0,
        )

        st.subheader("Current neurological status")

        gcs = st.number_input(
            "GCS",
            3.0, 15.0, 15.0,
        )

        sofa = st.number_input(
            "SOFA (24h)",
            0.0, 24.0, 5.0,
        )

        st.subheader("Current ventilator")

        fio2 = st.number_input(
            "FiO₂ (%)",
            21.0, 100.0, 40.0,
        )

        peep = st.number_input(
            "PEEP (cmH₂O)",
            0.0, 20.0, 5.0,
        )

        rr = st.number_input(
            "Respiratory rate (/min)",
            5.0, 40.0, 16.0,
        )

        st.subheader("Current support")

        invasive = st.checkbox("Invasive ventilation")
        niv = st.checkbox("NIV")
        highflow = st.checkbox("HFNC")
        vasopressor = st.checkbox("Vasopressor")
        crrt = st.checkbox("CRRT")

        st.divider()

        st.subheader("Previous — T-1h")

        p_hr = st.number_input(
            "Previous heart rate",
            20.0, 250.0, 105.0,
        )

        p_sbp = st.number_input(
            "Previous SBP",
            40.0, 250.0, 100.0,
        )

        p_dbp = st.number_input(
            "Previous DBP",
            20.0, 150.0, 65.0,
        )

        p_mbp = st.number_input(
            "Previous MAP",
            30.0, 180.0, 75.0,
        )

        p_spo2 = st.number_input(
            "Previous SpO₂ (%)",
            50.0, 100.0, 94.0,
        )

        p_temp = st.number_input(
            "Previous temperature",
            25.0, 45.0, 37.3,
        )

        p_glucose = st.number_input(
            "Previous glucose",
            20.0, 1000.0, 115.0,
        )

        st.subheader("Previous labs")

        p_ph = st.number_input(
            "Previous pH",
            6.5, 8.0, 7.42,
        )

        p_pco2 = st.number_input(
            "Previous pCO₂",
            10.0, 150.0, 38.0,
        )

        p_po2 = st.number_input(
            "Previous pO₂",
            20.0, 500.0, 110.0,
        )

        st.subheader("Previous neurological status")

        p_gcs = st.number_input(
            "Previous GCS",
            3.0, 15.0, 15.0,
        )

        p_sofa = st.number_input(
            "Previous SOFA",
            0.0, 24.0, 4.0,
        )

        st.subheader("Previous ventilator")

        p_fio2 = st.number_input(
            "Previous FiO₂ (%)",
            21.0, 100.0, 40.0,
        )

        p_peep = st.number_input(
            "Previous PEEP",
            0.0, 20.0, 5.0,
        )

        p_rr = st.number_input(
            "Previous respiratory rate",
            5.0, 40.0, 16.0,
        )

        st.subheader("Previous support")

        p_invasive = st.checkbox("Previous invasive ventilation")
        p_niv = st.checkbox("Previous NIV")
        p_highflow = st.checkbox("Previous HFNC")
        p_vasopressor = st.checkbox("Previous vasopressor")
        p_crrt = st.checkbox("Previous CRRT")

        analyze_form = st.button(
            "Review patient",
            type="primary",
            use_container_width=True,
        )

        if analyze_form:

            payload = {
                "static": {
                    "anchor_age": str(age),
                    "gender": gender,
                    "first_careunit": unit,
                    "pbw_kg": str(pbw),
                },
                "vitals": {
                    "heart_rate": hr,
                    "sbp": sbp,
                    "dbp": dbp,
                    "mbp": mbp,
                    "spO2": spo2,
                    "temperature": temp,
                    "glucose": glucose,
                },
                "labs": {
                    "pH": ph,
                    "pCO2": pco2,
                    "pO2": po2,
                },
                "gcs": {
                    "gcs": gcs,
                    "sofa_24_hours": sofa,
                },
                "ventilator": {
                    "set_fio21": fio2,
                    "set_peep1": peep,
                    "set_rr1": rr,
                },
                "interventions": {
                    "invasive": int(invasive),
                    "noninvasive": int(niv),
                    "highflow": int(highflow),
                    "vasopressor": int(vasopressor),
                    "crrt": int(crrt),
                },
                "outcomes": None,
                "prior_vitals": {
                    "heart_rate": p_hr,
                    "sbp": p_sbp,
                    "dbp": p_dbp,
                    "mbp": p_mbp,
                    "spO2": p_spo2,
                    "temperature": p_temp,
                    "glucose": p_glucose,
                },
                "prior_labs": {
                    "pH": p_ph,
                    "pCO2": p_pco2,
                    "pO2": p_po2,
                },
                "prior_gcs": {
                    "gcs": p_gcs,
                    "sofa_24_hours": p_sofa,
                },
                "prior_ventilator": {
                    "set_fio21": p_fio2,
                    "set_peep1": p_peep,
                    "set_rr1": p_rr,
                },
                "prior_interventions": {
                    "invasive": int(p_invasive),
                    "noninvasive": int(p_niv),
                    "highflow": int(p_highflow),
                    "vasopressor": int(p_vasopressor),
                    "crrt": int(p_crrt),
                },
            }

            post_payload(payload)


# ============================================================
# EMPTY STATE
# ============================================================

if st.session_state.result is None:
    st.markdown(
        """
        <div class="decision-box">
            <div class="decision-title">Ready for review</div>
            <div class="decision-text">
                Enter the patient data in the sidebar, or load a JSON patient
                snapshot. The dashboard will show the physiological trend,
                predicted outcomes, fuzzy severity, detected syndromes,
                treatment support, and ventilation recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

result = as_dict(st.session_state.result)
payload = as_dict(st.session_state.payload)

forecast = as_dict(result.get("forecasted_vitals"))
outcomes = as_dict(result.get("predicted_outcomes"))
severity = as_dict(result.get("severity_scores"))
syndromes = as_dict(result.get("detected_syndromes"))
protocols = as_dict(result.get("selected_protocols"))
escalation = as_dict(result.get("escalation_decision"))
weaning = as_dict(result.get("weaning_recommendation"))
recommended = as_dict(result.get("recommended_interventions"))
vent_changes = as_dict(result.get("recommended_ventilator_changes"))
final_action = as_dict(result.get("final_action"))

treatment_changed = result.get("treatment_change_detected")
ventilation_changed = result.get("ventilation_change_detected")
decision_trace = result.get("decision_trace")
clinical_summary = result.get("clinical_summary")


# ============================================================
# TOP STATUS
# ============================================================

st.header("Clinical overview")

status_cols = st.columns(4)

with status_cols[0]:
    syndrome_count = sum(
        1
        for value in syndromes.values()
        if safe_float(value, 0.0) is not None
        and safe_float(value, 0.0) > 0.50
    )

    render_status(
        "Syndromes above 0.50",
        str(syndrome_count),
        "alert" if syndrome_count else "good",
    )

with status_cols[1]:
    treatment_text = (
        "Change identified"
        if treatment_changed
        else "No change identified"
        if treatment_changed is not None
        else "Not reported"
    )

    render_status(
        "Treatment",
        treatment_text,
        "alert" if treatment_changed else "good",
    )

with status_cols[2]:
    ventilation_text = (
        "Change identified"
        if ventilation_changed
        else "No change identified"
        if ventilation_changed is not None
        else "Not reported"
    )

    render_status(
        "Ventilation",
        ventilation_text,
        "alert" if ventilation_changed else "good",
    )

with status_cols[3]:
    death_risk = outcomes.get("death_outcome")

    if death_risk is True:
        render_status("Predicted death outcome", "Positive", "alert")
    elif death_risk is False:
        render_status("Predicted death outcome", "Negative", "good")
    else:
        render_status("Predicted death outcome", "Not reported", "neutral")


# ============================================================
# FORECAST
# ============================================================

st.header("Physiological outlook")

st.markdown(
    '<div class="section-note">'
    "Current values are shown alongside the model's next-interval forecast."
    "</div>",
    unsafe_allow_html=True,
)

forecast_items = [
    ("Heart rate", "heart_rate", "bpm"),
    ("SBP", "sbp", "mmHg"),
    ("MAP", "mbp", "mmHg"),
    ("SpO₂", "spO2", "%"),
    ("Temperature", "temperature", "°C"),
]

cols = st.columns(len(forecast_items))

current_vitals = as_dict(payload.get("vitals"))

for col, label, key, unit in zip(
    cols,
    [item[0] for item in forecast_items],
    [item[1] for item in forecast_items],
    [item[2] for item in forecast_items],
):
    current_value = safe_float(current_vitals.get(key))
    forecast_value = safe_float(forecast.get(key))

    if forecast_value is None:
        col.metric(label, "—")
        continue

    delta = None

    if current_value is not None:
        delta = f"{forecast_value - current_value:+.1f} {unit}"

    col.metric(
        label,
        f"{forecast_value:.1f} {unit}",
        delta,
    )

st.subheader("Vital trend")
trend_chart(payload, result)


# ============================================================
# OUTCOMES
# ============================================================

st.header("Predicted outcomes")

st.markdown(
    '<div class="section-note">'
    "These are model predictions, not confirmed clinical outcomes."
    "</div>",
    unsafe_allow_html=True,
)

outcome_cols = st.columns(5)

with outcome_cols[0]:
    render_boolean_outcome(
        "Discharge",
        outcomes.get("discharge_outcome"),
    )

with outcome_cols[1]:
    render_boolean_outcome(
        "ICU exit",
        outcomes.get("icuouttime_outcome"),
    )

with outcome_cols[2]:
    render_boolean_outcome(
        "Death",
        outcomes.get("death_outcome"),
    )

with outcome_cols[3]:
    render_boolean_outcome(
        "Sepsis",
        outcomes.get("sepsis_outcome"),
    )

with outcome_cols[4]:
    los = safe_float(outcomes.get("los_outcome"))

    outcome_cols[4].metric(
        "Predicted LOS",
        f"{los:.1f} hr" if los is not None else "—",
    )


# ============================================================
# FUZZY ASSESSMENT
# ============================================================

st.header("Fuzzy clinical assessment")

severity_plot = {
    key: value
    for key, value in severity.items()
    if key not in {
        "shock_index",
        "pao2",
        "pf_ratio",
        "fio2",
        "peep",
    }
}

tab1, tab2, tab3 = st.tabs(
    [
        "Severity",
        "Syndromes",
        "Protocol fit",
    ]
)

with tab1:
    bar_chart(
        severity_plot,
        "Severity and membership strength",
    )

    raw_values = {
        key: severity.get(key)
        for key in [
            "shock_index",
            "pf_ratio",
            "pao2",
            "fio2",
            "peep",
        ]
        if severity.get(key) is not None
    }

    if raw_values:
        st.caption("Key calculated values")
        raw_cols = st.columns(len(raw_values))

        for col, (key, value) in zip(raw_cols, raw_values.items()):
            pretty = {
                "shock_index": "Shock index",
                "pf_ratio": "P/F ratio",
                "pao2": "PaO₂",
                "fio2": "FiO₂",
                "peep": "PEEP",
            }.get(key, key)

            col.metric(
                pretty,
                format_value(
                    value,
                    decimals=1 if key != "shock_index" else 2,
                ),
            )

with tab2:
    if syndromes:
        bar_chart(
            syndromes,
            "Detected syndrome membership",
        )
    else:
        st.info("No syndrome output was returned.")

with tab3:
    if protocols:
        bar_chart(
            protocols,
            "Protocol suitability",
        )
    else:
        st.info("No protocol output was returned.")


# ============================================================
# DECISION SUPPORT
# ============================================================

st.header("Decision support")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Escalation")

    if escalation:
        intervention_cards(escalation)
    else:
        st.info("No escalation output was returned.")

with col2:
    st.subheader("Weaning / liberation")

    if weaning:
        bar_chart(
            weaning,
            "Readiness scores",
        )
    else:
        st.info("No weaning output was returned.")


st.subheader("Recommended support")

if recommended:
    intervention_cards(recommended)
else:
    st.info("No treatment recommendation was returned.")


# ============================================================
# FINAL ACTION
# ============================================================

st.header("Overall recommendation")

if final_action:

    action_name = (
        final_action.get("action")
        or final_action.get("treatment_selected")
        or final_action.get("decision")
        or "Clinical decision support result"
    )

    action_reason = (
        final_action.get("reason")
        or final_action.get("rationale")
        or "The graph completed its decision-support sequence."
    )

    render_decision_box(
        str(action_name),
        str(action_reason),
        "alert" if treatment_changed or ventilation_changed else "neutral",
    )

    with st.expander("Final action details"):
        st.json(final_action)

else:
    render_decision_box(
        "No explicit final action was returned",
        "The graph returned intermediate recommendations but did not "
        "provide a final action object.",
        "neutral",
    )


# ============================================================
# VENTILATOR
# ============================================================

st.header("Ventilator recommendation")

if vent_changes:

    selected_mode = vent_changes.get("mode", "Not specified")

    render_decision_box(
        f"Recommended mode: {selected_mode}",
        "The values below are the settings returned by the decision-support graph.",
        "alert" if ventilation_changed else "neutral",
    )

    available = [
        ("FiO₂", "set_fio21", "%"),
        ("PEEP", "set_peep1", "cmH₂O"),
        ("Respiratory rate", "set_rr1", "/min"),
        ("Tidal volume", "set_tv1", "mL"),
        ("Pressure support", "pressure_support", "cmH₂O"),
        ("HFNC flow", "hfnc_flow_rate", "L/min"),
        ("Pressure control", "set_pc1", "cmH₂O"),
    ]

    existing = [
        item
        for item in available
        if item[1] in vent_changes
        and vent_changes[item[1]] is not None
    ]

    if existing:

        cols = st.columns(
            min(4, len(existing))
        )

        for col, (label, key, unit) in zip(
            cols,
            existing,
        ):
            value = safe_float(vent_changes.get(key))

            col.metric(
                label,
                f"{value:.1f} {unit}"
                if value is not None
                else "—",
            )

    else:
        st.info(
            "The API returned a ventilation recommendation but no "
            "displayable numeric settings."
        )

    with st.expander("Ventilator response"):
        st.json(vent_changes)

else:

    render_decision_box(
        "No new ventilator recommendation",
        "The pipeline did not return a ventilator change for this case.",
        "good",
    )


# ============================================================
# CLINICAL SUMMARY
# ============================================================

st.header("Clinical summary")

if clinical_summary:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="decision-text">
                {str(clinical_summary).replace(chr(10), "<br>")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No clinical summary was returned.")


# ============================================================
# DECISION TRACE
# ============================================================

if decision_trace:
    st.header("Decision trace")

    if isinstance(decision_trace, list):

        for index, item in enumerate(decision_trace, start=1):

            if isinstance(item, dict):

                stage = (
                    item.get("stage")
                    or item.get("agent")
                    or f"Step {index}"
                )

                details = (
                    item.get("result")
                    or item.get("decision")
                    or item.get("details")
                    or item
                )

                st.markdown(
                    f"**{index}. {stage}**"
                )

                if isinstance(details, (dict, list)):
                    st.json(details)
                else:
                    st.write(details)

            else:
                st.write(f"{index}. {item}")

    elif isinstance(decision_trace, dict):
        st.json(decision_trace)

    else:
        st.write(decision_trace)


# ============================================================
# RAW DATA
# ============================================================

with st.expander("Request sent to API"):
    st.json(payload)

with st.expander("Complete API response"):
    st.json(result)