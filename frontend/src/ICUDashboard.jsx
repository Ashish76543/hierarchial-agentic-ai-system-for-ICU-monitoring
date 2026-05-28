import { useState, useEffect } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Theme
// ─────────────────────────────────────────────────────────────────────────────
const C = {
  bg: "#060a0f", panel: "#0a1018", panelAlt: "#0c1420",
  border: "#1a2535", borderBright: "#243550",
  green: "#00ff88", cyan: "#00d4ff", yellow: "#ffd600",
  red: "#ff3b5c", orange: "#ff8c00", white: "#e8f0fe",
  dim: "#2a3e58", dimText: "#5a7898", bodyText: "#8aaccc",
};

// ─────────────────────────────────────────────────────────────────────────────
// Physiological thresholds  [critLow, warnLow, warnHigh, critHigh]
// ─────────────────────────────────────────────────────────────────────────────
const THRESH = {
  heart_rate:     [40,   50,   120,  150],
  sbp:            [60,   90,   180,  220],
  dbp:            [40,   60,   110,  130],
  mbp:            [50,   65,   110,  130],
  sbp_ni:         [60,   90,   180,  220],
  dbp_ni:         [40,   60,   110,  130],
  mbp_ni:         [50,   65,   110,  130],
  spO2:           [85,   92,   null, null],
  temperature:    [35,   36,   38.3, 39.5],
  glucose:        [50,   70,   180,  250],
  sofa_24_hours:  [null, null, 6,    10],
  set_fio21:      [null, null, 60,   80],
  set_peep1:      [null, null, 10,   16],
  set_rr1:        [null, null, 25,   32],
  set_pc1:        [null, null, 25,   35],
  ppeak:          [null, null, 30,   40],
  set_tv1:        [null, null, 550,  700],
  los_outcome:    [null, null, 24,   72],
  shock_index:    [null, null, 0.9,  1.0],
  pulse_pressure: [null, null, null, null],
};

function colorFor(key, value) {
  if (typeof value !== "number") return C.white;
  const t = THRESH[key];
  if (!t) return C.white;
  const [cL, wL, wH, cH] = t;
  if (cL !== null && value <= cL) return C.red;
  if (cH !== null && value >= cH) return C.red;
  if (wL !== null && value <= wL) return C.orange;
  if (wH !== null && value >= wH) return C.orange;
  return C.green;
}
function isAlarm(key, value) {
  if (typeof value !== "number") return false;
  const t = THRESH[key];
  if (!t) return false;
  const [cL, wL, wH, cH] = t;
  return (cL !== null && value <= cL) || (cH !== null && value >= cH) ||
         (wL !== null && value <= wL) || (wH !== null && value >= wH);
}

// ─────────────────────────────────────────────────────────────────────────────
// Reusable primitives
// ─────────────────────────────────────────────────────────────────────────────

/** Inline colored value */
function V({ k, v, dec = 1 }) {
  const col = colorFor(k, v);
  const alarm = isAlarm(k, v);
  return (
    <span style={{ color: col, fontFamily: "monospace", fontWeight: 700,
      textShadow: alarm ? `0 0 10px ${col}77` : "none" }}>
      {typeof v === "number" ? v.toFixed(dec) : (v ?? "--")}
    </span>
  );
}

/** Large tile */
function BigTile({ label, threshKey, value, unit, dec = 1, note, span2 }) {
  const col = colorFor(threshKey, value);
  const alarm = isAlarm(threshKey, value);
  return (
    <div style={{
      background: C.panel, border: `1px solid ${alarm ? col : C.border}`,
      borderRadius: 4, padding: "12px 14px", position: "relative", overflow: "hidden",
      gridColumn: span2 ? "span 2" : undefined,
      boxShadow: alarm ? `0 0 16px ${col}22, inset 0 0 24px ${col}06` : "none",
    }}>
      {alarm && <div style={{ position:"absolute", top:0, left:0, right:0, height:2,
        background: col, animation: "alarmBar 0.9s ease-in-out infinite alternate" }} />}
      <div style={{ fontSize: 8, color: C.dimText, letterSpacing: 2,
        textTransform: "uppercase", fontFamily: "monospace", marginBottom: 5 }}>
        {label}{alarm && <span style={{ marginLeft: 8, color: col, animation: "blink 1s step-end infinite" }}>⚠</span>}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 5 }}>
        <span style={{ fontSize: 38, fontWeight: 700, color: col, lineHeight: 1,
          fontFamily: "'Courier New', monospace", letterSpacing: -1,
          textShadow: alarm ? `0 0 20px ${col}66` : "none" }}>
          {typeof value === "number" ? value.toFixed(dec) : (value ?? "--")}
        </span>
        <span style={{ fontSize: 11, color: C.dimText, fontFamily: "monospace" }}>{unit}</span>
      </div>
      {note && <div style={{ fontSize: 8, color: C.dim, fontFamily: "monospace", marginTop: 4 }}>{note}</div>}
    </div>
  );
}

/** Table row */
function Row({ label, threshKey, value, unit, dec = 1, highlight }) {
  const col = threshKey ? colorFor(threshKey, value) : C.white;
  const alarm = threshKey ? isAlarm(threshKey, value) : false;
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "8px 14px", borderBottom: `1px solid ${C.border}`,
      background: alarm ? `${col}09` : highlight ? C.panelAlt : "transparent",
    }}>
      <span style={{ fontSize: 9, color: alarm ? col : C.dimText, letterSpacing: 1,
        fontFamily: "monospace", textTransform: "uppercase" }}>{label}</span>
      <div style={{ display: "flex", alignItems: "baseline", gap: 5 }}>
        <span style={{ fontSize: 16, fontWeight: 700, color: col, fontFamily: "monospace",
          textShadow: alarm ? `0 0 8px ${col}55` : "none" }}>
          {typeof value === "number" ? value.toFixed(dec) : (value ?? "--")}
        </span>
        {unit && <span style={{ fontSize: 8, color: C.dimText, fontFamily: "monospace" }}>{unit}</span>}
        {alarm && <span style={{ fontSize: 8, color: col, animation: "blink 1s step-end infinite" }}>⚠</span>}
      </div>
    </div>
  );
}

/** Bool outcome row */
function BoolRow({ label, value, positiveIsTrue = true }) {
  const isBool = typeof value === "boolean";
  const col = isBool ? ((value === positiveIsTrue) ? C.green : C.red) : C.cyan;
  const display = isBool ? (value ? "YES" : "NO") : "--";
  const alarm = isBool && value !== positiveIsTrue;
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "9px 14px", borderBottom: `1px solid ${C.border}`,
      background: alarm ? `${col}09` : "transparent",
    }}>
      <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1, fontFamily: "monospace", textTransform: "uppercase" }}>{label}</span>
      <span style={{ fontSize: 15, fontWeight: 700, color: col, fontFamily: "monospace",
        textShadow: alarm ? `0 0 8px ${col}55` : "none" }}>{display}</span>
    </div>
  );
}

/** Intervention tile */
function IntTile({ label, active, critical }) {
  const col = active ? (critical ? C.red : C.orange) : C.dim;
  return (
    <div style={{
      border: `1px solid ${active ? col : C.border}`, borderRadius: 3,
      padding: "10px 6px", textAlign: "center",
      background: active ? `${col}10` : "transparent",
      boxShadow: active ? `0 0 12px ${col}28` : "none",
    }}>
      <div style={{ fontSize: 8, color: active ? col : C.dimText, letterSpacing: 1,
        fontFamily: "monospace", textTransform: "uppercase", marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 16, color: col, fontFamily: "monospace", fontWeight: 700 }}>
        {active ? "ON" : "OFF"}
      </div>
    </div>
  );
}

/** Bar gauge */
function Bar({ label, value, max, unit, warn, crit }) {
  if (value == null) return null;
  const col = value >= crit ? C.red : value >= warn ? C.orange : C.green;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1, fontFamily: "monospace" }}>{label}</span>
        <span style={{ fontSize: 11, color: col, fontFamily: "monospace", fontWeight: 700 }}>{value} {unit}</span>
      </div>
      <div style={{ height: 6, background: C.dim, borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(100, (value / max) * 100)}%`, height: "100%",
          background: col, borderRadius: 3,
          boxShadow: value >= crit ? `0 0 6px ${col}` : "none",
          transition: "width 0.4s ease" }} />
      </div>
    </div>
  );
}

/** Section header */
function SH({ title, color = C.cyan }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "16px 0 8px" }}>
      <div style={{ width: 3, height: 13, background: color, borderRadius: 2, flexShrink: 0 }} />
      <span style={{ fontSize: 9, color, letterSpacing: 3, textTransform: "uppercase",
        fontFamily: "monospace", fontWeight: 700 }}>{title}</span>
      <div style={{ flex: 1, height: 1, background: C.border }} />
    </div>
  );
}

/** Panel wrapper */
function Panel({ children, style }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`,
      borderRadius: 4, overflow: "hidden", ...style }}>
      {children}
    </div>
  );
}

/** Clock */
function Clock() {
  const [t, setT] = useState(new Date());
  useEffect(() => { const i = setInterval(() => setT(new Date()), 1000); return () => clearInterval(i); }, []);
  return <span style={{ fontFamily: "monospace", fontSize: 12, color: C.cyan }}>
    {t.toLocaleTimeString("en-US", { hour12: false })}
  </span>;
}

// ─────────────────────────────────────────────────────────────────────────────
// SAMPLE DATA — mirrors exact API response structure
// ─────────────────────────────────────────────────────────────────────────────
const SAMPLE = {
  forecasted_vitals: {
    heart_rate: 128.76, sbp: 80.31, dbp: 48.18, mbp: 57.61,
    temperature: 39.1, sbp_ni: 72.62, dbp_ni: 45.84, mbp_ni: 59.5,
    spO2: 82.25, glucose: 196.23, sofa_24_hours: 9.67,
  },
  predicted_outcomes: {
    discharge_outcome: false, icuouttime_outcome: true,
    death_outcome: true, sepsis_outcome: false, los_outcome: 24.79,
  },
  clinical_summary: "TREND: Patient deteriorating. SpO2 dropped from 89% to 82%. HR elevated to 128bpm. CURRENT STATUS: Critical hypoxemia, tachycardia, hemodynamic instability MBP 57mmHg. Invasive ventilation active. FORECAST: Continued worsening expected. OUTCOMES: HIGH mortality risk, ICU exit likely, no sepsis detected, estimated LOS 24.8h.",
  severity_scores: {
    shock_index: 1.60, hypotension: true, hypoxia: true, severe_hypoxia: true,
    acidosis: true, hypercapnia: false, hypoxemia_lab: true,
    neurological_impairment: false, high_fio2: true, high_peep: true,
  },
  detected_syndromes: {
    respiratory_failure: true, severe_respiratory_failure: true,
    circulatory_shock: true, neurological_failure: false, sepsis: false,
  },
  selected_protocols: {
    oxygen_protocol: "invasive", circulation_protocol: "vasopressor", renal_protocol: null,
  },
  recommended_interventions: { invasive: 1, noninvasive: 0, highflow: 0, vasopressor: 1, crrt: 0 },
  recommended_ventilator_changes: {
    mode: "INVASIVE", set_tv1: 420, total_tv: 420,
    set_fio21: 90, set_peep1: 16, total_peep: 16,
    set_rr1: 32, total_rr: 32, set_ie_ratio1: 2,
    set_pc1: 30, _pinsp_draeger2: 30, _pinsp_hamilton2: 30,
    _pcv_level2: 30, _set_pc_draeger: 30, ppeak: 30, rr: 32,
  },
  weaning_recommendation: null,
  escalation_decision: { invasive: 1, noninvasive: 0, highflow: 0, vasopressor: 1, crrt: 0 },
};

// ─────────────────────────────────────────────────────────────────────────────
// Input modal — sends input JSON directly to FastAPI, displays response
// ─────────────────────────────────────────────────────────────────────────────
const API_URL = 'http://localhost:8000/api/analyze';

function InputModal({ onSubmit, onClose }) {
  const [txt, setTxt] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const go = async () => {
    setErr("");
    let parsed;
    try {
      parsed = JSON.parse(txt);
    } catch (e) {
      setErr("Invalid JSON: " + e.message);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      });
      if (!res.ok) {
        const detail = await res.text();
        setErr("API error " + res.status + ": " + detail);
        setLoading(false);
        return;
      }
      const result = await res.json();
      onSubmit(result);
      onClose();
    } catch (e) {
      setErr("Could not reach FastAPI at " + API_URL + ". Is the server running? " + e.message);
    }
    setLoading(false);
  };

  return (
    <div style={{ position: "fixed", inset: 0, background: "#000000dc", zIndex: 1000,
      display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ background: C.panel, border: `1px solid ${C.cyan}55`, borderRadius: 6,
        width: "min(780px, 96vw)", maxHeight: "90vh", overflow: "auto", padding: 22 }}>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          <span style={{ color: C.cyan, fontSize: 10, letterSpacing: 3, fontFamily: "monospace" }}>
            PASTE PATIENT INPUT JSON
          </span>
          <button onClick={onClose} style={{ background: "none", border: "none", color: C.dimText, cursor: "pointer", fontSize: 18 }}>✕</button>
        </div>

        <div style={{ fontSize: 8, color: C.dimText, fontFamily: "monospace", marginBottom: 10, letterSpacing: 1 }}>
          Paste the full input payload (static, vitals, labs, gcs, ventilator, interventions, prior_*).
          Will POST to <span style={{ color: C.cyan }}>{API_URL}</span> and display the result.
        </div>

        <textarea
          value={txt}
          onChange={e => { setTxt(e.target.value); setErr(""); }}
          placeholder='{ "static": {...}, "vitals": {...}, ... }'
          style={{ width: "100%", height: 400, background: C.bg, border: `1px solid ${C.border}`,
            color: C.white, fontFamily: "monospace", fontSize: 11, padding: 10, borderRadius: 3,
            resize: "vertical", outline: "none", boxSizing: "border-box", lineHeight: 1.6 }}
        />

        {err && (
          <div style={{ color: C.red, fontSize: 9, marginTop: 8, fontFamily: "monospace",
            background: "#1a0509", border: `1px solid ${C.red}44`, borderRadius: 3,
            padding: "8px 12px", lineHeight: 1.6, wordBreak: "break-all" }}>
            {err}
          </div>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12, alignItems: "center" }}>
          {loading && (
            <span style={{ fontSize: 9, color: C.yellow, fontFamily: "monospace",
              letterSpacing: 2, animation: "blink 1s step-end infinite" }}>
              CALLING API — PLEASE WAIT...
            </span>
          )}
          <button onClick={onClose} disabled={loading} style={{ padding: "7px 16px", background: "none",
            border: `1px solid ${C.border}`, color: C.dimText, cursor: "pointer",
            fontFamily: "monospace", fontSize: 9, borderRadius: 3 }}>CANCEL</button>
          <button onClick={go} disabled={loading} style={{ padding: "7px 20px",
            background: loading ? `${C.dim}22` : `${C.cyan}18`,
            border: `1px solid ${loading ? C.dim : C.cyan}`,
            color: loading ? C.dim : C.cyan,
            cursor: loading ? "not-allowed" : "pointer",
            fontFamily: "monospace", fontSize: 9, borderRadius: 3 }}>
            {loading ? "ANALYZING..." : "ANALYZE →"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE 1 — OVERVIEW
// Predicted outcomes · SOFA · Clinical summary · Key alarms summary
// ─────────────────────────────────────────────────────────────────────────────
function PageOverview({ data }) {
  const fv = data?.forecasted_vitals || {};
  const po = data?.predicted_outcomes || {};
  const ss = data?.severity_scores || {};

  const shockIndex = fv.heart_rate && fv.sbp ? fv.heart_rate / fv.sbp : null;
  const pulsePressure = fv.sbp && fv.dbp ? fv.sbp - fv.dbp : null;

  // All alarm states for summary strip
  const alarms = [
    { label: "HR",    alarm: isAlarm("heart_rate", fv.heart_rate),   val: fv.heart_rate?.toFixed(0),   unit: "bpm" },
    { label: "SpO₂",  alarm: isAlarm("spO2", fv.spO2),               val: fv.spO2?.toFixed(1),          unit: "%" },
    { label: "MBP",   alarm: isAlarm("mbp", fv.mbp),                 val: fv.mbp?.toFixed(1),           unit: "mmHg" },
    { label: "SBP",   alarm: isAlarm("sbp", fv.sbp),                 val: fv.sbp?.toFixed(1),           unit: "mmHg" },
    { label: "SOFA",  alarm: isAlarm("sofa_24_hours", fv.sofa_24_hours), val: fv.sofa_24_hours?.toFixed(1), unit: "" },
    { label: "TEMP",  alarm: isAlarm("temperature", fv.temperature), val: fv.temperature?.toFixed(1),   unit: "°C" },
    { label: "GLUC",  alarm: isAlarm("glucose", fv.glucose),         val: fv.glucose?.toFixed(0),       unit: "mg/dL" },
  ].filter(a => a.alarm);

  return (
    <div>
      {/* Active alarms strip */}
      {alarms.length > 0 && (
        <div style={{ background: "#0f0609", border: `1px solid ${C.red}44`, borderRadius: 4, padding: "10px 14px", marginBottom: 16, display: "flex", gap: 10, flexWrap: "wrap" }}>
          <span style={{ fontSize: 8, color: C.red, letterSpacing: 2, fontFamily: "monospace", alignSelf: "center" }}>ACTIVE ALARMS</span>
          {alarms.map(a => (
            <div key={a.label} style={{ padding: "3px 10px", border: `1px solid ${C.red}`, borderRadius: 3,
              background: `${C.red}10`, animation: "alarmBar 0.9s ease-in-out infinite alternate" }}>
              <span style={{ fontSize: 9, color: C.red, fontFamily: "monospace", fontWeight: 700 }}>
                {a.label}: {a.val} {a.unit}
              </span>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>

        {/* Col 1: Predicted outcomes */}
        <div>
          <SH title="Predicted Outcomes" color={C.yellow} />
          <Panel>
            <BoolRow label="Discharge" value={po.discharge_outcome} positiveIsTrue />
            <BoolRow label="ICU Exit" value={po.icuouttime_outcome} positiveIsTrue />
            <BoolRow label="Death Risk" value={po.death_outcome} positiveIsTrue={false} />
            <BoolRow label="Sepsis" value={po.sepsis_outcome} positiveIsTrue={false} />
            <Row label="Estimated LOS" threshKey="los_outcome" value={po.los_outcome} unit="hrs" dec={1} />
          </Panel>

          <SH title="Forecasted Key Values" color={C.green} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <BigTile label="Heart Rate" threshKey="heart_rate" value={fv.heart_rate} unit="bpm" dec={1} note="Normal 50–120" />
            <BigTile label="SpO₂" threshKey="spO2" value={fv.spO2} unit="%" dec={1} note="Target ≥ 92%" />
            <BigTile label="MBP (inv)" threshKey="mbp" value={fv.mbp} unit="mmHg" dec={1} note="Target ≥ 65" />
            <BigTile label="SOFA Score" threshKey="sofa_24_hours" value={fv.sofa_24_hours} unit="/24" dec={1} note="Alarm ≥ 6" />
          </div>
        </div>

        {/* Col 2: Severity scores */}
        <div>
          <SH title="Severity Scores" color={C.red} />
          <Panel>
            {shockIndex != null && (
              <Row label="Shock Index (HR/SBP)" value={shockIndex} dec={2} unit=""
                threshKey="shock_index" />
            )}
            {pulsePressure != null && (
              <Row label="Pulse Pressure (SBP–DBP)" value={pulsePressure} dec={0} unit="mmHg"
                threshKey={null} />
            )}
            <Row label="SOFA Score" threshKey="sofa_24_hours" value={fv.sofa_24_hours} unit="/24" dec={1} />
            <Row label="Temperature" threshKey="temperature" value={fv.temperature} unit="°C" dec={1} />
            <Row label="Glucose" threshKey="glucose" value={fv.glucose} unit="mg/dL" dec={0} />
          </Panel>

          <SH title="Severity Flags" color={C.orange} />
          <Panel>
            {Object.entries(ss).filter(([,v]) => typeof v === "boolean").map(([k, v]) => (
              <div key={k} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "7px 14px", borderBottom: `1px solid ${C.border}`,
                background: v ? `${C.red}08` : "transparent",
              }}>
                <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1,
                  fontFamily: "monospace", textTransform: "uppercase" }}>
                  {k.replace(/_/g, " ")}
                </span>
                <span style={{ fontSize: 11, fontWeight: 700,
                  color: v ? C.red : C.green, fontFamily: "monospace" }}>
                  {v ? "TRUE" : "FALSE"}
                </span>
              </div>
            ))}
          </Panel>
        </div>

        {/* Col 3: Clinical summary */}
        <div>
          <SH title="Clinical Summary" color={C.cyan} />
          <Panel style={{ padding: "14px 16px" }}>
            <p style={{ margin: 0, fontSize: 10, color: C.bodyText, lineHeight: 1.8, fontFamily: "monospace" }}>
              {data?.clinical_summary || "No clinical summary loaded."}
            </p>
          </Panel>

          <SH title="Detected Syndromes" color={C.red} />
          <Panel>
            {Object.entries(data?.detected_syndromes || {}).map(([k, v]) => (
              <div key={k} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "7px 14px", borderBottom: `1px solid ${C.border}`,
                background: v ? `${C.red}08` : "transparent",
              }}>
                <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1,
                  fontFamily: "monospace", textTransform: "uppercase" }}>
                  {k.replace(/_/g, " ")}
                </span>
                <span style={{ fontSize: 11, fontWeight: 700,
                  color: v ? C.red : C.green, fontFamily: "monospace" }}>
                  {v ? "DETECTED" : "CLEAR"}
                </span>
              </div>
            ))}
          </Panel>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE 2 — HEMODYNAMICS
// All BP (invasive + NIBP) · HR · derived indices · temp · glucose
// ─────────────────────────────────────────────────────────────────────────────
function PageHemodynamics({ data }) {
  const fv = data?.forecasted_vitals || {};
  const shockIndex = fv.heart_rate && fv.sbp ? fv.heart_rate / fv.sbp : null;
  const pulsePressure = fv.sbp && fv.dbp ? fv.sbp - fv.dbp : null;

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>

        {/* Col 1: Invasive BP + HR */}
        <div>
          <SH title="Invasive Arterial Line" color={C.orange} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 14 }}>
            <BigTile label="HR" threshKey="heart_rate" value={fv.heart_rate} unit="bpm" dec={1} note="Normal 50–120" />
            <BigTile label="SBP" threshKey="sbp" value={fv.sbp} unit="mmHg" dec={1} note="Normal 90–180" />
            <BigTile label="DBP" threshKey="dbp" value={fv.dbp} unit="mmHg" dec={1} note="Normal 60–110" />
            <BigTile label="MBP" threshKey="mbp" value={fv.mbp} unit="mmHg" dec={1} note="Target ≥ 65" />
          </div>

          <SH title="Non-Invasive BP (NIBP)" color={C.cyan} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
            <BigTile label="SBP-NI" threshKey="sbp_ni" value={fv.sbp_ni} unit="mmHg" dec={1} />
            <BigTile label="DBP-NI" threshKey="dbp_ni" value={fv.dbp_ni} unit="mmHg" dec={1} />
            <BigTile label="MBP-NI" threshKey="mbp_ni" value={fv.mbp_ni} unit="mmHg" dec={1} />
          </div>
        </div>

        {/* Col 2: Derived + comparison table */}
        <div>
          <SH title="Derived Hemodynamic Indices" color={C.yellow} />
          <Panel style={{ marginBottom: 14 }}>
            <Row label="Heart Rate" threshKey="heart_rate" value={fv.heart_rate} unit="bpm" dec={1} />
            <Row label="Shock Index (HR ÷ SBP)" value={shockIndex} dec={2} unit=""
              threshKey="shock_index" />
            <Row label="Pulse Pressure (SBP − DBP)" value={pulsePressure} dec={0} unit="mmHg" />
            <Row label="MAP (invasive)" threshKey="mbp" value={fv.mbp} unit="mmHg" dec={1} />
            <Row label="MAP (non-invasive)" threshKey="mbp_ni" value={fv.mbp_ni} unit="mmHg" dec={1} />
          </Panel>

          <SH title="Invasive vs Non-Invasive BP Delta" color={C.dim} />
          <Panel>
            {[
              ["SBP delta", fv.sbp, fv.sbp_ni],
              ["DBP delta", fv.dbp, fv.dbp_ni],
              ["MBP delta", fv.mbp, fv.mbp_ni],
            ].map(([label, inv, ni]) => {
              const delta = (inv != null && ni != null) ? (inv - ni) : null;
              const col = Math.abs(delta) > 15 ? C.orange : C.green;
              return (
                <div key={label} style={{ display: "flex", justifyContent: "space-between",
                  padding: "7px 14px", borderBottom: `1px solid ${C.border}` }}>
                  <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1, fontFamily: "monospace", textTransform: "uppercase" }}>{label}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: col, fontFamily: "monospace" }}>
                    {delta != null ? (delta >= 0 ? "+" : "") + delta.toFixed(1) : "--"} <span style={{ fontSize: 8, color: C.dimText }}>mmHg</span>
                  </span>
                </div>
              );
            })}
          </Panel>
        </div>

        {/* Col 3: Other forecasted vitals */}
        <div>
          <SH title="Other Forecasted Vitals" color={C.green} />
          <Panel style={{ marginBottom: 14 }}>
            <Row label="Temperature" threshKey="temperature" value={fv.temperature} unit="°C" dec={1} />
            <Row label="Glucose" threshKey="glucose" value={fv.glucose} unit="mg/dL" dec={0} />
            <Row label="SOFA Score" threshKey="sofa_24_hours" value={fv.sofa_24_hours} unit="/24" dec={1} />
            <Row label="SpO₂" threshKey="spO2" value={fv.spO2} unit="%" dec={1} />
          </Panel>

          <SH title="Complete Forecasted Vitals Reference" color={C.dim} />
          <Panel>
            {Object.entries(data?.forecasted_vitals || {}).map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between",
                padding: "5px 14px", borderBottom: `1px solid ${C.border}` }}>
                <span style={{ fontSize: 8, color: C.dimText, fontFamily: "monospace",
                  textTransform: "uppercase", letterSpacing: 1 }}>{k.replace(/_/g, " ")}</span>
                <span style={{ fontSize: 12, fontWeight: 700, fontFamily: "monospace",
                  color: colorFor(k, v) }}>
                  {typeof v === "number" ? v.toFixed(4) : String(v ?? "--")}
                </span>
              </div>
            ))}
          </Panel>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE 3 — RESPIRATORY & VENTILATOR
// SpO₂ · all ventilator settings (every field from API) · risk bars
// ─────────────────────────────────────────────────────────────────────────────
function PageRespVent({ data }) {
  const fv = data?.forecasted_vitals || {};
  const rv = data?.recommended_ventilator_changes || {};

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>

        {/* Col 1: Oxygenation */}
        <div>
          <SH title="Oxygenation" color={C.cyan} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 14 }}>
            <BigTile label="SpO₂" threshKey="spO2" value={fv.spO2} unit="%" dec={1} note="Target ≥ 92%" span2 />
            <BigTile label="FiO₂ Set" threshKey="set_fio21" value={rv.set_fio21} unit="%" dec={0} note="Alarm ≥ 60%" />
            <BigTile label="PEEP" threshKey="set_peep1" value={rv.set_peep1} unit="cmH₂O" dec={0} note="Alarm ≥ 10" />
          </div>

          <SH title="Vent Risk Gauges" color={C.red} />
          <Panel style={{ padding: "14px 16px" }}>
            <Bar label="FiO₂" value={rv.set_fio21} max={100} unit="%" warn={60} crit={80} />
            <Bar label="PEEP" value={rv.set_peep1} max={25} unit="cmH₂O" warn={10} crit={16} />
            <Bar label="Respiratory Rate" value={rv.set_rr1} max={40} unit="/min" warn={25} crit={32} />
            <Bar label="Peak Pressure" value={rv.ppeak} max={50} unit="cmH₂O" warn={30} crit={40} />
            <Bar label="Insp. Pressure (PC)" value={rv.set_pc1} max={45} unit="cmH₂O" warn={25} crit={35} />
          </Panel>
        </div>

        {/* Col 2: Primary vent settings */}
        <div>
          <SH title="Ventilator — Set Parameters" color={C.yellow} />
          <Panel style={{ marginBottom: 14 }}>
            <div style={{ padding: "7px 14px", borderBottom: `1px solid ${C.border}`, background: C.panelAlt }}>
              <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1, fontFamily: "monospace" }}>MODE</span>
              <span style={{ fontSize: 16, color: C.cyan, fontFamily: "monospace", fontWeight: 700, marginLeft: 12 }}>{rv.mode ?? "--"}</span>
            </div>
            <Row label="FiO₂ (set_fio21)" threshKey="set_fio21" value={rv.set_fio21} unit="%" dec={0} />
            <Row label="PEEP (set_peep1)" threshKey="set_peep1" value={rv.set_peep1} unit="cmH₂O" dec={0} />
            <Row label="Total PEEP" threshKey="set_peep1" value={rv.total_peep} unit="cmH₂O" dec={0} />
            <Row label="Resp Rate (set_rr1)" threshKey="set_rr1" value={rv.set_rr1} unit="/min" dec={0} />
            <Row label="Total RR" threshKey="set_rr1" value={rv.total_rr} unit="/min" dec={0} />
            <Row label="Tidal Volume (set_tv1)" threshKey="set_tv1" value={rv.set_tv1} unit="mL" dec={0} />
            <Row label="Total TV" threshKey="set_tv1" value={rv.total_tv} unit="mL" dec={0} />
            <Row label="I:E Ratio" value={rv.set_ie_ratio1 ? `1:${rv.set_ie_ratio1}` : null} unit="" dec={0} />
          </Panel>
        </div>

        {/* Col 3: Pressure settings — all machine-specific fields */}
        <div>
          <SH title="Ventilator — Pressure Parameters" color={C.orange} />
          <Panel style={{ marginBottom: 14 }}>
            <Row label="PC Level (set_pc1)" threshKey="set_pc1" value={rv.set_pc1} unit="cmH₂O" dec={0} />
            <Row label="Pinsp Draeger (_pinsp_draeger2)" threshKey="set_pc1" value={rv._pinsp_draeger2} unit="cmH₂O" dec={0} />
            <Row label="Pinsp Hamilton (_pinsp_hamilton2)" threshKey="set_pc1" value={rv._pinsp_hamilton2} unit="cmH₂O" dec={0} />
            <Row label="PCV Level (_pcv_level2)" threshKey="set_pc1" value={rv._pcv_level2} unit="cmH₂O" dec={0} />
            <Row label="PC Draeger (_set_pc_draeger)" threshKey="set_pc1" value={rv._set_pc_draeger} unit="cmH₂O" dec={0} />
            <Row label="Peak Pressure (ppeak)" threshKey="ppeak" value={rv.ppeak} unit="cmH₂O" dec={0} />
            <Row label="RR (rr)" threshKey="set_rr1" value={rv.rr} unit="/min" dec={0} />
          </Panel>

          <SH title="Weaning Recommendation" color={C.green} />
          <Panel>
            {data?.weaning_recommendation == null ? (
              <div style={{ padding: "12px 14px" }}>
                <span style={{ fontSize: 10, color: C.dimText, fontFamily: "monospace" }}>
                  No weaning indicated at this time.
                </span>
              </div>
            ) : (
              Object.entries(data.weaning_recommendation).map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between",
                  padding: "7px 14px", borderBottom: `1px solid ${C.border}` }}>
                  <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1,
                    fontFamily: "monospace", textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</span>
                  <span style={{ fontSize: 13, fontWeight: 700,
                    color: v === 1 ? C.orange : C.green, fontFamily: "monospace" }}>
                    {v === 1 ? "ON" : v === 0 ? "OFF" : String(v)}
                  </span>
                </div>
              ))
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE 4 — CLINICAL DECISIONS
// Interventions · escalation · protocols · weaning · all decision outputs
// ─────────────────────────────────────────────────────────────────────────────
function PageDecisions({ data }) {
  const ri = data?.recommended_interventions || {};
  const ed = data?.escalation_decision || {};
  const sp = data?.selected_protocols || {};
  const wr = data?.weaning_recommendation;

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>

        {/* Col 1: Recommended interventions */}
        <div>
          <SH title="Recommended Interventions" color={C.orange} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 16 }}>
            <IntTile label="Invasive Vent" active={!!ri.invasive} critical />
            <IntTile label="Non-Invasive" active={!!ri.noninvasive} />
            <IntTile label="High Flow O₂" active={!!ri.highflow} />
            <IntTile label="Vasopressor" active={!!ri.vasopressor} critical />
            <IntTile label="CRRT" active={!!ri.crrt} critical />
            <div style={{ border: `1px solid ${C.border}`, borderRadius: 3,
              padding: "10px 6px", textAlign: "center" }}>
              <div style={{ fontSize: 8, color: C.dimText, letterSpacing: 1, fontFamily: "monospace", marginBottom: 3 }}>VENT MODE</div>
              <div style={{ fontSize: 13, color: data?.recommended_ventilator_changes?.mode ? C.cyan : C.dimText,
                fontFamily: "monospace", fontWeight: 700 }}>
                {data?.recommended_ventilator_changes?.mode || "--"}
              </div>
            </div>
          </div>

          <SH title="Raw Intervention Values" color={C.dim} />
          <Panel>
            {Object.entries(ri).map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between",
                padding: "6px 14px", borderBottom: `1px solid ${C.border}`,
                background: v === 1 ? `${C.orange}08` : "transparent" }}>
                <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1,
                  fontFamily: "monospace", textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</span>
                <span style={{ fontSize: 13, fontWeight: 700,
                  color: v === 1 ? C.orange : C.dim, fontFamily: "monospace" }}>{v}</span>
              </div>
            ))}
          </Panel>
        </div>

        {/* Col 2: Escalation + Protocols */}
        <div>
          <SH title="Escalation Decision" color={C.red} />
          <Panel style={{ marginBottom: 14 }}>
            {Object.keys(ed).length === 0 ? (
              <div style={{ padding: "12px 14px" }}>
                <span style={{ fontSize: 10, color: C.dimText, fontFamily: "monospace" }}>No escalation data.</span>
              </div>
            ) : Object.entries(ed).map(([k, v]) => {
              const isOn = v === 1;
              const col = isOn ? C.red : C.green;
              return (
                <div key={k} style={{ display: "flex", justifyContent: "space-between",
                  alignItems: "center", padding: "8px 14px", borderBottom: `1px solid ${C.border}`,
                  background: isOn ? `${C.red}09` : "transparent" }}>
                  <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1,
                    fontFamily: "monospace", textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: col, fontFamily: "monospace",
                    textShadow: isOn ? `0 0 8px ${col}55` : "none" }}>
                    {isOn ? "ESCALATE" : "HOLD"}
                  </span>
                </div>
              );
            })}
          </Panel>

          <SH title="Selected Protocols" color={C.cyan} />
          <Panel>
            {Object.keys(sp).length === 0 ? (
              <div style={{ padding: "12px 14px" }}>
                <span style={{ fontSize: 10, color: C.dimText, fontFamily: "monospace" }}>No protocols selected.</span>
              </div>
            ) : Object.entries(sp).map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between",
                alignItems: "center", padding: "8px 14px", borderBottom: `1px solid ${C.border}`,
                background: v ? `${C.cyan}08` : "transparent" }}>
                <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1,
                  fontFamily: "monospace", textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</span>
                <span style={{ fontSize: 13, fontWeight: 700,
                  color: v ? C.cyan : C.dim, fontFamily: "monospace" }}>
                  {v ? String(v).toUpperCase() : "NONE"}
                </span>
              </div>
            ))}
          </Panel>
        </div>

        {/* Col 3: Weaning + syndrome summary */}
        <div>
          <SH title="Weaning Recommendation" color={C.green} />
          <Panel style={{ marginBottom: 14 }}>
            {wr == null ? (
              <div style={{ padding: "12px 14px" }}>
                <span style={{ fontSize: 10, color: C.dimText, fontFamily: "monospace" }}>
                  No weaning recommended. Patient not currently meeting weaning criteria.
                </span>
              </div>
            ) : Object.entries(wr).map(([k, v]) => {
              const col = v === 1 ? C.orange : C.green;
              return (
                <div key={k} style={{ display: "flex", justifyContent: "space-between",
                  alignItems: "center", padding: "8px 14px", borderBottom: `1px solid ${C.border}` }}>
                  <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1,
                    fontFamily: "monospace", textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: col, fontFamily: "monospace" }}>
                    {v === 1 ? "ON" : v === 0 ? "OFF" : String(v)}
                  </span>
                </div>
              );
            })}
          </Panel>

          <SH title="Detected Syndromes" color={C.red} />
          <Panel style={{ marginBottom: 14 }}>
            {Object.entries(data?.detected_syndromes || {}).map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between",
                alignItems: "center", padding: "7px 14px", borderBottom: `1px solid ${C.border}`,
                background: v ? `${C.red}08` : "transparent" }}>
                <span style={{ fontSize: 9, color: C.dimText, letterSpacing: 1,
                  fontFamily: "monospace", textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</span>
                <span style={{ fontSize: 11, fontWeight: 700,
                  color: v ? C.red : C.green, fontFamily: "monospace" }}>
                  {v ? "DETECTED" : "CLEAR"}
                </span>
              </div>
            ))}
          </Panel>

          <SH title="Outcomes Summary" color={C.yellow} />
          <Panel>
            <BoolRow label="Discharge" value={data?.predicted_outcomes?.discharge_outcome} positiveIsTrue />
            <BoolRow label="ICU Exit" value={data?.predicted_outcomes?.icuouttime_outcome} positiveIsTrue />
            <BoolRow label="Death Risk" value={data?.predicted_outcomes?.death_outcome} positiveIsTrue={false} />
            <BoolRow label="Sepsis" value={data?.predicted_outcomes?.sepsis_outcome} positiveIsTrue={false} />
            <Row label="LOS" threshKey="los_outcome" value={data?.predicted_outcomes?.los_outcome} unit="hrs" dec={1} />
          </Panel>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP — nav + shared header
// ─────────────────────────────────────────────────────────────────────────────
const PAGES = [
  { id: "overview",     label: "Overview",          short: "01" },
  { id: "hemodynamics", label: "Hemodynamics",       short: "02" },
  { id: "respiratory",  label: "Resp & Ventilator",  short: "03" },
  { id: "decisions",    label: "Clinical Decisions", short: "04" },
];

export default function ICUDashboard() {
  const [data, setData] = useState(null);
  const [page, setPage] = useState("overview");
  const [showInput, setShowInput] = useState(false);

  const handleLoad = (parsed) => {
    setData(parsed);
    setShowInput(false);
  };

  const po = data?.predicted_outcomes || {};
  const critAlarm = po.death_outcome === true;

  const alarmCount = [
    isAlarm("heart_rate", data?.forecasted_vitals?.heart_rate),
    isAlarm("spO2",       data?.forecasted_vitals?.spO2),
    isAlarm("mbp",        data?.forecasted_vitals?.mbp),
    isAlarm("sbp",        data?.forecasted_vitals?.sbp),
    isAlarm("temperature",data?.forecasted_vitals?.temperature),
    isAlarm("glucose",    data?.forecasted_vitals?.glucose),
    isAlarm("sofa_24_hours", data?.forecasted_vitals?.sofa_24_hours),
  ].filter(Boolean).length;

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.white, fontFamily: "'Courier New', monospace" }}>
      <style>{`
        @keyframes alarmBar  { from{opacity:1} to{opacity:0.15} }
        @keyframes blink     { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes critPulse { 0%,100%{background:#070d15} 50%{background:#ff3b5c0e} }
        @keyframes tickScroll { from{transform:translateX(100vw)} to{transform:translateX(-100%)} }
        * { box-sizing:border-box; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:#060a0f; }
        ::-webkit-scrollbar-thumb { background:#1a2535; border-radius:2px; }
      `}</style>

      {showInput && <InputModal onSubmit={handleLoad} onClose={() => setShowInput(false)} />}

      {/* ── Top header bar ── */}
      <div style={{
        background: "#070d15", borderBottom: `1px solid ${C.border}`,
        padding: "0 16px", height: 46,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        animation: critAlarm ? "critPulse 1.4s ease-in-out infinite" : "none",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%",
              background: C.green, boxShadow: `0 0 8px ${C.green}` }} />
            <span style={{ fontSize: 11, letterSpacing: 4, color: C.white, fontWeight: 700 }}>
              ICU DECISION SUPPORT
            </span>
          </div>
          <span style={{ fontSize: 8, color: C.dimText, letterSpacing: 2,
            borderLeft: `1px solid ${C.border}`, paddingLeft: 14 }}>
            LANGGRAPH · BIOMISTRAL-7B · CRITICAL CARE
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {alarmCount > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "3px 10px",
              border: `1px solid ${C.red}`, borderRadius: 3,
              animation: "alarmBar 0.8s ease-in-out infinite alternate" }}>
              <span style={{ fontSize: 8, color: C.red, letterSpacing: 2 }}>
                {alarmCount} ACTIVE ALARM{alarmCount > 1 ? "S" : ""}
              </span>
            </div>
          )}
          <button onClick={() => setShowInput(true)} style={{
            background: `${C.cyan}11`, border: `1px solid ${C.cyan}44`,
            color: C.cyan, cursor: "pointer", fontSize: 8, letterSpacing: 2,
            padding: "4px 12px", borderRadius: 2, fontFamily: "monospace"
          }}>LOAD DATA</button>
          <Clock />
        </div>
      </div>

      {/* ── Clinical summary ticker ── */}
      <div style={{ background: "#050c14", borderBottom: `1px solid ${C.border}`,
        height: 24, display: "flex", alignItems: "center", overflow: "hidden" }}>
        <span style={{ fontSize: 8, color: C.yellow, letterSpacing: 2, padding: "0 12px",
          borderRight: `1px solid ${C.border}`, flexShrink: 0, fontFamily: "monospace" }}>SUMMARY</span>
        <div style={{ overflow: "hidden", flex: 1, position: "relative", height: "100%",
          display: "flex", alignItems: "center" }}>
          <div style={{ whiteSpace: "nowrap", fontSize: 8, color: C.dimText,
            fontFamily: "monospace", animation: "tickScroll 50s linear infinite", position: "absolute" }}>
            {data?.clinical_summary || "No summary loaded."}
          </div>
        </div>
      </div>

      {/* ── Page navigation tabs ── */}
      <div style={{ background: "#070d15", borderBottom: `1px solid ${C.border}`,
        display: "flex", alignItems: "stretch", padding: "0 16px" }}>
        {PAGES.map(p => {
          const active = page === p.id;
          return (
            <button key={p.id} onClick={() => setPage(p.id)} style={{
              background: active ? C.panel : "none",
              border: "none",
              borderBottom: active ? `2px solid ${C.cyan}` : "2px solid transparent",
              borderTop: "2px solid transparent",
              color: active ? C.cyan : C.dimText,
              cursor: "pointer", fontFamily: "monospace",
              fontSize: 9, letterSpacing: 2, textTransform: "uppercase",
              padding: "10px 20px", marginRight: 2,
              transition: "color 0.15s, border-color 0.15s",
            }}>
              <span style={{ marginRight: 8, opacity: 0.4, fontSize: 8 }}>{p.short}</span>
              {p.label}
            </button>
          );
        })}

        {/* Quick vitals in tab bar */}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 16 }}>
          {[
            ["HR", "heart_rate", data?.forecasted_vitals?.heart_rate, "bpm", 0],
            ["SpO₂", "spO2", data?.forecasted_vitals?.spO2, "%", 1],
            ["MBP", "mbp", data?.forecasted_vitals?.mbp, "mmHg", 0],
          ].map(([label, key, val, unit, dec]) => (
            <div key={label} style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
              <span style={{ fontSize: 7, color: C.dimText, fontFamily: "monospace", letterSpacing: 1 }}>{label}</span>
              <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "monospace",
                color: colorFor(key, val),
                textShadow: isAlarm(key, val) ? `0 0 8px ${colorFor(key, val)}88` : "none" }}>
                {typeof val === "number" ? val.toFixed(dec) : "--"}
              </span>
              <span style={{ fontSize: 7, color: C.dimText, fontFamily: "monospace" }}>{unit}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Page content ── */}
      <div style={{ padding: "14px 16px" }}>
        {!data ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60vh", gap: 16 }}>
            <div style={{ fontSize: 11, color: C.dimText, letterSpacing: 3, fontFamily: "monospace" }}>NO DATA LOADED</div>
            <div style={{ fontSize: 9, color: C.dim, fontFamily: "monospace", letterSpacing: 2 }}>Click LOAD DATA and paste your API JSON response</div>
            <button onClick={() => setShowInput(true)} style={{
              marginTop: 8, padding: "10px 28px", background: `${C.cyan}18`,
              border: `1px solid ${C.cyan}`, color: C.cyan, cursor: "pointer",
              fontFamily: "monospace", fontSize: 10, letterSpacing: 2, borderRadius: 3
            }}>LOAD DATA →</button>
          </div>
        ) : (
          <>
            {page === "overview"     && <PageOverview     data={data} />}
            {page === "hemodynamics" && <PageHemodynamics data={data} />}
            {page === "respiratory"  && <PageRespVent     data={data} />}
            {page === "decisions"    && <PageDecisions    data={data} />}
          </>
        )}
      </div>

      {/* ── Footer ── */}
      <div style={{ borderTop: `1px solid ${C.border}`, padding: "6px 16px",
        display: "flex", justifyContent: "space-between" }}>
        <span style={{ fontSize: 8, color: C.dim, letterSpacing: 2, fontFamily: "monospace" }}>
          ICU DECISION SUPPORT · LANGGRAPH + BIOMISTRAL-7B-GGUF
        </span>
        <span style={{ fontSize: 8, color: C.dim, letterSpacing: 1, fontFamily: "monospace" }}>
          AI-GENERATED OUTPUT — CLINICAL OVERSIGHT MANDATORY
        </span>
      </div>
    </div>
  );
}