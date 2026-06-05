from groq import Groq
from embeddings import query_similar
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

client = Groq(api_key=GROQ_API_KEY)

RISK_SYSTEM_PROMPT = """You are a conservative clinical risk assessment AI grounded in NHS NICE guidelines.

Given one patient record + similar cases, return ONLY valid JSON:

{
  "risk_level": "Low" | "Medium" | "High" | "Critical",
  "risk_score": <int 0-100>,
  "summary": "<2 short sentences>",
  "risk_factors": ["list of real issues from data"],
  "protective_factors": ["list of positive factors"],
  "nice_recommendations": ["1-3 specific NICE-aligned actions with citation like (NG136 Hypertension)"],
  "similar_pattern": "<1 sentence>"
}

Rules:
- Be conservative. Only mark High/Critical if clear red flags (e.g. obesity + diabetes + emergency).
- Always reference actual patient values from the record.
- Include at least one NICE guideline citation when giving recommendations.
- Return ONLY the JSON. No extra text.
"""


def get_patient_record(patient_id: str, cleaned_csv_path: str, patient_id_col: str) -> dict | None:
    """Fetch the exact patient row from the cleaned CSV."""
    try:
        df = pd.read_csv(cleaned_csv_path)
        df = df.fillna("Unknown")

        if patient_id_col and patient_id_col in df.columns:
            match = df[df[patient_id_col].astype(str) == str(patient_id)]
            if not match.empty:
                return match.iloc[0].to_dict()

        # Fallback: treat patient_id as row index
        idx = int(patient_id)
        if idx < len(df):
            return df.iloc[idx].to_dict()

    except Exception:
        pass
    return None


def format_record_text(record: dict) -> str:
    parts = []
    for k, v in record.items():
        if str(v).lower() not in ("unknown", "nan", ""):
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


def assess_risk(patient_id: str, cleaned_csv_path: str, patient_id_col: str) -> dict:
    record = get_patient_record(patient_id, cleaned_csv_path, patient_id_col)
    if not record:
        return {"error": f"Patient '{patient_id}' not found in dataset."}

    patient_text = format_record_text(record)

    # Similar patients
    similar = query_similar(patient_text, n_results=5)
    similar_context = "\n".join(
        f"[Similar {i+1}]: {d['document'][:250]}"
        for i, d in enumerate(similar) if d["document"] != patient_text
    )

    prompt = f"""Patient Record:
{patient_text}

Similar patients:
{similar_context or "No similar cases found."}

Assess this patient's risk:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RISK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=700
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()

        import json
        risk_data = json.loads(raw)
        risk_data["patient_id"] = patient_id
        risk_data["record"] = record
        return risk_data

    except Exception as e:
        return {"error": f"Risk assessment failed: {str(e)}"}


def build_risk_html(risk: dict) -> str:
    if "error" in risk:
        return f"""<div class="card error-card">
            <div class="card-header"><span class="status-dot red"></span><h3>Assessment Failed</h3></div>
            <p class="error-msg">{risk['error']}</p>
        </div>"""

    level = risk.get("risk_level", "Unknown")
    score = risk.get("risk_score", 0)
    summary = risk.get("summary", "")
    risk_factors = risk.get("risk_factors", [])
    protective = risk.get("protective_factors", [])
    recommendations = risk.get("recommendations", [])
    similar_pattern = risk.get("similar_pattern", "")
    patient_id = risk.get("patient_id", "")
    record = risk.get("record", {})

    # Color by level
    level_colors = {
        "Low":      ("var(--green)",  "#1a3a1a", "🟢"),
        "Medium":   ("var(--yellow)", "#3a2e00", "🟡"),
        "High":     ("#e74c3c",       "#3a1a1a", "🔴"),
        "Critical": ("#ff4444",       "#4a0000", "🚨"),
    }
    color, bg_color, icon = level_colors.get(level, ("var(--accent)", "var(--bg3)", "⚪"))

    # Score bar
    score_bar = f"""
    <div class="score-bar-wrap">
        <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{score}%; background:{color};"></div>
        </div>
        <span class="score-label">{score}/100</span>
    </div>"""

    # Risk factors
    rf_html = "".join(f'<li class="risk-item bad">⚠ {f}</li>' for f in risk_factors)
    pf_html = "".join(f'<li class="risk-item good">✓ {f}</li>' for f in protective)
    rec_html = "".join(f'<li class="rec-item">→ {r}</li>' for r in recommendations)

    # Patient record table (top 12 fields)
    record_rows = ""
    for k, v in list(record.items())[:12]:
        if str(v).lower() not in ("unknown", "nan", ""):
            record_rows += f"<tr><td class='rec-key'>{k}</td><td>{v}</td></tr>"

    return f"""
    <div class="risk-card" style="border-color:{color}; background: linear-gradient(135deg, var(--bg2), {bg_color});">

        <div class="risk-header">
            <div>
                <div class="risk-patient-id">Patient: {patient_id}</div>
                <div class="risk-level-badge" style="background:{color}; color:#000;">
                    {icon} {level} Risk
                </div>
            </div>
            {score_bar}
        </div>

        <p class="risk-summary">{summary}</p>

        <div class="risk-columns">
            <div class="risk-col">
                <h4 class="risk-section-title" style="color:{color};">Risk Factors</h4>
                <ul class="risk-list">{rf_html}</ul>
            </div>
            <div class="risk-col">
                <h4 class="risk-section-title" style="color:var(--green);">Protective Factors</h4>
                <ul class="risk-list">{pf_html}</ul>
            </div>
        </div>

        <div style="margin-top:16px;">
            <h4 class="risk-section-title">Recommendations</h4>
            <ul class="rec-list">{rec_html}</ul>
        </div>

        {f'<div class="similar-pattern">🔍 <em>{similar_pattern}</em></div>' if similar_pattern else ''}

        <details class="source-details" style="margin-top:16px;">
            <summary>📋 View full patient record</summary>
            <div class="table-wrap" style="margin-top:10px;">
                <table class="col-table">
                    <thead><tr><th>Field</th><th>Value</th></tr></thead>
                    <tbody>{record_rows}</tbody>
                </table>
            </div>
        </details>
    </div>"""
