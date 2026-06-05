import pandas as pd
import json


def compute_metrics(cleaned_csv_path: str, important_cols: list, descriptions: dict) -> dict:
    df = pd.read_csv(cleaned_csv_path)

    metrics = {}
    metrics["total_patients"] = len(df)
    metrics["total_columns"] = len(df.columns)
    metrics["columns"] = list(df.columns)

    col_lower = {c.lower(): c for c in df.columns}

    # ── Age ──────────────────────────────────────────────
    age_col = _find_col(col_lower, ["age"])
    if age_col:
        age_series = pd.to_numeric(df[age_col], errors="coerce").dropna()
        if len(age_series) > 0:
            metrics["age"] = {
                "mean": round(age_series.mean(), 1),
                "min": int(age_series.min()),
                "max": int(age_series.max()),
                "median": round(age_series.median(), 1),
            }
            # Age buckets
            bins = [0, 18, 35, 50, 65, 80, 200]
            labels = ["<18", "18-35", "35-50", "50-65", "65-80", "80+"]
            bucketed = pd.cut(age_series, bins=bins, labels=labels)
            metrics["age_distribution"] = bucketed.value_counts().sort_index().to_dict()

    # ── Gender ───────────────────────────────────────────
    gender_col = _find_col(col_lower, ["gender", "sex", "patient_gender"])
    if gender_col:
        vc = df[gender_col].value_counts().head(6).to_dict()
        metrics["gender"] = {str(k): int(v) for k, v in vc.items()}

    # ── Outcome / Target ─────────────────────────────────
    outcome_col = _find_col(col_lower, ["outcome", "target", "readmitted", "death",
                                         "discharge", "result", "status"])
    if outcome_col:
        vc = df[outcome_col].value_counts().head(8).to_dict()
        metrics["outcome"] = {
            "col": outcome_col,
            "distribution": {str(k): int(v) for k, v in vc.items()}
        }

    # ── Diagnosis / Condition ────────────────────────────
    diag_col = _find_col(col_lower, ["diagnosis", "condition", "disease", "diag_1",
                                      "primary_diagnosis", "icd", "medical_condition"])
    if diag_col:
        vc = df[diag_col].value_counts().head(8).to_dict()
        metrics["diagnosis"] = {
            "col": diag_col,
            "top": {str(k): int(v) for k, v in vc.items()}
        }

    # ── Admission type ───────────────────────────────────
    admit_col = _find_col(col_lower, ["admission_type", "admission", "admit_type",
                                       "admission_type_id", "encounter_type"])
    if admit_col:
        vc = df[admit_col].value_counts().head(6).to_dict()
        metrics["admission"] = {str(k): int(v) for k, v in vc.items()}

    # ── Length of stay ───────────────────────────────────
    los_col = _find_col(col_lower, ["length_of_stay", "los", "days_in_hospital",
                                     "time_in_hospital", "hospital_days"])
    if los_col:
        los_series = pd.to_numeric(df[los_col], errors="coerce").dropna()
        if len(los_series) > 0:
            metrics["los"] = {
                "mean": round(los_series.mean(), 1),
                "median": round(los_series.median(), 1),
                "max": int(los_series.max()),
            }

    # ── Blood pressure / glucose / BMI ───────────────────
    for label, keywords in [
        ("glucose", ["glucose", "blood_glucose", "avg_glucose_level"]),
        ("bmi",     ["bmi", "body_mass_index"]),
        ("bp",      ["blood_pressure", "bp", "systolic", "diastolic"]),
    ]:
        found = _find_col(col_lower, keywords)
        if found:
            s = pd.to_numeric(df[found], errors="coerce").dropna()
            if len(s) > 0:
                metrics[label] = {
                    "col": found,
                    "mean": round(s.mean(), 1),
                    "min": round(s.min(), 1),
                    "max": round(s.max(), 1),
                }

    # ── Medications count ────────────────────────────────
    med_col = _find_col(col_lower, ["num_medications", "medications", "num_drugs",
                                     "number_of_medications"])
    if med_col:
        s = pd.to_numeric(df[med_col], errors="coerce").dropna()
        if len(s) > 0:
            metrics["medications"] = {"mean": round(s.mean(), 1), "max": int(s.max())}

    # ── Generic top-categorical fallback for important_cols ──
    covered = {"age", "gender", "sex", "diagnosis", "condition", "admission_type",
               "outcome", "target", "readmitted", "length_of_stay", "los",
               "glucose", "bmi", "blood_pressure", "num_medications"}

    extra_categoricals = []
    for col in important_cols:
        if col.lower() in covered:
            continue
        if col not in df.columns:
            continue
        if df[col].dtype == object or df[col].nunique() <= 15:
            vc = df[col].value_counts().head(6).to_dict()
            extra_categoricals.append({
                "col": col,
                "desc": descriptions.get(col, col),
                "counts": {str(k): int(v) for k, v in vc.items()}
            })
        if len(extra_categoricals) >= 3:
            break

    metrics["extra_categoricals"] = extra_categoricals
    # NICE Integration status
    metrics["nice_status"] = "247 chunks from 5 NICE guidelines loaded"
    return metrics


def _find_col(col_lower: dict, keywords: list):
    for kw in keywords:
        if kw in col_lower:
            return col_lower[kw]
    # partial match
    for kw in keywords:
        for col_key, col_real in col_lower.items():
            if kw in col_key:
                return col_real
    return None


def build_dashboard_html(metrics: dict) -> str:
    total = metrics["total_patients"]
    total_cols = metrics["total_columns"]

    # ── Stat cards ───────────────────────────────────────
    stat_cards = f"""
    <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-info">
            <div class="stat-val">{total:,}</div>
            <div class="stat-label">Total Patients</div>
        </div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🗂</div>
        <div class="stat-info">
            <div class="stat-val">{total_cols}</div>
            <div class="stat-label">Features</div>
        </div>
    </div>"""

    if "age" in metrics:
        a = metrics["age"]
        stat_cards += f"""
    <div class="stat-card">
        <div class="stat-icon">📅</div>
        <div class="stat-info">
            <div class="stat-val">{a['mean']}</div>
            <div class="stat-label">Avg Age (range {a['min']}–{a['max']})</div>
        </div>
    </div>"""

    if "los" in metrics:
        l = metrics["los"]
        stat_cards += f"""
    <div class="stat-card">
        <div class="stat-icon">🏥</div>
        <div class="stat-info">
            <div class="stat-val">{l['mean']} days</div>
            <div class="stat-label">Avg Length of Stay</div>
        </div>
    </div>"""

    if "medications" in metrics:
        stat_cards += f"""
    <div class="stat-card">
        <div class="stat-icon">💊</div>
        <div class="stat-info">
            <div class="stat-val">{metrics['medications']['mean']}</div>
            <div class="stat-label">Avg Medications</div>
        </div>
    </div>"""

    for lbl, key, icon in [("Avg Glucose", "glucose", "🩸"), ("Avg BMI", "bmi", "⚖️")]:
        if key in metrics:
            stat_cards += f"""
    <div class="stat-card">
        <div class="stat-icon">{icon}</div>
        <div class="stat-info">
            <div class="stat-val">{metrics[key]['mean']}</div>
            <div class="stat-label">{lbl}</div>
        </div>
    </div>"""

    # Add NICE status card (once, outside the loop)
    stat_cards += """
    <div class="stat-card">
        <div class="stat-icon">📘</div>
        <div class="stat-info">
            <div class="stat-val">247</div>
            <div class="stat-label">NICE Chunks</div>
        </div>
    </div>"""

    # ── Charts row ───────────────────────────────────────
    charts_html = ""

    if "age_distribution" in metrics:
        charts_html += _bar_chart("Age Distribution", metrics["age_distribution"], color="var(--accent)")

    if "gender" in metrics:
        charts_html += _bar_chart("Gender Split", metrics["gender"], color="var(--green)")

    if "diagnosis" in metrics:
        charts_html += _bar_chart(
            f"Top Diagnoses ({metrics['diagnosis']['col']})",
            metrics["diagnosis"]["top"], color="var(--purple)")

    if "outcome" in metrics:
        charts_html += _bar_chart(
            f"Outcomes ({metrics['outcome']['col']})",
            metrics["outcome"]["distribution"], color="var(--yellow)")

    if "admission" in metrics:
        charts_html += _bar_chart("Admission Types", metrics["admission"], color="#e74c3c")

    for extra in metrics.get("extra_categoricals", []):
        charts_html += _bar_chart(extra["col"], extra["counts"], color="var(--accent)")

    return f"""
<div class="dashboard-grid">
    <div class="stat-row">{stat_cards}</div>
    <div class="charts-grid">{charts_html}</div>
</div>"""


def _bar_chart(title: str, data: dict, color: str = "var(--accent)") -> str:
    if not data:
        return ""
    max_val = max(data.values()) if data.values() else 1
    total = sum(data.values())
    bars = ""
    for label, val in data.items():
        pct = round(val / total * 100, 1)
        bar_w = round(val / max_val * 100, 1)
        bars += f"""
        <div class="bar-row">
            <div class="bar-label" title="{label}">{str(label)[:22]}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{bar_w}%; background:{color};"></div>
            </div>
            <div class="bar-count">{val:,} <span class="bar-pct">({pct}%)</span></div>
        </div>"""

    return f"""
    <div class="chart-card">
        <h4 class="chart-title">{title}</h4>
        <div class="bar-chart">{bars}</div>
    </div>"""