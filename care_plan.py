"""
care_plan.py
────────────
Generates a 7–30 day NICE-aligned personalised care plan from:
  • risk_agent_output  – dict produced by your risk assessment agent
  • nice_guidelines    – list of relevant NICE guideline references

Returns structured JSON with daily micro-habits and weekly actions.
"""

import json
import re
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request
import urllib.error
from datetime import date, timedelta
from typing import Any


# ─────────────────────────────────────────────────────────────
# NICE guideline reference bank (extend as needed)
# ─────────────────────────────────────────────────────────────

NICE_GUIDELINES: dict[str, dict] = {
    "hypertension": {
        "ref": "NICE NG136",
        "title": "Hypertension in adults: diagnosis and management",
        "key_actions": [
            "Monitor blood pressure daily at consistent time",
            "Reduce sodium intake to <6 g/day",
            "Engage in 150 min moderate aerobic activity per week",
            "Limit alcohol to ≤14 units/week",
            "Maintain BMI 18.5–24.9",
        ],
    },
    "diabetes_type2": {
        "ref": "NICE NG28",
        "title": "Type 2 diabetes in adults: management",
        "key_actions": [
            "Self-monitor blood glucose as directed",
            "Follow low glycaemic index diet",
            "Complete 10-min post-meal walks",
            "Take medication at consistent daily times",
            "Attend structured diabetes education programme",
        ],
    },
    "depression": {
        "ref": "NICE CG90",
        "title": "Depression in adults: recognition and management",
        "key_actions": [
            "Engage in 30 min of physical activity 3×/week",
            "Maintain regular sleep schedule (same wake time)",
            "Practice behavioural activation – schedule one enjoyable activity daily",
            "Limit alcohol and caffeine",
            "Use validated mood-tracking tool weekly (e.g. PHQ-9)",
        ],
    },
    "obesity": {
        "ref": "NICE CG189",
        "title": "Obesity: identification, assessment and management",
        "key_actions": [
            "Set a 500–600 kcal/day deficit via diet",
            "Record daily food intake in diary",
            "Aim for 10,000 steps per day",
            "Attend structured weight-management programme",
            "Weigh weekly at the same time and day",
        ],
    },
    "copd": {
        "ref": "NICE NG115",
        "title": "Chronic obstructive pulmonary disease in over 16s",
        "key_actions": [
            "Use inhaler as prescribed — check technique monthly",
            "Complete pulmonary rehabilitation exercises",
            "Avoid smoking and second-hand smoke",
            "Perform daily spirometry if device available",
            "Report increased breathlessness or sputum promptly",
        ],
    },
    "anxiety": {
        "ref": "NICE CG113",
        "title": "Generalised anxiety disorder and panic disorder in adults",
        "key_actions": [
            "Practice diaphragmatic breathing 2× daily (5 min each)",
            "Use worry-postponement journalling technique",
            "Limit caffeine and stimulants",
            "Engage in low-intensity exercise daily",
            "Complete a validated anxiety measure weekly (e.g. GAD-7)",
        ],
    },
    "falls_prevention": {
        "ref": "NICE CG161",
        "title": "Falls in older people: assessing risk and prevention",
        "key_actions": [
            "Complete balance and strength exercises daily (Otago programme)",
            "Review footwear — wear non-slip, well-fitting shoes",
            "Remove home hazards (loose rugs, poor lighting)",
            "Review medications with GP if ≥4 regular medicines",
            "Have annual vision check",
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# Claude API call
# ─────────────────────────────────────────────────────────────

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-20250514"


def _call_claude(prompt: str, max_tokens: int = 4096) -> str:
    """Send a prompt to Claude and return the text response."""
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "system": (
            "You are a clinical decision-support assistant that generates "
            "structured, NICE-aligned personalised care plans. "
            "Always respond with valid JSON only — no markdown fences, "
            "no preamble, no trailing commentary."
        ),
    }).encode()

    req = urllib.request.Request(
    API_URL,
    data=payload,
    headers={
        "Content-Type": "application/json",
        "x-api-key": "YOUR_ANTHROPIC_API_KEY",
        "anthropic-version": "2023-06-01",
    },
    method="POST",
)

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    # Extract text from response
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    raise ValueError("No text block found in Claude response.")


# ─────────────────────────────────────────────────────────────
# Care plan builder
# ─────────────────────────────────────────────────────────────

def generate_care_plan(
    risk_agent_output: dict[str, Any],
    duration_days: int = 14,
    guideline_keys: list[str] | None = None,
) -> dict:
    """
    Generate a NICE-aligned care plan.

    Args:
        risk_agent_output:  Output from the risk assessment agent. Expected keys:
                              patient_id, patient_name, age, conditions (list),
                              risk_level ('low'|'moderate'|'high'),
                              risk_factors (list), current_medications (list).
        duration_days:      Plan length in days (7–30). Defaults to 14.
        guideline_keys:     Optional list of keys from NICE_GUIDELINES to include.
                            If omitted, conditions from risk_agent_output are matched.

    Returns:
        Structured care plan as a Python dict (mirrors the JSON response).
    """
    duration_days = max(7, min(30, duration_days))

    # ── Resolve guidelines ──────────────────────────────────
    if guideline_keys:
        guidelines = {k: NICE_GUIDELINES[k] for k in guideline_keys if k in NICE_GUIDELINES}
    else:
        conditions = [c.lower().replace(" ", "_") for c in risk_agent_output.get("conditions", [])]
        guidelines = {k: v for k, v in NICE_GUIDELINES.items() if k in conditions}

    if not guidelines:
        # Fall back to all guidelines if no match
        guidelines = NICE_GUIDELINES

    start_date = date.today()
    end_date   = start_date + timedelta(days=duration_days - 1)

    # ── Build prompt ────────────────────────────────────────
    prompt = f"""
Generate a personalised {duration_days}-day NICE-aligned care plan.

PATIENT RISK PROFILE:
{json.dumps(risk_agent_output, indent=2)}

APPLICABLE NICE GUIDELINES:
{json.dumps(guidelines, indent=2)}

PLAN DATES: {start_date.isoformat()} to {end_date.isoformat()}

Return a single JSON object with this exact structure:

{{
  "care_plan": {{
    "metadata": {{
      "patient_id":      "<from risk profile>",
      "patient_name":    "<from risk profile>",
      "generated_date":  "{start_date.isoformat()}",
      "plan_start":      "{start_date.isoformat()}",
      "plan_end":        "{end_date.isoformat()}",
      "duration_days":   {duration_days},
      "risk_level":      "<from risk profile>",
      "nice_references": ["<list of NICE ref strings used>"]
    }},
    "goals": [
      {{
        "goal_id":     "G1",
        "description": "<SMART goal>",
        "target_date": "<ISO date>",
        "linked_guideline": "<NICE ref>"
      }}
    ],
    "daily_micro_habits": [
      {{
        "habit_id":    "H1",
        "title":       "<short title>",
        "description": "<what the patient should do>",
        "frequency":   "daily",
        "duration":    "<e.g. 5 min>",
        "time_of_day": "<morning|afternoon|evening|any>",
        "rationale":   "<why this helps, linked to NICE>",
        "linked_guideline": "<NICE ref>"
      }}
    ],
    "weekly_actions": [
      {{
        "week":        1,
        "action_id":   "W1",
        "title":       "<short title>",
        "description": "<what to do this week>",
        "linked_guideline": "<NICE ref>",
        "success_metric": "<how to know it was completed>"
      }}
    ],
    "monitoring": {{
      "self_monitoring": ["<list of daily self-checks>"],
      "clinical_review":  "<when to contact GP or clinician>",
      "red_flags":        ["<symptoms requiring immediate escalation>"]
    }},
    "resources": [
      {{
        "title": "<resource name>",
        "type":  "<app|leaflet|website|programme>",
        "url":   "<URL if applicable or null>"
      }}
    ]
  }}
}}

Rules:
- daily_micro_habits: produce 5–8 habits tailored to the patient's risk factors.
- weekly_actions: one meaningful action per week for {duration_days // 7} week(s).
- All habits and actions must be evidence-based and trace to a NICE guideline.
- Use plain, patient-friendly language (max reading age 12).
- Return valid JSON only.
"""

    raw = _call_claude(prompt)

    # Strip accidental markdown fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    care_plan = json.loads(raw)
    return care_plan


# ─────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────

def save_care_plan(care_plan: dict, path: str = "care_plan_output.json") -> str:
    """Persist the care plan JSON to disk and return the path."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(care_plan, fh, indent=2, ensure_ascii=False)
    return path


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_risk_output = {
        "patient_id":           "P-1001",
        "patient_name":         "Jane Smith",
        "age":                  67,
        "conditions":           ["hypertension", "diabetes_type2"],
        "risk_level":           "high",
        "risk_factors": [
            "BMI 31.4",
            "sedentary lifestyle",
            "smoker (10/day)",
            "HbA1c 68 mmol/mol",
            "systolic BP 158 mmHg",
        ],
        "current_medications":  ["metformin 500 mg BD", "amlodipine 5 mg OD"],
    }

    print("Generating care plan … (calling Claude API)")
    plan = generate_care_plan(
        risk_agent_output=sample_risk_output,
        duration_days=14,
    )

    out_path = save_care_plan(plan)
    print(f"\nCare plan saved → {out_path}")
    print(json.dumps(plan, indent=2))