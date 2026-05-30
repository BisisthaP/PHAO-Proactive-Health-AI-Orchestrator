"""
action_logger.py
────────────────
Simulated action logging for PHAO Section 4.
Logs every clinical action (book appointment, order test, medication review, etc.)
to SQLite with full audit trail.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional

DB_PATH = "feedback.db"


# ─────────────────────────────────────────
# Table setup (called from init_db)
# ─────────────────────────────────────────

def init_action_log_table():
    """Create action_log table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS action_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   TEXT    NOT NULL,
                action_type  TEXT    NOT NULL,
                action_data  TEXT,
                triggered_by TEXT    NOT NULL DEFAULT 'system',
                status       TEXT    NOT NULL DEFAULT 'simulated',
                timestamp    TEXT    NOT NULL
            );
        """)


# ─────────────────────────────────────────
# Action types
# ─────────────────────────────────────────

VALID_ACTION_TYPES = [
    "book_appointment",
    "order_test",
    "medication_review",
    "referral",
    "send_reminder",
    "escalate_to_gp",
    "care_plan_generated",
    "recommendation_approved",
    "recommendation_rejected",
]


# ─────────────────────────────────────────
# Core logging function
# ─────────────────────────────────────────

def log_action(
    patient_id: str,
    action_type: str,
    action_data: Optional[dict] = None,
    triggered_by: str = "system",
    status: str = "simulated",
) -> int:
    """
    Log a clinical action for a patient.

    Args:
        patient_id:   The patient this action is for.
        action_type:  One of VALID_ACTION_TYPES.
        action_data:  Optional dict with action details (e.g. test name, date).
        triggered_by: Who triggered it — 'system', 'clinician', 'care_plan'.
        status:       'simulated' | 'pending' | 'completed'.

    Returns:
        The id of the logged action.
    """
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(
            f"Invalid action_type '{action_type}'. "
            f"Must be one of: {VALID_ACTION_TYPES}"
        )

    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO action_log
                (patient_id, action_type, action_data, triggered_by, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                action_type,
                json.dumps(action_data) if action_data else None,
                triggered_by,
                status,
                timestamp,
            ),
        )
        return cursor.lastrowid


# ─────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────

def log_book_appointment(patient_id: str, reason: str, urgency: str = "routine") -> int:
    return log_action(
        patient_id=patient_id,
        action_type="book_appointment",
        action_data={"reason": reason, "urgency": urgency},
        triggered_by="care_plan",
    )


def log_order_test(patient_id: str, test_name: str, nice_ref: str = "") -> int:
    return log_action(
        patient_id=patient_id,
        action_type="order_test",
        action_data={"test_name": test_name, "nice_ref": nice_ref},
        triggered_by="care_plan",
    )


def log_medication_review(patient_id: str, medications: list, reason: str = "") -> int:
    return log_action(
        patient_id=patient_id,
        action_type="medication_review",
        action_data={"medications": medications, "reason": reason},
        triggered_by="care_plan",
    )


def log_care_plan_generated(patient_id: str, duration_days: int, risk_level: str) -> int:
    return log_action(
        patient_id=patient_id,
        action_type="care_plan_generated",
        action_data={"duration_days": duration_days, "risk_level": risk_level},
        triggered_by="system",
    )


# ─────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────

def get_actions_for_patient(patient_id: str) -> list[dict]:
    """Return all logged actions for a patient, newest first."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM action_log WHERE patient_id = ? ORDER BY timestamp DESC",
            (patient_id,),
        ).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        if r["action_data"]:
            r["action_data"] = json.loads(r["action_data"])
        results.append(r)
    return results


def get_all_actions(limit: int = 100) -> list[dict]:
    """Return the most recent actions across all patients."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM action_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        if r["action_data"]:
            r["action_data"] = json.loads(r["action_data"])
        results.append(r)
    return results