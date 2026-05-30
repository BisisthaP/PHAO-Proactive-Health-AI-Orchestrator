"""
feedback_injection.py
─────────────────────
Pulls approved clinician feedback from SQLite and injects it into
future RAG prompts so the system learns from past decisions.

Used by Person B's RAG pipeline — import get_feedback_context() and
include its output in the system prompt.
"""

import sqlite3
from feedback import get_connection


# ─────────────────────────────────────────
# Core injection function
# ─────────────────────────────────────────

def get_feedback_context(patient_id: str | None = None, limit: int = 5) -> str:
    """
    Retrieve approved feedback and format it as a prompt injection block.

    Args:
        patient_id:  If provided, only fetch feedback for this patient.
                     If None, fetch the most recent approved feedback globally.
        limit:       Max number of feedback entries to include (default 5).

    Returns:
        A formatted string ready to inject into a RAG system prompt.
        Returns empty string if no approved feedback exists.
    """
    rows = _fetch_approved_feedback(patient_id=patient_id, limit=limit)

    if not rows:
        return ""

    lines = [
        "── Clinician-Approved Feedback (use this to improve your response) ──"
    ]

    for row in rows:
        lines.append(
            f"• Patient {row['patient_id']} | Recommendation: \"{row['recommendation_text']}\" "
            f"| Approved ✓ | Clinician note: \"{row['comments'] or 'No comment'}\""
        )

    lines.append("────────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def get_rejected_feedback_context(patient_id: str | None = None, limit: int = 5) -> str:
    """
    Retrieve rejected feedback so the RAG system avoids similar recommendations.

    Returns:
        A formatted string listing rejected recommendations to avoid.
    """
    rows = _fetch_rejected_feedback(patient_id=patient_id, limit=limit)

    if not rows:
        return ""

    lines = [
        "── Clinician-Rejected Feedback (DO NOT repeat these recommendations) ──"
    ]

    for row in rows:
        lines.append(
            f"• Patient {row['patient_id']} | REJECTED: \"{row['recommendation_text']}\" "
            f"| Reason: \"{row['comments'] or 'No reason given'}\""
        )

    lines.append("────────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def build_full_feedback_prompt_block(patient_id: str | None = None, limit: int = 5) -> str:
    """
    Build a complete feedback block combining approved + rejected feedback.
    Inject this into your RAG system prompt before generating recommendations.

    Example usage in rag.py:
        from feedback_injection import build_full_feedback_prompt_block
        feedback_block = build_full_feedback_prompt_block(patient_id="P-1001")
        system_prompt = f\"\"\"{base_system_prompt}

{feedback_block}
\"\"\"
    """
    approved = get_feedback_context(patient_id=patient_id, limit=limit)
    rejected = get_rejected_feedback_context(patient_id=patient_id, limit=limit)

    parts = []
    if approved:
        parts.append(approved)
    if rejected:
        parts.append(rejected)

    if not parts:
        return ""

    return "\n\n".join(parts)


# ─────────────────────────────────────────
# DB queries
# ─────────────────────────────────────────

def _fetch_approved_feedback(patient_id: str | None, limit: int) -> list[dict]:
    return _fetch_feedback(approved=1, patient_id=patient_id, limit=limit)


def _fetch_rejected_feedback(patient_id: str | None, limit: int) -> list[dict]:
    return _fetch_feedback(approved=0, patient_id=patient_id, limit=limit)


def _fetch_feedback(approved: int, patient_id: str | None, limit: int) -> list[dict]:
    with get_connection() as conn:
        if patient_id:
            rows = conn.execute(
                """
                SELECT r.patient_id, r.recommendation_text, f.comments, f.timestamp
                FROM feedback f
                JOIN recommendations r ON f.recommendation_id = r.id
                WHERE f.approved = ? AND r.patient_id = ?
                ORDER BY f.timestamp DESC
                LIMIT ?
                """,
                (approved, patient_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT r.patient_id, r.recommendation_text, f.comments, f.timestamp
                FROM feedback f
                JOIN recommendations r ON f.recommendation_id = r.id
                WHERE f.approved = ?
                ORDER BY f.timestamp DESC
                LIMIT ?
                """,
                (approved, limit),
            ).fetchall()
    return [dict(row) for row in rows]


# ─────────────────────────────────────────
# Demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=== Full Feedback Prompt Block ===\n")
    block = build_full_feedback_prompt_block()
    print(block if block else "No feedback in database yet.")