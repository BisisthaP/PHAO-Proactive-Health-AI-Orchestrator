import sqlite3
from datetime import datetime


DB_PATH = "feedback.db"


# ─────────────────────────────────────────
# Database setup
# ─────────────────────────────────────────

def get_connection():
    """Return a connection with row_factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not already exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id           TEXT    NOT NULL,
                recommendation_text  TEXT    NOT NULL,
                status               TEXT    NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id   INTEGER NOT NULL,
                approved            INTEGER NOT NULL,          -- 1 = approved, 0 = rejected
                comments            TEXT,
                timestamp           TEXT    NOT NULL,
                FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
            );
        """)


# ─────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────

def save_recommendation(patient_id: str, recommendation_text: str, status: str = "pending") -> int:
    """
    Insert a new recommendation and return its generated id.

    Args:
        patient_id:           Identifier for the patient.
        recommendation_text:  Clinical or care recommendation.
        status:               Initial status (default: 'pending').

    Returns:
        The integer id of the newly inserted row.
    """
    if not patient_id or not recommendation_text:
        raise ValueError("patient_id and recommendation_text are required.")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO recommendations (patient_id, recommendation_text, status)
            VALUES (?, ?, ?)
            """,
            (patient_id, recommendation_text, status),
        )
        return cursor.lastrowid


def update_status(recommendation_id: int, new_status: str) -> bool:
    """
    Update the status field of an existing recommendation.

    Args:
        recommendation_id:  The id of the recommendation to update.
        new_status:         New status value (e.g. 'approved', 'rejected', 'pending').

    Returns:
        True if a row was updated, False if the id was not found.
    """
    if not new_status:
        raise ValueError("new_status cannot be empty.")

    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE recommendations SET status = ? WHERE id = ?",
            (new_status, recommendation_id),
        )
        return cursor.rowcount > 0


def store_feedback(recommendation_id: int, approved: bool, comments: str = "") -> int:
    """
    Record feedback for a recommendation and return the feedback id.

    Also updates the parent recommendation status to 'approved' or 'rejected'
    to keep both tables consistent.

    Args:
        recommendation_id:  The id of the recommendation being reviewed.
        approved:           True if the recommendation is approved, False otherwise.
        comments:           Optional free-text comments from the reviewer.

    Returns:
        The integer id of the newly inserted feedback row.
    """
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with get_connection() as conn:
        # Verify the recommendation exists
        row = conn.execute(
            "SELECT id FROM recommendations WHERE id = ?", (recommendation_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No recommendation found with id={recommendation_id}.")

        # Insert feedback
        cursor = conn.execute(
            """
            INSERT INTO feedback (recommendation_id, approved, comments, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (recommendation_id, int(approved), comments, timestamp),
        )
        feedback_id = cursor.lastrowid

        # Keep recommendation status in sync
        new_status = "approved" if approved else "rejected"
        conn.execute(
            "UPDATE recommendations SET status = ? WHERE id = ?",
            (new_status, recommendation_id),
        )

        return feedback_id


# ─────────────────────────────────────────
# Helper / query functions
# ─────────────────────────────────────────

def get_recommendation(recommendation_id: int) -> dict | None:
    """Return a recommendation row as a dict, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)
        ).fetchone()
        return dict(row) if row else None


def get_feedback_for_recommendation(recommendation_id: int) -> list[dict]:
    """Return all feedback rows for a given recommendation."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE recommendation_id = ? ORDER BY timestamp",
            (recommendation_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    # 1. Save a recommendation
    rec_id = save_recommendation(
        patient_id="P-1001",
        recommendation_text="Increase daily water intake to at least 2 litres.",
    )
    print(f"Saved recommendation  → id={rec_id}")

    # 2. Update its status manually
    updated = update_status(rec_id, "under_review")
    print(f"Status updated        → success={updated}")

    # 3. Store feedback (approval)
    fb_id = store_feedback(
        recommendation_id=rec_id,
        approved=True,
        comments="Clinician confirmed this is appropriate for the patient.",
    )
    print(f"Stored feedback       → id={fb_id}")

    # 4. Verify
    print("\nRecommendation record:")
    print(get_recommendation(rec_id))

    print("\nFeedback records:")
    for fb in get_feedback_for_recommendation(rec_id):
        print(fb)