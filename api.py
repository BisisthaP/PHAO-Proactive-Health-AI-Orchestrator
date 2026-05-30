from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Literal, Optional

from feedback import (
    init_db,
    save_recommendation,
    update_status,
    store_feedback,
    get_recommendation,
    get_feedback_for_recommendation,
)
from care_plan import generate_care_plan
from action_logger import (
    init_action_log_table,
    log_action,
    log_care_plan_generated,
    log_book_appointment,
    log_order_test,
    log_medication_review,
    get_actions_for_patient,
    VALID_ACTION_TYPES,
)
from feedback_injection import build_full_feedback_prompt_block


# ─────────────────────────────────────────
# Lifespan: initialise DB on startup
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_action_log_table()
    yield


app = FastAPI(
    title="PHAO — Recommendation Feedback API",
    description="Human-in-the-Loop feedback, care plan generation, and action logging for PHAO.",
    version="2.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────

class SaveRecommendationRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier.")
    recommendation_text: str = Field(..., description="Clinical recommendation text.")

class SaveRecommendationResponse(BaseModel):
    recommendation_id: int
    patient_id: str
    recommendation_text: str
    status: str
    message: str


class ApproveRecommendationRequest(BaseModel):
    recommendation_id: int = Field(..., description="ID of the recommendation to action.")
    action: Literal["approve", "reject", "edit"] = Field(
        ..., description="Action to perform: approve, reject, or edit."
    )
    updated_text: Optional[str] = Field(
        None, description="New recommendation text (required when action='edit')."
    )

class ApproveRecommendationResponse(BaseModel):
    recommendation_id: int
    action: str
    status: str
    updated_text: Optional[str] = None
    message: str


class SubmitFeedbackRequest(BaseModel):
    recommendation_id: int = Field(..., description="ID of the recommendation being reviewed.")
    approved: bool = Field(..., description="True to approve, False to reject.")
    comments: Optional[str] = Field("", description="Optional reviewer comments.")

class SubmitFeedbackResponse(BaseModel):
    feedback_id: int
    recommendation_id: int
    approved: bool
    comments: str
    timestamp: str
    message: str


class CarePlanRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier.")
    patient_name: str = Field(..., description="Patient full name.")
    age: int = Field(..., description="Patient age.")
    conditions: list[str] = Field(..., description="List of conditions e.g. ['hypertension', 'diabetes_type2'].")
    risk_level: Literal["low", "moderate", "high"] = Field(..., description="Risk level from risk agent.")
    risk_factors: list[str] = Field(..., description="List of identified risk factors.")
    current_medications: list[str] = Field(default=[], description="Current medications.")
    duration_days: int = Field(default=14, ge=7, le=30, description="Plan length in days (7-30).")

class CarePlanResponse(BaseModel):
    patient_id: str
    duration_days: int
    care_plan: dict
    actions_logged: list[str]
    message: str


class LogActionRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier.")
    action_type: str = Field(..., description=f"One of: {VALID_ACTION_TYPES}")
    action_data: Optional[dict] = Field(None, description="Optional action details.")
    triggered_by: str = Field(default="clinician", description="Who triggered the action.")

class LogActionResponse(BaseModel):
    action_id: int
    patient_id: str
    action_type: str
    status: str
    message: str


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.post(
    "/save_recommendation",
    response_model=SaveRecommendationResponse,
    status_code=201,
    summary="Save a new recommendation for a patient",
    tags=["HITL"],
)
def save_recommendation_endpoint(body: SaveRecommendationRequest):
    """Save a new recommendation with status `pending`."""
    if not body.patient_id.strip():
        raise HTTPException(status_code=422, detail="patient_id cannot be blank.")
    if not body.recommendation_text.strip():
        raise HTTPException(status_code=422, detail="recommendation_text cannot be blank.")

    rec_id = save_recommendation(
        patient_id=body.patient_id.strip(),
        recommendation_text=body.recommendation_text.strip(),
        status="pending",
    )
    return SaveRecommendationResponse(
        recommendation_id=rec_id,
        patient_id=body.patient_id.strip(),
        recommendation_text=body.recommendation_text.strip(),
        status="pending",
        message="Recommendation saved successfully.",
    )


@app.post(
    "/approve_recommendation",
    response_model=ApproveRecommendationResponse,
    summary="Approve, reject, or edit a recommendation",
    tags=["HITL"],
)
def approve_recommendation(body: ApproveRecommendationRequest):
    """Approve, reject, or edit an existing recommendation."""
    rec = get_recommendation(body.recommendation_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Recommendation {body.recommendation_id} not found.")

    if body.action == "approve":
        update_status(body.recommendation_id, "approved")
        log_action(rec["patient_id"], "recommendation_approved",
                   {"recommendation_id": body.recommendation_id}, triggered_by="clinician")
        return ApproveRecommendationResponse(
            recommendation_id=body.recommendation_id, action="approve",
            status="approved", message="Recommendation approved successfully.",
        )

    elif body.action == "reject":
        update_status(body.recommendation_id, "rejected")
        log_action(rec["patient_id"], "recommendation_rejected",
                   {"recommendation_id": body.recommendation_id}, triggered_by="clinician")
        return ApproveRecommendationResponse(
            recommendation_id=body.recommendation_id, action="reject",
            status="rejected", message="Recommendation rejected successfully.",
        )

    elif body.action == "edit":
        if not body.updated_text or not body.updated_text.strip():
            raise HTTPException(status_code=422, detail="updated_text is required when action is 'edit'.")
        from feedback import get_connection
        with get_connection() as conn:
            conn.execute(
                "UPDATE recommendations SET recommendation_text = ?, status = 'pending' WHERE id = ?",
                (body.updated_text.strip(), body.recommendation_id),
            )
        return ApproveRecommendationResponse(
            recommendation_id=body.recommendation_id, action="edit",
            status="pending", updated_text=body.updated_text.strip(),
            message="Recommendation updated and reset to pending.",
        )


@app.post(
    "/submit_feedback",
    response_model=SubmitFeedbackResponse,
    summary="Submit feedback for a recommendation",
    tags=["HITL"],
)
def submit_feedback(body: SubmitFeedbackRequest):
    """Store clinician feedback and sync recommendation status."""
    rec = get_recommendation(body.recommendation_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Recommendation {body.recommendation_id} not found.")

    feedback_id = store_feedback(
        recommendation_id=body.recommendation_id,
        approved=body.approved,
        comments=body.comments or "",
    )
    all_feedback = get_feedback_for_recommendation(body.recommendation_id)
    stored = next((f for f in all_feedback if f["id"] == feedback_id), None)

    return SubmitFeedbackResponse(
        feedback_id=feedback_id,
        recommendation_id=body.recommendation_id,
        approved=body.approved,
        comments=body.comments or "",
        timestamp=stored["timestamp"] if stored else "",
        message="Feedback submitted successfully.",
    )


@app.get(
    "/care-plan",
    response_model=CarePlanResponse,
    summary="Generate a NICE-aligned care plan for a patient",
    tags=["Care Plan"],
)
def get_care_plan(
    patient_id: str = Query(..., description="Patient identifier e.g. P-1001"),
    patient_name: str = Query(default="Unknown", description="Patient full name"),
    age: int = Query(default=50, ge=0, le=120, description="Patient age"),
    conditions: str = Query(default="hypertension", description="Comma-separated conditions e.g. hypertension,diabetes_type2"),
    risk_level: str = Query(default="moderate", description="low | moderate | high"),
    risk_factors: str = Query(default="", description="Comma-separated risk factors"),
    medications: str = Query(default="", description="Comma-separated current medications"),
    duration_days: int = Query(default=14, ge=7, le=30, description="Plan length 7-30 days"),
):
    """
    Generate a 7–30 day NICE-aligned personalised care plan.

    Uses the risk agent output format + NICE guidelines to produce
    daily micro-habits, weekly actions, monitoring plan, and resources.
    Automatically logs a `care_plan_generated` action.
    """
    # Build risk agent output from query params
    risk_agent_output = {
        "patient_id":          patient_id,
        "patient_name":        patient_name,
        "age":                 age,
        "conditions":          [c.strip() for c in conditions.split(",") if c.strip()],
        "risk_level":          risk_level,
        "risk_factors":        [r.strip() for r in risk_factors.split(",") if r.strip()],
        "current_medications": [m.strip() for m in medications.split(",") if m.strip()],
    }

    try:
        care_plan = generate_care_plan(
            risk_agent_output=risk_agent_output,
            duration_days=duration_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Care plan generation failed: {str(e)}")

    # Log simulated actions based on risk level
    actions_logged = []

    log_care_plan_generated(patient_id, duration_days, risk_level)
    actions_logged.append("care_plan_generated")

    if risk_level == "high":
        log_book_appointment(patient_id, reason="High risk review", urgency="urgent")
        actions_logged.append("book_appointment (urgent)")
        log_order_test(patient_id, test_name="Full blood panel", nice_ref="NG136")
        actions_logged.append("order_test (full blood panel)")
    elif risk_level == "moderate":
        log_book_appointment(patient_id, reason="Routine review", urgency="routine")
        actions_logged.append("book_appointment (routine)")

    if risk_agent_output["current_medications"]:
        log_medication_review(patient_id, medications=risk_agent_output["current_medications"])
        actions_logged.append("medication_review")

    return CarePlanResponse(
        patient_id=patient_id,
        duration_days=duration_days,
        care_plan=care_plan,
        actions_logged=actions_logged,
        message=f"{duration_days}-day NICE-aligned care plan generated successfully.",
    )


@app.post(
    "/care-plan",
    response_model=CarePlanResponse,
    summary="Generate a care plan from full risk agent output (POST)",
    tags=["Care Plan"],
)
def post_care_plan(body: CarePlanRequest):
    """
    POST version — accepts full risk agent JSON body directly.
    Useful when piping output from the LangGraph risk agent.
    """
    risk_agent_output = {
        "patient_id":          body.patient_id,
        "patient_name":        body.patient_name,
        "age":                 body.age,
        "conditions":          body.conditions,
        "risk_level":          body.risk_level,
        "risk_factors":        body.risk_factors,
        "current_medications": body.current_medications,
    }

    try:
        care_plan = generate_care_plan(
            risk_agent_output=risk_agent_output,
            duration_days=body.duration_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Care plan generation failed: {str(e)}")

    actions_logged = []
    log_care_plan_generated(body.patient_id, body.duration_days, body.risk_level)
    actions_logged.append("care_plan_generated")

    if body.risk_level == "high":
        log_book_appointment(body.patient_id, reason="High risk review", urgency="urgent")
        actions_logged.append("book_appointment (urgent)")
        log_order_test(body.patient_id, test_name="Full blood panel", nice_ref="NG136")
        actions_logged.append("order_test (full blood panel)")
    elif body.risk_level == "moderate":
        log_book_appointment(body.patient_id, reason="Routine review", urgency="routine")
        actions_logged.append("book_appointment (routine)")

    if body.current_medications:
        log_medication_review(body.patient_id, medications=body.current_medications)
        actions_logged.append("medication_review")

    return CarePlanResponse(
        patient_id=body.patient_id,
        duration_days=body.duration_days,
        care_plan=care_plan,
        actions_logged=actions_logged,
        message=f"{body.duration_days}-day NICE-aligned care plan generated successfully.",
    )


@app.post(
    "/log-action",
    response_model=LogActionResponse,
    status_code=201,
    summary="Log a simulated clinical action",
    tags=["Action Log"],
)
def log_clinical_action(body: LogActionRequest):
    """Manually log a simulated clinical action for a patient."""
    if body.action_type not in VALID_ACTION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type. Must be one of: {VALID_ACTION_TYPES}",
        )
    action_id = log_action(
        patient_id=body.patient_id,
        action_type=body.action_type,
        action_data=body.action_data,
        triggered_by=body.triggered_by,
    )
    return LogActionResponse(
        action_id=action_id,
        patient_id=body.patient_id,
        action_type=body.action_type,
        status="simulated",
        message="Action logged successfully.",
    )


@app.get(
    "/actions/{patient_id}",
    summary="Get all logged actions for a patient",
    tags=["Action Log"],
)
def get_patient_actions(patient_id: str):
    """Return the full action log for a given patient."""
    actions = get_actions_for_patient(patient_id)
    return {"patient_id": patient_id, "total": len(actions), "actions": actions}


@app.get(
    "/feedback-context",
    summary="Get feedback injection block for RAG prompts",
    tags=["Feedback Injection"],
)
def get_feedback_injection(patient_id: Optional[str] = Query(None, description="Filter by patient ID")):
    """
    Returns clinician-approved and rejected feedback formatted
    as a prompt block ready to inject into RAG system prompts.
    """
    block = build_full_feedback_prompt_block(patient_id=patient_id)
    return {
        "patient_id": patient_id,
        "feedback_prompt_block": block if block else "No feedback available yet.",
        "message": "Inject `feedback_prompt_block` into your RAG system prompt.",
    }