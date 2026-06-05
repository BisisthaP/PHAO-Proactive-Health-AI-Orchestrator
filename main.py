from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import shutil
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


# ── Pydantic response models ─────────────────────────────────────────────

class PreprocessResponse(BaseModel):
    """Structured response for the /preprocess endpoint."""
    success: bool
    message: str
    filename: str
    original_shape: List[int]
    cleaned_shape: List[int]
    patient_id_col: Optional[str]
    columns_used: List[str]
    dropped_columns: List[str]
    vital_columns: List[str]
    date_columns: List[str]
    num_patient_chunks: int
    num_nice_chunks: int
    total_chunks: int
    guidelines_loaded: List[str]
    sample_patient_ids: List[str]

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

session_store = {}


# ── Routes ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return HTMLResponse(content=_error_card("Only CSV files are supported."), status_code=400)

    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return HTMLResponse(content=_error_card(f"Could not parse CSV: {str(e)}"), status_code=400)

    rows, cols = df.shape
    session_store["current_file"] = file_path
    session_store["filename"] = file.filename
    session_store["shape"] = (rows, cols)
    session_store["columns"] = list(df.columns)
    session_store["preprocessed"] = False

    null_pct = (df.isnull().sum() / len(df) * 100).round(1).to_dict()
    flagged = [col for col, pct in null_pct.items() if pct >= 50]

    col_preview = []
    for col in df.columns:
        col_preview.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_pct": null_pct[col],
            "flagged": col in flagged
        })

    return HTMLResponse(content=_upload_success_card(
        filename=file.filename,
        rows=rows,
        cols=cols,
        col_preview=col_preview,
        flagged_count=len(flagged)
    ))


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(file: UploadFile = File(...)):
    """
    Full preprocessing + hybrid embedding pipeline.

    Accepts an uploaded patient CSV and:
      1. Cleans it (drop sparse columns, detect IDs/vitals/dates, fill gaps).
      2. Loads + chunks the NICE guidelines from ``guidelines/``.
      3. Builds a hybrid ChromaDB vector store (patient rows + NICE chunks).

    Returns a structured JSON summary of the run.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Save the uploaded file locally.
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    try:
        from preprocessing import preprocess_patient_data
        from guidelines_loader import load_nice_guidelines
        from embeddings import build_hybrid_vectorstore

        # 1. Clean + analyse the patient data.
        cleaned_df, meta = preprocess_patient_data(df)

        # Persist the cleaned CSV for downstream modules (risk, dashboard).
        cleaned_path = file_path.replace(".csv", "_cleaned.csv")
        cleaned_df.to_csv(cleaned_path, index=False)

        # 2. Load NICE guideline chunks.
        nice_documents = load_nice_guidelines("guidelines")
        guidelines_loaded = sorted({
            d.metadata.get("guideline_name", "unknown") for d in nice_documents
        })

        # 3. Build the hybrid vector store.
        stats = build_hybrid_vectorstore(
            patient_df=cleaned_df,
            nice_documents=nice_documents,
            patient_id_col=meta["patient_id_col"],
        )

        # Populate session state so chat / dashboard / risk endpoints work.
        session_store.update({
            "current_file": file_path,
            "filename": file.filename,
            "cleaned_file": cleaned_path,
            "preprocessed": True,
            "patient_id_col": meta["patient_id_col"],
            "patient_ids": meta["patient_ids"],
            "important_cols": meta["vital_cols"] + (
                [meta["patient_id_col"]] if meta["patient_id_col"] else []
            ),
            "descriptions": {},
            "dropped_cols": meta["dropped_cols"],
            "kept_cols": meta["kept_cols"],
            "cleaned_shape": meta["cleaned_shape"],
        })

        return PreprocessResponse(
            success=True,
            message="Preprocessing and hybrid embedding completed successfully.",
            filename=file.filename,
            original_shape=list(meta["original_shape"]),
            cleaned_shape=list(meta["cleaned_shape"]),
            patient_id_col=meta["patient_id_col"],
            columns_used=meta["kept_cols"],
            dropped_columns=meta["dropped_cols"],
            vital_columns=meta["vital_cols"],
            date_columns=meta["date_cols"],
            num_patient_chunks=stats["patient_chunks"],
            num_nice_chunks=stats["nice_chunks"],
            total_chunks=stats["total_chunks"],
            guidelines_loaded=guidelines_loaded,
            sample_patient_ids=meta["sample_patient_ids"],
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed: {e}\n{traceback.format_exc()}",
        )


@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    if "current_file" not in session_store:
        return HTMLResponse(content='<span style="color:#7d8590;font-size:11px;">No file loaded</span>')
    preprocessed = session_store.get("preprocessed", False)
    dot = "🟢" if preprocessed else "🟡"
    fname = session_store.get("filename", "")
    label = "Ready" if preprocessed else "Awaiting preprocessing"
    return HTMLResponse(content=f'<span style="font-size:11px;">{dot} {fname}<br><span style="color:#7d8590;">{label}</span></span>')


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not session_store.get("preprocessed"):
        return HTMLResponse(content='<div class="placeholder-card">Upload and preprocess a file first.</div>')
    try:
        from dashboard import compute_metrics, build_dashboard_html
        metrics = compute_metrics(
            cleaned_csv_path=session_store["cleaned_file"],
            important_cols=session_store.get("important_cols", []),
            descriptions=session_store.get("descriptions", {})
        )
        return HTMLResponse(content=build_dashboard_html(metrics))
    except Exception as e:
        import traceback
        return HTMLResponse(content=f'<div class="placeholder-card" style="color:var(--red);">Dashboard error: {e}<br><pre>{traceback.format_exc()}</pre></div>')


@app.post("/chat")
async def chat(request: Request):
    if not session_store.get("preprocessed"):
        return JSONResponse({"answer": "Please upload and preprocess a file first.", "html": '<div class="rag-answer">Please upload and preprocess a file first.</div>'})
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        if not question:
            return JSONResponse({"answer": "Please enter a question.", "html": ""})

        from rag import rag_query, format_answer_html
        result = rag_query(question)
        html = format_answer_html(result)
        return JSONResponse({"answer": result["answer"], "html": html})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse({"answer": f"Error: {str(e)}", "html": f'<div class="rag-answer" style="color:var(--red);">{str(e)}<br><pre>{tb}</pre></div>'})


@app.get("/patients", response_class=HTMLResponse)
async def patients(request: Request):
    if not session_store.get("preprocessed"):
        return HTMLResponse(content='<option>No data loaded</option>')
    ids = session_store.get("patient_ids", [])
    options = "".join(f"<option value='{pid}'>{pid}</option>" for pid in ids[:200])
    return HTMLResponse(content=options)


@app.get("/risk", response_class=HTMLResponse)
async def risk(request: Request, patient_id: str = ""):
    if not session_store.get("preprocessed"):
        return HTMLResponse(content='<div class="placeholder-card">Upload and preprocess a file first.</div>')
    if not patient_id:
        return HTMLResponse(content='<div class="placeholder-card">Select a patient to assess.</div>')
    try:
        from risk import assess_risk, build_risk_html
        result = assess_risk(
            patient_id=patient_id,
            cleaned_csv_path=session_store["cleaned_file"],
            patient_id_col=session_store.get("patient_id_col", "")
        )
        return HTMLResponse(content=build_risk_html(result))
    except Exception as e:
        import traceback
        return HTMLResponse(content=f'<div class="placeholder-card" style="color:var(--red);">Error: {e}<br><pre>{traceback.format_exc()}</pre></div>')


# ── HTML Builders ────────────────────────────────────────

def _upload_success_card(filename, rows, cols, col_preview, flagged_count):
    rows_html = ""
    for c in col_preview:
        flag_badge = '<span class="badge badge-warn">⚠ will drop</span>' if c["flagged"] else '<span class="badge badge-ok">✓ keep</span>'
        null_bar_width = min(c["null_pct"], 100)
        null_color = "#e74c3c" if c["null_pct"] >= 50 else "#2ecc71" if c["null_pct"] < 10 else "#f39c12"
        rows_html += f"""
        <tr>
            <td>{c['name']}</td>
            <td><span class="dtype">{c['dtype']}</span></td>
            <td>
                <div class="null-bar-wrap">
                    <div class="null-bar" style="width:{null_bar_width}%; background:{null_color};"></div>
                    <span>{c['null_pct']}%</span>
                </div>
            </td>
            <td>{flag_badge}</td>
        </tr>"""

    return f"""
    <div class="card success-card">
        <div class="card-header">
            <span class="status-dot green"></span>
            <h3>File Uploaded Successfully</h3>
        </div>
        <div class="file-meta">
            <div class="meta-item"><span class="meta-label">File</span><span class="meta-val">{filename}</span></div>
            <div class="meta-item"><span class="meta-label">Rows</span><span class="meta-val">{rows:,}</span></div>
            <div class="meta-item"><span class="meta-label">Columns</span><span class="meta-val">{cols}</span></div>
            <div class="meta-item"><span class="meta-label">Flagged for drop</span><span class="meta-val warn">{flagged_count}</span></div>
        </div>
        <h4 class="section-title">Column Preview</h4>
        <div class="table-wrap">
            <table class="col-table">
                <thead><tr><th>Column</th><th>Type</th><th>Null %</th><th>Status</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        <div class="next-step">
            <p>✅ File stored. Click to run AI preprocessing and build embeddings.</p>
            <button class="btn btn-primary" hx-post="/preprocess" hx-target="#main-content" hx-swap="innerHTML" hx-indicator="#spinner">
                Run AI Preprocessing →
            </button>
        </div>
    </div>"""


def _preprocess_success_card(original_shape, cleaned_shape, dropped_cols, kept_cols,
                               patient_id_col, important_cols, descriptions, total_embedded):
    orig_r, orig_c = original_shape
    clean_r, clean_c = cleaned_shape

    dropped_html = "".join(f'<span class="badge badge-warn">{c}</span> ' for c in dropped_cols) or "<em>None</em>"
    important_html = "".join(f'<span class="badge badge-ok">{c}</span> ' for c in important_cols)

    desc_rows = ""
    for col, desc in descriptions.items():
        desc_rows += f"<tr><td><strong>{col}</strong></td><td>{desc}</td></tr>"

    pid_display = patient_id_col if patient_id_col else "<em>Not detected — using row index</em>"

    return f"""
    <div class="card success-card">
        <div class="card-header">
            <span class="status-dot green"></span>
            <h3>Preprocessing Complete — Embeddings Stored</h3>
        </div>

        <div class="file-meta">
            <div class="meta-item"><span class="meta-label">Original</span><span class="meta-val">{orig_r:,} rows × {orig_c} cols</span></div>
            <div class="meta-item"><span class="meta-label">After Cleaning</span><span class="meta-val">{clean_r:,} rows × {clean_c} cols</span></div>
            <div class="meta-item"><span class="meta-label">Vectors Stored</span><span class="meta-val" style="color:var(--accent)">{total_embedded:,}</span></div>
            <div class="meta-item"><span class="meta-label">Patient ID Col</span><span class="meta-val">{pid_display}</span></div>
        </div>

        <h4 class="section-title" style="margin-top:16px;">Dropped Columns (≥50% null)</h4>
        <div style="margin-bottom:16px;">{dropped_html}</div>

        <h4 class="section-title">Key Medical Columns (Gemini identified)</h4>
        <div style="margin-bottom:16px;">{important_html}</div>

        <h4 class="section-title">Column Descriptions (Gemini)</h4>
        <div class="table-wrap" style="margin-bottom:16px;">
            <table class="col-table">
                <thead><tr><th>Column</th><th>Clinical Meaning</th></tr></thead>
                <tbody>{desc_rows}</tbody>
            </table>
        </div>

        <div class="next-step">
            <p>✅ ChromaDB ready. Navigate to <strong>Dashboard</strong>, <strong>Chat</strong>, or <strong>Risk Detection</strong>.</p>
            <span class="badge badge-ok">Stage 2 Complete</span>
        </div>
    </div>"""


def _error_card(message):
    return f"""
    <div class="card error-card">
        <div class="card-header">
            <span class="status-dot red"></span>
            <h3>Error</h3>
        </div>
        <p class="error-msg">{message}</p>
    </div>"""
