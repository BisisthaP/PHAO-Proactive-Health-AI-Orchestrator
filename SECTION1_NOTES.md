# Section 1 — Preprocessing & NICE Guidelines Integration

Owner: Person A · Branch: `feature/preprocessing-nice`

This section turns a raw patient CSV into a cleaned DataFrame and a **hybrid
ChromaDB vector store** containing both patient rows and chunked NHS NICE
guidelines, ready for the RAG / risk modules.

## Files

| File | Purpose |
|------|---------|
| `preprocessing.py` | `preprocess_patient_data(df) -> (clean_df, metadata)` — drops sparse columns, auto-detects patient ID / vital / date columns, fills gaps. Pure pandas, no LLM. |
| `guidelines_loader.py` | `load_nice_guidelines(dir="guidelines") -> list[Document]` — reads `.md`/`.txt`/`.pdf`, chunks to ~400–600 tokens, adds filtering metadata. Cached. |
| `embeddings.py` | `build_hybrid_vectorstore(patient_df, nice_documents, ...)` + `query_hybrid(query, doc_type=...)` — single Chroma collection, `doc_type` = `patient` \| `nice_guideline`. |
| `main.py` | `POST /preprocess` — accepts a CSV upload, runs the full pipeline, returns JSON. |
| `guidelines/*.md` | Sample NICE markdown summaries (PDFs also supported). |

## Metadata contracts

**Patient chunk metadata:** `doc_type="patient"`, `patient_id`, plus raw column values.

**NICE chunk metadata:** `doc_type="nice_guideline"`, `guideline_name`, `guideline_id`,
`condition` (`hypertension` / `type2_diabetes` / `cardiovascular_risk` /
`chronic_kidney_disease`), `section`, `recommendation_type`
(`treatment` / `monitoring` / `diagnosis` / `lifestyle` / `general`), `source`, `chunk_index`.

The RAG layer (Section 2) can filter retrieval with:
```python
from embeddings import query_hybrid
query_hybrid("BP treatment threshold", n_results=4, doc_type="nice_guideline")
query_hybrid(patient_text, n_results=5, doc_type="patient")
```

## `/preprocess` endpoint

```bash
curl -X POST http://localhost:8000/preprocess \
  -F "file=@data/Test1 - Sheet1.csv"
```

Sample response:
```json
{
  "success": true,
  "message": "Preprocessing and hybrid embedding completed successfully.",
  "filename": "Test1 - Sheet1.csv",
  "original_shape": [199, 15],
  "cleaned_shape": [199, 15],
  "patient_id_col": "Name",
  "columns_used": ["Name", "Age", "Gender", "..."],
  "dropped_columns": [],
  "vital_columns": [],
  "date_columns": ["Date of Admission", "Discharge Date"],
  "num_patient_chunks": 199,
  "num_nice_chunks": 247,
  "total_chunks": 446,
  "guidelines_loaded": ["NG136_Hypertension", "NG28_Type2_Diabetes", "..."],
  "sample_patient_ids": ["Bobby JacksOn", "..."]
}
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# then POST a CSV to /preprocess
```
