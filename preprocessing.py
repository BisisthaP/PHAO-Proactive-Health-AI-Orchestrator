import pandas as pd
import google.generativeai as genai
import json
import os

GEMINI_API_KEY = "add_key"

genai.configure(api_key=GEMINI_API_KEY)


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.5):
    """Drop columns with >= threshold null ratio. No AI needed, pure pandas."""
    null_ratio = df.isnull().sum() / len(df)
    dropped = null_ratio[null_ratio >= threshold].index.tolist()
    df_clean = df.drop(columns=dropped)
    return df_clean, dropped


def gemini_analyze_columns(df: pd.DataFrame) -> dict:
    """
    Ask Gemini to identify:
    - The patient ID column
    - The most important columns for medical analysis
    - A short description of what each column means
    Returns a dict with keys: patient_id_col, important_cols, descriptions
    """
    col_samples = {}
    for col in df.columns:
        sample = df[col].dropna().head(5).tolist()
        col_samples[col] = {
            "dtype": str(df[col].dtype),
            "sample_values": [str(v) for v in sample],
            "unique_count": int(df[col].nunique())
        }

    prompt = f"""You are a medical data analyst. I have a hospital patient CSV with these columns:

{json.dumps(col_samples, indent=2)}

Return ONLY a valid JSON object (no markdown, no backticks) with this exact structure:
{{
  "patient_id_col": "<column name that best identifies a unique patient, or null if none>",
  "important_cols": ["<col1>", "<col2>", ...],
  "descriptions": {{
    "<col_name>": "<one line description of what this column means clinically>"
  }}
}}

Rules:
- important_cols should be the top medically relevant columns (max 12)
- descriptions should cover ALL columns present
- patient_id_col must be one of the actual column names or null
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    text = response.text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: use all columns
        result = {
            "patient_id_col": None,
            "important_cols": list(df.columns[:12]),
            "descriptions": {col: "No description available" for col in df.columns}
        }

    return result


def build_row_text(row: pd.Series, descriptions: dict) -> str:
    """Convert a dataframe row into a readable text passage for embedding."""
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        desc = descriptions.get(col, col)
        parts.append(f"{col} ({desc}): {val}")
    return " | ".join(parts)


def preprocess_dataframe(file_path: str) -> dict:
    """
    Full preprocessing pipeline:
    1. Load CSV
    2. Drop high-null columns (pandas)
    3. Gemini column analysis
    4. Build text passages per row
    Returns everything needed for embedding.
    """
    df = pd.read_csv(file_path)
    original_cols = list(df.columns)
    original_shape = df.shape

    # Step 1: Drop high-null columns
    df, dropped_cols = drop_high_null_columns(df, threshold=0.5)

    # Step 2: Fill remaining NaNs with "Unknown"
    df = df.fillna("Unknown")

    # Step 3: Gemini analysis
    gemini_result = gemini_analyze_columns(df)

    patient_id_col = gemini_result.get("patient_id_col")
    important_cols = gemini_result.get("important_cols", list(df.columns[:12]))
    descriptions = gemini_result.get("descriptions", {})

    # Step 4: Build text passages
    passages = []
    patient_ids = []

    for idx, row in df.iterrows():
        text = build_row_text(row, descriptions)
        passages.append(text)

        if patient_id_col and patient_id_col in row:
            patient_ids.append(str(row[patient_id_col]))
        else:
            patient_ids.append(str(idx))

    return {
        "df": df,
        "original_shape": original_shape,
        "cleaned_shape": df.shape,
        "dropped_cols": dropped_cols,
        "kept_cols": list(df.columns),
        "patient_id_col": patient_id_col,
        "important_cols": important_cols,
        "descriptions": descriptions,
        "passages": passages,
        "patient_ids": patient_ids,
    }
