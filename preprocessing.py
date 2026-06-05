"""
preprocessing.py — Enhanced patient-data preprocessing for PHAO.

Section 1 (Preprocessing & NICE Guidelines Integration).

This module turns a raw, messy patient CSV/EHR DataFrame into a clean
DataFrame plus a rich metadata dictionary that the rest of the pipeline
(embeddings, RAG, risk agent) relies on.

Key responsibilities:
    * Drop columns that are mostly empty (< 50% non-null).
    * Auto-detect the patient identifier column (id / patient / mrn / nhs ...).
    * Detect clinical vital / time-series columns (bp, glucose, heart rate ...).
    * Detect date columns.
    * Fill remaining missing values intelligently (numeric -> median,
      categorical -> "Unknown").

The whole module is pure pandas — no external LLM calls — so it runs
fully locally and is fast and deterministic.
"""

from __future__ import annotations

import re
import warnings
import pandas as pd

# ── Detection keyword banks ──────────────────────────────────────────────

# Substrings that strongly suggest a column is a unique patient identifier.
_ID_KEYWORDS = ("patient_id", "patientid", "mrn", "nhs", "subject_id",
                "subjectid", "case_id", "record_id", "uhid", "hospital_no")
# Weaker hints — only treated as an ID if the column is also (near) unique.
_ID_HINTS = ("id", "patient", "subject", "case", "mrn", "nhs", "uid")

# Clinical vitals / measurements that are worth tracking and trending.
_VITAL_KEYWORDS = (
    "bp", "blood pressure", "blood_pressure", "systolic", "diastolic",
    "glucose", "sugar", "hba1c", "a1c", "heart rate", "heart_rate", "pulse",
    "bmi", "weight", "height", "cholesterol", "ldl", "hdl", "triglyceride",
    "egfr", "creatinine", "temperature", "temp", "spo2", "oxygen",
    "respiratory", "resp_rate", "respiration",
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Lower-case a column name and collapse separators for matching."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.5):
    """
    Drop columns whose non-null ratio is below ``threshold``.

    A column is kept only if at least ``threshold`` (default 50%) of its
    values are populated.

    Returns
    -------
    (cleaned_df, dropped_cols) : tuple[pd.DataFrame, list[str]]
    """
    if len(df) == 0:
        return df.copy(), []
    non_null_ratio = df.notna().sum() / len(df)
    dropped = non_null_ratio[non_null_ratio < threshold].index.tolist()
    return df.drop(columns=dropped), dropped


def _looks_like_dates(series: pd.Series, sample: int = 50) -> bool:
    """Heuristically decide whether a column's values parse as dates."""
    values = series.dropna().astype(str).head(sample)
    if len(values) == 0:
        return False
    # Avoid matching plain integers/floats (e.g. IDs, ages) as dates.
    if pd.api.types.is_numeric_dtype(series):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    # Treat as a date column when most non-null samples parse cleanly.
    return parsed.notna().mean() >= 0.7


def detect_patient_id_column(df: pd.DataFrame) -> str | None:
    """
    Find the column most likely to be the unique patient identifier.

    Strategy
    --------
    1. Exact-ish keyword match (patient_id, mrn, nhs ...) wins immediately.
    2. Otherwise score weaker hint columns by uniqueness ratio.
    3. As a last resort pick any fully-unique, non-date column.
    Returns the column name, or ``None`` if nothing convincing is found.
    """
    if df.shape[1] == 0:
        return None

    norm_map = {col: _norm(col) for col in df.columns}
    n = max(len(df), 1)

    # 1. Strong keyword match.
    for col, norm in norm_map.items():
        if any(kw in norm for kw in _ID_KEYWORDS):
            return col

    # 2. Weaker hints, ranked by uniqueness.
    candidates = []
    for col, norm in norm_map.items():
        if any(re.search(rf"(^|_){hint}($|_)", norm) or norm.endswith(hint)
               for hint in _ID_HINTS):
            uniqueness = df[col].nunique(dropna=True) / n
            candidates.append((uniqueness, col))
    if candidates:
        candidates.sort(reverse=True)
        best_uniqueness, best_col = candidates[0]
        if best_uniqueness >= 0.5:  # at least half the rows are distinct
            return best_col

    # 3. Any near-perfectly-unique, non-date column.
    for col in df.columns:
        if _looks_like_dates(df[col]):
            continue
        if df[col].nunique(dropna=True) / n >= 0.95:
            return col

    return None


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect date / timestamp columns.

    A column qualifies only if its *values* actually parse as dates. A
    date-like name alone is not enough (e.g. "Admission Type" is categorical,
    not a date), which avoids common false positives.
    """
    return [col for col in df.columns if _looks_like_dates(df[col])]


def detect_vital_columns(df: pd.DataFrame, date_cols: list[str]) -> list[str]:
    """
    Detect clinical vital / time-series columns.

    A column qualifies if its name matches a known vital keyword. Date
    columns are excluded (they are tracked separately).
    """
    vital_cols = []
    for col in df.columns:
        if col in date_cols:
            continue
        norm = _norm(col)
        if any(kw.replace(" ", "_") in norm or kw in str(col).lower()
               for kw in _VITAL_KEYWORDS):
            vital_cols.append(col)
    return vital_cols


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values intelligently.

    * Numeric columns      -> median (falls back to 0 if all-null).
    * Everything else      -> the literal string ``"Unknown"``.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                df[col] = df[col].fillna(median if pd.notna(median) else 0)
            else:
                df[col] = df[col].fillna("Unknown")
    return df


def build_row_text(row: pd.Series, patient_id_col: str | None = None) -> str:
    """Convert a DataFrame row into a readable passage for embedding."""
    parts = []
    if patient_id_col and patient_id_col in row:
        parts.append(f"Patient ID: {row[patient_id_col]}")
    for col, val in row.items():
        if col == patient_id_col:
            continue
        if pd.isna(val) or str(val).strip() == "":
            continue
        parts.append(f"{col}: {val}")
    return " | ".join(parts)


# ── Main entry point ───────────────────────────────────────────────────────

def preprocess_patient_data(
    df: pd.DataFrame,
    null_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean a raw patient DataFrame and extract structural metadata.

    Pipeline
    --------
    1. Drop columns with fewer than ``null_threshold`` non-null values.
    2. Auto-detect the patient ID column.
    3. Detect date columns and clinical vital / time-series columns.
    4. Fill remaining missing values (numeric -> median, else -> "Unknown").

    Parameters
    ----------
    df : pd.DataFrame
        The raw uploaded patient data.
    null_threshold : float, default 0.5
        Minimum fraction of populated values required to keep a column.

    Returns
    -------
    (cleaned_df, metadata) : tuple[pd.DataFrame, dict]
        ``metadata`` keys:
            patient_id_col, vital_cols, date_cols, numeric_cols,
            categorical_cols, dropped_cols, kept_cols, original_shape,
            cleaned_shape, n_patients, sample_patient_ids, patient_ids.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If ``df`` is empty (no rows or no columns).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Input DataFrame is empty (no rows or no columns).")

    original_shape = df.shape

    # 1. Drop mostly-empty columns.
    df, dropped_cols = drop_high_null_columns(df, threshold=null_threshold)

    if df.shape[1] == 0:
        raise ValueError(
            "All columns were dropped — every column was below the "
            f"{null_threshold:.0%} non-null threshold."
        )

    # 2 & 3. Structural detection (done before fill so detection sees real gaps).
    patient_id_col = detect_patient_id_column(df)
    date_cols = detect_date_columns(df)
    vital_cols = detect_vital_columns(df, date_cols)

    # 4. Fill missing values.
    df = fill_missing_values(df)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Build the list of patient identifiers (fall back to the row index).
    if patient_id_col and patient_id_col in df.columns:
        patient_ids = df[patient_id_col].astype(str).tolist()
    else:
        patient_ids = [str(i) for i in range(len(df))]

    metadata = {
        "patient_id_col": patient_id_col,
        "vital_cols": vital_cols,
        "date_cols": date_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_cols": dropped_cols,
        "kept_cols": list(df.columns),
        "original_shape": original_shape,
        "cleaned_shape": df.shape,
        "n_patients": len(df),
        "patient_ids": patient_ids,
        "sample_patient_ids": patient_ids[:5],
    }

    return df, metadata


# ── Backward-compatible file-path wrapper ────────────────────────────────────

def preprocess_dataframe(file_path: str) -> dict:
    """
    Convenience wrapper used by the file-upload flow.

    Loads a CSV from ``file_path``, runs :func:`preprocess_patient_data`,
    and additionally returns ready-to-embed text passages so callers that
    expect the legacy dict shape keep working.
    """
    df = pd.read_csv(file_path)
    cleaned_df, meta = preprocess_patient_data(df)

    passages = [
        build_row_text(row, meta["patient_id_col"])
        for _, row in cleaned_df.iterrows()
    ]

    return {
        "df": cleaned_df,
        "passages": passages,
        # Legacy keys kept for any older callers.
        "important_cols": meta["vital_cols"] + (
            [meta["patient_id_col"]] if meta["patient_id_col"] else []
        ),
        "descriptions": {},
        **meta,
    }
