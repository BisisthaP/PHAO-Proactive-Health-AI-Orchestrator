"""
guidelines_loader.py — Load & chunk NHS NICE clinical guidelines for PHAO.

Section 1 (Preprocessing & NICE Guidelines Integration).

Reads NICE guideline files (``.md``, ``.txt`` or ``.pdf``) from a folder,
splits them into ~400–600 token passages, and returns LangChain
``Document`` objects enriched with metadata so the RAG layer can filter on
condition, guideline and recommendation type.

Metadata attached to every chunk:
    guideline_name      e.g. "NG136_Hypertension"
    guideline_id        e.g. "NG136"
    condition           e.g. "hypertension"
    section             best-guess section heading for the chunk
    recommendation_type treatment / monitoring / diagnosis / lifestyle / general
    source              the originating filename
    chunk_index         position of the chunk within its source document
    doc_type            always "nice_guideline" (lets the vector store filter)

The loader is robust (skips unreadable files) and cacheable (results are
memoised against the directory's file signature, so repeated calls in the
same process are instant).
"""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document

# ── Configuration ────────────────────────────────────────────────────────

# Approximate token budget per chunk. We approximate tokens with whitespace
# words (1 token ≈ 0.75 words), so a 500-token target ≈ 375 words.
_TARGET_TOKENS = 500
_OVERLAP_TOKENS = 80
_WORDS_PER_TOKEN = 0.75
_TARGET_WORDS = int(_TARGET_TOKENS * _WORDS_PER_TOKEN)      # ~375
_OVERLAP_WORDS = int(_OVERLAP_TOKENS * _WORDS_PER_TOKEN)    # ~60

_SUPPORTED_EXTS = (".md", ".txt", ".pdf")

# Known NICE guideline -> condition mapping (extends automatically via keywords).
_CONDITION_KEYWORDS = {
    "hypertension": "hypertension",
    "ng136": "hypertension",
    "diabetes": "type2_diabetes",
    "ng28": "type2_diabetes",
    "cvd": "cardiovascular_risk",
    "cardio": "cardiovascular_risk",
    "lipid": "cardiovascular_risk",
    "ng238": "cardiovascular_risk",
    "cg181": "cardiovascular_risk",
    "ckd": "chronic_kidney_disease",
    "kidney": "chronic_kidney_disease",
    "ng203": "chronic_kidney_disease",
}

# In-process cache: {dir_signature: list[Document]}
_CACHE: dict[tuple, list[Document]] = {}


# ── Text extraction ──────────────────────────────────────────────────────

def _read_pdf(path: str) -> str:
    """Extract text from a PDF using pypdf. Returns '' if it cannot be read."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - dependency guard
        print("[guidelines_loader] pypdf not installed; cannot read PDF:", path)
        return ""
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:  # noqa: BLE001 - we want to skip, not crash
        print(f"[guidelines_loader] Failed to read PDF {path}: {exc}")
        return ""


def _read_text_file(path: str) -> str:
    """Read a markdown / plain-text file with a forgiving encoding."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception as exc:  # noqa: BLE001
        print(f"[guidelines_loader] Failed to read {path}: {exc}")
        return ""


def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    return _read_text_file(path)


def _clean_text(text: str) -> str:
    """Normalise whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Metadata inference ────────────────────────────────────────────────────

def _infer_guideline_id(filename: str) -> str:
    """Pull a NICE code (e.g. NG136, CG181) out of the filename if present."""
    match = re.search(r"\b([A-Za-z]{2,3}\s?\d{1,4})\b", filename)
    if match:
        return match.group(1).upper().replace(" ", "")
    return os.path.splitext(filename)[0]


def _infer_condition(filename: str) -> str:
    """Map a filename to a clinical condition via keyword lookup."""
    low = filename.lower()
    for keyword, condition in _CONDITION_KEYWORDS.items():
        if keyword in low:
            return condition
    return "general"


_REC_TYPE_PATTERNS = (
    ("treatment", ("offer", "prescribe", "treat", "start ", "titrate",
                   "medication", "drug", "statin", "therapy", "dose")),
    ("monitoring", ("monitor", "measure", "review", "follow-up", "follow up",
                    "check", "assess", "test ", "screen")),
    ("diagnosis", ("diagnos", "confirm", "identify", "criteria", "classif")),
    ("lifestyle", ("lifestyle", "diet", "exercise", "smoking", "alcohol",
                   "weight loss", "physical activity")),
)


def _infer_recommendation_type(text: str) -> str:
    """Classify a chunk's recommendation type from its wording."""
    low = text.lower()
    scores = {label: sum(low.count(kw) for kw in kws)
              for label, kws in _REC_TYPE_PATTERNS}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[0-9]+(\.[0-9]+)*\s+\S.+|[A-Z][A-Za-z \-]{3,60})$")


def _guess_section(chunk: str) -> str:
    """Best-effort section heading for a chunk (first heading-like line)."""
    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING_RE.match(stripped) and len(stripped) <= 80:
            return re.sub(r"^#{1,6}\s+", "", stripped)
        break  # only inspect the first non-empty line
    return "General"


# ── Chunking ──────────────────────────────────────────────────────────────

def _chunk_text(text: str,
                target_words: int = _TARGET_WORDS,
                overlap_words: int = _OVERLAP_WORDS) -> list[str]:
    """
    Split text into overlapping ~target_words passages.

    Splitting respects paragraph boundaries: paragraphs are accumulated
    until the word budget is reached, then a new (overlapping) chunk starts.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        # A single oversized paragraph is hard-split into word windows.
        if len(words) > target_words:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            step = max(target_words - overlap_words, 1)
            for start in range(0, len(words), step):
                window = words[start:start + target_words]
                if window:
                    chunks.append(" ".join(window))
            continue

        if current_len + len(words) > target_words and current:
            chunks.append(" ".join(current))
            # Start the next chunk with a word-level overlap tail.
            tail = " ".join(current).split()[-overlap_words:]
            current = tail + words
            current_len = len(current)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── Public API ──────────────────────────────────────────────────────────────

def _dir_signature(guidelines_dir: str) -> tuple:
    """A hashable signature of the directory contents for caching."""
    sig = []
    for name in sorted(os.listdir(guidelines_dir)):
        if name.lower().endswith(_SUPPORTED_EXTS):
            path = os.path.join(guidelines_dir, name)
            stat = os.stat(path)
            sig.append((name, stat.st_size, int(stat.st_mtime)))
    return (os.path.abspath(guidelines_dir), tuple(sig))


def clear_cache() -> None:
    """Clear the in-process guideline cache (useful after editing files)."""
    _CACHE.clear()


def load_nice_guidelines(guidelines_dir: str = "guidelines") -> list[Document]:
    """
    Load and chunk all NICE guideline files in ``guidelines_dir``.

    Supports ``.md``, ``.txt`` and ``.pdf`` files. Each file is read,
    cleaned, split into ~400–600 token passages, and converted into
    LangChain ``Document`` objects with rich filtering metadata.

    Parameters
    ----------
    guidelines_dir : str, default "guidelines"
        Folder containing the guideline files.

    Returns
    -------
    list[Document]
        One Document per chunk. Empty list if the folder is missing or
        contains no readable guideline files.
    """
    if not os.path.isdir(guidelines_dir):
        print(f"[guidelines_loader] Directory not found: {guidelines_dir}")
        return []

    # Serve from cache when the directory is unchanged.
    signature = _dir_signature(guidelines_dir)
    if signature in _CACHE:
        return _CACHE[signature]

    documents: list[Document] = []

    for name in sorted(os.listdir(guidelines_dir)):
        if not name.lower().endswith(_SUPPORTED_EXTS):
            continue

        path = os.path.join(guidelines_dir, name)
        raw = _clean_text(_read_file(path))
        if not raw:
            print(f"[guidelines_loader] Skipping empty/unreadable file: {name}")
            continue

        guideline_name = os.path.splitext(name)[0]
        guideline_id = _infer_guideline_id(name)
        condition = _infer_condition(name)

        chunks = _chunk_text(raw)
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_type": "nice_guideline",
                        "guideline_name": guideline_name,
                        "guideline_id": guideline_id,
                        "condition": condition,
                        "section": _guess_section(chunk),
                        "recommendation_type": _infer_recommendation_type(chunk),
                        "source": name,
                        "chunk_index": idx,
                    },
                )
            )

        print(f"[guidelines_loader] {name}: {len(chunks)} chunks "
              f"(id={guideline_id}, condition={condition})")

    _CACHE[signature] = documents
    print(f"[guidelines_loader] Loaded {len(documents)} guideline chunks total.")
    return documents


if __name__ == "__main__":
    docs = load_nice_guidelines("guidelines")
    print(f"\nLoaded {len(docs)} chunks.")
    if docs:
        print("\nSample chunk metadata:")
        for k, v in docs[0].metadata.items():
            print(f"  {k}: {v}")
        print("\nSample chunk text (first 300 chars):")
        print(docs[0].page_content[:300])
