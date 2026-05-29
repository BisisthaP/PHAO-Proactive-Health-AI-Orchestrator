from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hospital_patients"

# Load model once at import time
print("[embeddings] Loading sentence-transformer model...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[embeddings] Model loaded.")

# ChromaDB persistent client
_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Cache of ChromaDB clients keyed by persist directory so we don't recreate
# a client for the default path on every hybrid build.
_clients: dict[str, "chromadb.PersistentClient"] = {CHROMA_DIR: _client}


def _get_client(persist_dir: str = CHROMA_DIR):
    """Return a persistent ChromaDB client for ``persist_dir`` (cached)."""
    if persist_dir not in _clients:
        _clients[persist_dir] = chromadb.PersistentClient(path=persist_dir)
    return _clients[persist_dir]


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool values (and no None)."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (int, float, str)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def reset_collection():
    """Wipe and recreate the collection (called on new file upload)."""
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return _client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def embed_and_store(passages: list, patient_ids: list, metadata_rows: list, batch_size: int = 64):
    """
    Embed all passages and store in ChromaDB.
    metadata_rows: list of dicts, one per row, with column values as metadata.
    """
    collection = reset_collection()

    total = len(passages)
    print(f"[embeddings] Embedding {total} passages in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_passages = passages[start:end]
        batch_ids = patient_ids[start:end]
        batch_meta = metadata_rows[start:end]

        # Sanitize metadata: ChromaDB only allows str/int/float/bool values
        clean_meta = []
        for m in batch_meta:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_meta.append(clean)

        embeddings = _model.encode(batch_passages, show_progress_bar=False).tolist()

        # ChromaDB requires unique IDs — prefix with "row_" + index
        chroma_ids = [f"row_{start + i}_{pid}" for i, pid in enumerate(batch_ids)]

        collection.add(
            ids=chroma_ids,
            embeddings=embeddings,
            documents=batch_passages,
            metadatas=clean_meta
        )
        print(f"[embeddings] Stored {end}/{total}")

    print("[embeddings] All embeddings stored in ChromaDB.")
    return collection.count()


def query_similar(query_text: str, n_results: int = 5) -> list:
    """
    Embed a query and return top-n similar patient passages.
    Returns list of dicts with 'document' and 'metadata'.
    """
    collection = get_collection()
    query_embedding = _model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count())
    )

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "document": doc,
            "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
        })
    return output


def get_patient_by_id(patient_id: str) -> dict | None:
    """Fetch a specific patient's record by their ID."""
    collection = get_collection()
    results = collection.get(
        where={"$contains": patient_id} if False else None,  # fallback to search
    )
    # Use query-based retrieval instead
    hits = query_similar(f"patient {patient_id}", n_results=1)
    return hits[0] if hits else None


def collection_count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


# ── Hybrid patient + NICE guideline vector store ─────────────────────────────

def _patient_row_to_document(row, patient_id_col=None, descriptions=None):
    """Convert a patient DataFrame row into (text, metadata)."""
    import pandas as pd

    descriptions = descriptions or {}
    parts = []
    meta = {"doc_type": "patient"}

    pid = str(row[patient_id_col]) if (patient_id_col and patient_id_col in row) else None
    if pid is not None:
        parts.append(f"Patient ID: {pid}")
        meta["patient_id"] = pid

    for col, val in row.items():
        if pd.isna(val) or str(val).strip() == "":
            continue
        desc = descriptions.get(col)
        label = f"{col} ({desc})" if desc else col
        parts.append(f"{label}: {val}")
        # Keep raw column values as filterable metadata (capped to stay light).
        if len(meta) < 24:
            meta[col] = val if isinstance(val, (int, float, bool)) else str(val)

    return " | ".join(parts), meta


def build_hybrid_vectorstore(
    patient_df,
    nice_documents,
    persist_dir: str = CHROMA_DIR,
    patient_id_col: str | None = None,
    descriptions: dict | None = None,
    batch_size: int = 64,
) -> dict:
    """
    Build a hybrid ChromaDB collection from patient rows + NICE guidelines.

    Both sources live in the same collection but are distinguished by a
    ``doc_type`` metadata field ("patient" vs "nice_guideline"), enabling the
    RAG layer to retrieve from either source via metadata filtering.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Cleaned patient data (from :func:`preprocessing.preprocess_patient_data`).
    nice_documents : list[langchain_core.documents.Document]
        Chunked NICE guideline documents (from
        :func:`guidelines_loader.load_nice_guidelines`).
    persist_dir : str, default "chroma_db"
        ChromaDB persistence directory.
    patient_id_col : str | None
        Detected patient ID column (used for metadata + chunk IDs).
    descriptions : dict | None
        Optional per-column descriptions to enrich patient passages.
    batch_size : int, default 64
        Embedding batch size.

    Returns
    -------
    dict
        ``{patient_chunks, nice_chunks, total_chunks, persist_dir, collection}``.
    """
    client = _get_client(persist_dir)

    # Reset the collection so re-uploading a file starts clean.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    # 1. Patient rows -> documents.
    for idx, row in patient_df.iterrows():
        text, meta = _patient_row_to_document(row, patient_id_col, descriptions)
        if not text:
            continue
        pid = meta.get("patient_id", str(idx))
        ids.append(f"patient_{idx}_{pid}")
        documents.append(text)
        metadatas.append(_sanitize_metadata(meta))
    patient_chunks = len(documents)

    # 2. NICE guideline documents.
    for j, doc in enumerate(nice_documents or []):
        meta = dict(doc.metadata or {})
        meta.setdefault("doc_type", "nice_guideline")
        ids.append(f"nice_{j}")
        documents.append(doc.page_content)
        metadatas.append(_sanitize_metadata(meta))
    nice_chunks = len(documents) - patient_chunks

    if not documents:
        return {
            "patient_chunks": 0,
            "nice_chunks": 0,
            "total_chunks": 0,
            "persist_dir": persist_dir,
            "collection": COLLECTION_NAME,
        }

    # 3. Embed and store in batches.
    total = len(documents)
    print(f"[embeddings] Building hybrid store: {patient_chunks} patient + "
          f"{nice_chunks} NICE = {total} chunks.")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        embeddings = _model.encode(documents[start:end], show_progress_bar=False).tolist()
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings,
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"[embeddings] Stored {end}/{total}")

    print(f"[embeddings] Hybrid store ready in '{persist_dir}'.")
    return {
        "patient_chunks": patient_chunks,
        "nice_chunks": nice_chunks,
        "total_chunks": total,
        "persist_dir": persist_dir,
        "collection": COLLECTION_NAME,
    }


def query_hybrid(query_text: str, n_results: int = 5, doc_type: str | None = None,
                 where: dict | None = None) -> list:
    """
    Query the hybrid store with optional metadata filtering.

    Parameters
    ----------
    query_text : str
        Natural-language query.
    n_results : int, default 5
        Number of results to return.
    doc_type : str | None
        Restrict to "patient" or "nice_guideline" only.
    where : dict | None
        Arbitrary ChromaDB metadata filter (merged with ``doc_type``).

    Returns
    -------
    list[dict]
        Each item: ``{document, metadata, distance}``.
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    filters = dict(where) if where else {}
    if doc_type:
        filters["doc_type"] = doc_type

    query_embedding = _model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, count),
        where=filters or None,
    )

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for i, doc in enumerate(docs):
        output.append({
            "document": doc,
            "metadata": metas[i] if metas else {},
            "distance": dists[i] if dists else None,
        })
    return output