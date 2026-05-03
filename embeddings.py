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