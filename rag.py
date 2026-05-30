import os
import re
import math
import json
import pypdf
from collections import Counter
from dotenv import load_dotenv
from groq import Groq
from embeddings import get_collection, query_similar

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are MediRAG, an advanced, elite clinical medical AI assistant. You have access to:
1. Hospital Patient Dataset: Real patient records retrieved from our hospital clinical database.
2. NICE Guidelines: Official UK National Institute for Health and Care Excellence (NICE) guidelines for Hypertension (NG136), Chronic Kidney Disease (NG203), Cardiovascular Disease Risk (NG238), and Type 2 Diabetes (NG28).

Your objective is to provide a comprehensive, clinically grounded, and evidence-based answer to the user's question by synthesizing both patient records AND relevant clinical guidelines.

Strict Clinical Grounding & Citation Rules:
- Answer the user's question using ONLY the provided patient records and NICE guidelines sections as context.
- For EVERY clinical assertion, target value, threshold, or diagnostic recommendation, you MUST explicitly cite the specific NICE guideline and page (e.g., "[NICE NG28, p. 5]" or "[NICE NG136, p. 12]").
- For EVERY reference to patient data, demographics, patterns, or patient metrics, you MUST explicitly cite the patient record ID (e.g., "[Patient ID: 104]" or "[Patient ID: row_5]").
- Directly compare patient values against the official NICE guideline targets (e.g., "Patient ID: 45 has a blood pressure of 145/95 mmHg, which exceeds the NICE NG136 target of <140/90 mmHg for adults under 80 [NICE NG136, p. 14]").
- If the patient records or guidelines do not contain the necessary information to answer, state this clearly and specify what information is missing. Never make up patient details or fabricate clinical recommendations.
- Keep your answers structured, professional, clear, and highly clinical. Format key metrics or comparisons using markdown tables or bullet points when appropriate.
"""

GUIDELINE_NAMES = {
    "NG136_Hypertension.pdf": "NICE NG136 (Hypertension)",
    "NG203_CKD.pdf": "NICE NG203 (Chronic Kidney Disease)",
    "NG238_CVD_Risk.pdf": "NICE NG238 (Cardiovascular Disease Risk)",
    "NG28_Type2_Diabetes.pdf": "NICE NG28 (Type 2 Diabetes)"
}


# ── BM25 Sparse Retriever Implementation ──────────────────────────

class SimpleBM25:
    """A lightweight, high-performance BM25 ranker for hybrid search."""
    def __init__(self, corpus: list, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus  # list of dicts with 'document' and 'metadata'
        self.k1 = k1
        self.b = b
        self.documents = [doc["document"].lower().split() for doc in corpus]
        self.doc_len = [len(doc) for doc in self.documents]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1
        self.doc_count = len(corpus)
        
        self.df = Counter()
        for doc in self.documents:
            self.df.update(set(doc))
            
    def score(self, query_terms: list) -> list:
        """Returns corpus items with their BM25 scores."""
        scores = []
        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_len[idx]
            term_counts = Counter(doc)
            for term in query_terms:
                term = term.lower()
                df = self.df.get(term, 0)
                # Apply smoothed IDF
                idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
                tf = term_counts.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * (numerator / denominator)
            scores.append((score, self.corpus[idx]))
        return scores


# ── ChromaDB NICE Guidelines Setup ────────────────────────────────

def get_nice_collection():
    """Fetch or create the ChromaDB collection for NICE guidelines."""
    from embeddings import _client
    return _client.get_or_create_collection(
        name="nice_guidelines",
        metadata={"hnsw:space": "cosine"}
    )


def index_nice_guidelines():
    """Automatically parse and embed NICE guideline PDFs if the collection is empty."""
    collection = get_nice_collection()
    if collection.count() > 0:
        return
        
    print("[rag] NICE guidelines collection empty. Beginning indexing...")
    import os
    from embeddings import _model
    
    guidelines_dir = "/home/almostblue/dev/phoa/guidelines"
    if not os.path.exists(guidelines_dir):
        print(f"[rag] Guidelines directory not found at: {guidelines_dir}")
        return
        
    pdf_files = [f for f in os.listdir(guidelines_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("[rag] No clinical guideline PDFs found.")
        return
        
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for pdf_file in pdf_files:
        path = os.path.join(guidelines_dir, pdf_file)
        guideline_title = GUIDELINE_NAMES.get(pdf_file, pdf_file.replace(".pdf", ""))
        print(f"[rag] Parsing guidelines: {pdf_file}...")
        
        try:
            reader = pypdf.PdfReader(path)
            chunk_idx = 0
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or len(text.strip()) < 40:
                    continue
                
                # Split text into overlapping word chunks (~120 words with 20 words overlap)
                words = text.split()
                chunk_words = 120
                overlap_words = 20
                
                for i in range(0, len(words), chunk_words - overlap_words):
                    chunk_text = " ".join(words[i:i + chunk_words])
                    if len(chunk_text.strip()) < 40:
                        continue
                    
                    doc_id = f"{pdf_file.replace('.pdf', '')}_p{page_idx+1}_{chunk_idx}"
                    all_ids.append(doc_id)
                    all_chunks.append(chunk_text)
                    all_metadata.append({
                        "source": pdf_file,
                        "guideline": guideline_title,
                        "page": page_idx + 1
                    })
                    chunk_idx += 1
        except Exception as e:
            print(f"[rag] Error parsing {pdf_file}: {e}")
            
    if all_chunks:
        print(f"[rag] Embedding {len(all_chunks)} chunks using SentenceTransformer...")
        embeddings = _model.encode(all_chunks, show_progress_bar=False).tolist()
        
        # Batch upload to ChromaDB
        batch_size = 100
        for start in range(0, len(all_chunks), batch_size):
            end = min(start + batch_size, len(all_chunks))
            collection.add(
                ids=all_ids[start:end],
                embeddings=embeddings[start:end],
                documents=all_chunks[start:end],
                metadatas=all_metadata[start:end]
            )
        print(f"[rag] NICE guidelines indexing complete! {collection.count()} chunks stored.")


# ── Advanced Retrievers with RRF & Metadata Filters ────────────────

def get_dataset_info() -> str:
    """Fetch columns and sample data from the patient collection to help the clinical parser."""
    try:
        coll = get_collection()
        count = coll.count()
        if count == 0:
            return "No patient data loaded."
        sample = coll.get(limit=1)
        if sample and sample["metadatas"]:
            meta = sample["metadatas"][0]
            return f"Columns: {list(meta.keys())}. Sample: {meta}"
    except Exception:
        pass
    return "Columns: PatientID, Age, Gender, BloodPressure, HeartRate, Diabetes, CKD, Hypertension, BMI, Cholesterol."


def parse_clinical_intent(question: str) -> dict:
    """Uses Groq (with LLaMA-3.3) or Gemini fallback to parse clinical intent and metadata filters."""
    schema_info = get_dataset_info()
    
    prompt = f"""You are an advanced medical clinical intent router. Analyze the user query: "{question}"
    
    Hospital Dataset Schema:
    {schema_info}
    
    Task: Parse the clinical context and metadata filtering details, and return a JSON object with:
    1. "is_clinical_recommendation": true if the query is seeking clinical advice, thresholds, diagnoses, clinical guidelines (NICE), treatments, or recommendations; false if it's purely a patient count/statistics question.
    2. "patient_id": a specific patient ID (as a string or integer) if the query refers to a specific patient, else null.
    3. "nice_guideline_filters": list of strings matching relevant guidelines: ["Hypertension", "CKD", "CVD", "Diabetes"] or empty if none.
    4. "patient_metadata_filters": a flat dict of simple ChromaDB filters, or null. ChromaDB where clauses only support exact match or simple operator dicts, e.g. {{"Age": {{"$gt": 60}}}}, or flat equality {{"Gender": "Male"}}. Keep keys exactly as they appear in the schema! Use null if no clear filters exist.
    5. "dense_query": expanded search query for vector retrieval (e.g. synonyms, clinical terms).
    6. "sparse_keywords": a list of 3-5 distinct keyword strings for text search (e.g. ["diabetes", "glucose", "hba1c"]).
    
    Return ONLY a raw JSON block (do NOT put it in markdown code blocks, do NOT add comments, do NOT write any introductory or concluding text).
    JSON:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise clinical JSON parser. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[rag] Groq query parser failed: {e}. Trying Gemini...")
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel("gemini-2.5-flash")
            gemini_resp = model.generate_content(prompt)
            text = gemini_resp.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception as ex:
            print(f"[rag] Gemini query parser failed: {ex}. Using regex fallback.")
            
            # Direct Regex/Keyword heuristic fallback
            lower_q = question.lower()
            patient_id = None
            import re
            pid_match = re.search(r'(?:patient|id)\s*#?\s*([a-zA-Z0-9_-]+)', lower_q)
            if pid_match:
                patient_id = pid_match.group(1)
                
            is_clinical = any(w in lower_q for w in ["guideline", "nice", "recommend", "treat", "threshold", "clinical", "should", "diagnose", "manage"])
            
            nice_filters = []
            if "hyper" in lower_q or "bp" in lower_q or "blood pressure" in lower_q:
                nice_filters.append("Hypertension")
            if "ckd" in lower_q or "kidney" in lower_q or "renal" in lower_q:
                nice_filters.append("CKD")
            if "cvd" in lower_q or "cardio" in lower_q or "heart" in lower_q or "stroke" in lower_q:
                nice_filters.append("CVD")
            if "diab" in lower_q or "glucose" in lower_q or "sugar" in lower_q:
                nice_filters.append("Diabetes")
                
            return {
                "is_clinical_recommendation": is_clinical,
                "patient_id": patient_id,
                "nice_guideline_filters": nice_filters,
                "patient_metadata_filters": None,
                "dense_query": question,
                "sparse_keywords": [w for w in question.split() if len(w) > 4]
            }


def get_patient_candidates(dense_query: str, sparse_terms: list, metadata_filter: dict = None, top_k: int = 15) -> list:
    """Retrieves patient records using Dense and Sparse search combined via Reciprocal Rank Fusion (RRF)."""
    coll = get_collection()
    count = coll.count()
    if count == 0:
        return []
        
    dense_kwargs = {"n_results": min(100, count)}
    if metadata_filter:
        dense_kwargs["where"] = metadata_filter

    from embeddings import _model
    query_emb = _model.encode([dense_query]).tolist()
    
    try:
        dense_results = coll.query(
            query_embeddings=query_emb,
            **dense_kwargs
        )
    except Exception as e:
        print(f"[rag] ChromaDB query with metadata filter failed: {e}. Retrying without filter...")
        dense_kwargs.pop("where", None)
        dense_results = coll.query(
            query_embeddings=query_emb,
            **dense_kwargs
        )

    candidates = []
    if dense_results and "documents" in dense_results and dense_results["documents"]:
        for i, doc in enumerate(dense_results["documents"][0]):
            candidates.append({
                "id": dense_results["ids"][0][i],
                "document": doc,
                "metadata": dense_results["metadatas"][0][i] if dense_results["metadatas"] else {},
                "dense_score": dense_results["distances"][0][i] if "distances" in dense_results else 1.0,
                "type": "patient"
            })
            
    if not candidates:
        return []
        
    # BM25 sparse keyword ranking
    bm25 = SimpleBM25(candidates)
    bm25_scores = bm25.score(sparse_terms)
    
    # RRF (Reciprocal Rank Fusion)
    candidates_by_dense = sorted(candidates, key=lambda x: x["dense_score"])
    candidates_by_sparse = sorted(bm25_scores, key=lambda x: x[0], reverse=True)
    
    rrf_scores = {}
    for rank, c in enumerate(candidates_by_dense):
        doc_id = c["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank + 1)
        
    for rank, (score, c) in enumerate(candidates_by_sparse):
        doc_id = c["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank + 1)
        
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    cand_map = {c["id"]: c for c in candidates}
    
    final_candidates = []
    for cid in sorted_ids[:top_k]:
        cand = cand_map[cid]
        cand["rrf_score"] = rrf_scores[cid]
        final_candidates.append(cand)
        
    return final_candidates


def get_nice_candidates(dense_query: str, sparse_terms: list, guideline_filters: list = None, top_k: int = 15) -> list:
    """Retrieves NICE guideline sections using hybrid search (Dense + Sparse) + specific guideline filtering."""
    collection = get_nice_collection()
    count = collection.count()
    if count == 0:
        return []
        
    dense_kwargs = {"n_results": min(100, count)}
    
    if guideline_filters:
        mapping = {
            "hypertension": "NG136_Hypertension.pdf",
            "ckd": "NG203_CKD.pdf",
            "cvd": "NG238_CVD_Risk.pdf",
            "diabetes": "NG28_Type2_Diabetes.pdf"
        }
        matched_files = []
        for g in guideline_filters:
            gl = g.lower()
            for key, val in mapping.items():
                if key in gl or gl in key:
                    matched_files.append(val)
                    
        if matched_files:
            if len(matched_files) == 1:
                dense_kwargs["where"] = {"source": matched_files[0]}
            else:
                dense_kwargs["where"] = {"source": {"$in": matched_files}}
                
    from embeddings import _model
    query_emb = _model.encode([dense_query]).tolist()
    
    try:
        dense_results = collection.query(
            query_embeddings=query_emb,
            **dense_kwargs
        )
    except Exception as e:
        print(f"[rag] ChromaDB guidelines query failed: {e}. Retrying without filter...")
        dense_kwargs.pop("where", None)
        dense_results = collection.query(
            query_embeddings=query_emb,
            **dense_kwargs
        )

    candidates = []
    if dense_results and "documents" in dense_results and dense_results["documents"]:
        for i, doc in enumerate(dense_results["documents"][0]):
            candidates.append({
                "id": dense_results["ids"][0][i],
                "document": doc,
                "metadata": dense_results["metadatas"][0][i] if dense_results["metadatas"] else {},
                "dense_score": dense_results["distances"][0][i] if "distances" in dense_results else 1.0,
                "type": "nice"
            })
            
    if not candidates:
        return []
        
    # BM25 sparse keyword ranking
    bm25 = SimpleBM25(candidates)
    bm25_scores = bm25.score(sparse_terms)
    
    # RRF (Reciprocal Rank Fusion)
    candidates_by_dense = sorted(candidates, key=lambda x: x["dense_score"])
    candidates_by_sparse = sorted(bm25_scores, key=lambda x: x[0], reverse=True)
    
    rrf_scores = {}
    for rank, c in enumerate(candidates_by_dense):
        doc_id = c["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank + 1)
        
    for rank, (score, c) in enumerate(candidates_by_sparse):
        doc_id = c["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank + 1)
        
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    cand_map = {c["id"]: c for c in candidates}
    
    final_candidates = []
    for cid in sorted_ids[:top_k]:
        cand = cand_map[cid]
        cand["rrf_score"] = rrf_scores[cid]
        final_candidates.append(cand)
        
    return final_candidates


def build_context_text(nice_docs: list, patient_docs: list) -> str:
    """Assembles structured context blocks to guide the LLM's clinical reasoning."""
    context_parts = []
    
    context_parts.append("=== NICE CLINICAL GUIDELINES SECTION ===")
    if nice_docs:
        for idx, doc in enumerate(nice_docs, 1):
            meta = doc["metadata"]
            context_parts.append(
                f"[Guideline Source {idx}]:\n"
                f"Document: {meta.get('guideline', 'NICE Guideline')}\n"
                f"Filename: {meta.get('source', 'Unknown')}\n"
                f"Page: {meta.get('page', 'Unknown')}\n"
                f"Content:\n\"\"\"{doc['document']}\"\"\""
            )
    else:
        context_parts.append("No NICE guidelines matched this clinical query.")
        
    context_parts.append("\n=== HOSPITAL PATIENT RECORDS SECTION ===")
    if patient_docs:
        for idx, doc in enumerate(patient_docs, 1):
            meta = doc["metadata"]
            patient_id = meta.get("PatientID", meta.get("patient_id", doc["id"]))
            context_parts.append(
                f"[Patient Record {idx}]:\n"
                f"Patient ID: {patient_id}\n"
                f"Data: {doc['document']}"
            )
    else:
        context_parts.append("No patient records matched this query.")
        
    return "\n\n".join(context_parts)


# ── Full Hybrid RAG Pipeline ──────────────────────────────────────

def rag_query(question: str, n_results: int = 8) -> dict:
    """
    Advanced Hybrid RAG Pipeline:
    1. Lazy initialization: Verify NICE guidelines are indexed in ChromaDB.
    2. Intent Parser: Call Groq/Gemini to extract clinical intent, patient IDs, and metadata filters.
    3. Document Budgeting: Allocate slots to NICE guidelines vs Patient records dynamically.
    4. Retrievals: Run hybrid RRF searches with metadata filtering for guidelines and patient records.
    5. Synthesis: Construct structured prompt and call Groq LLaMA-3.3 (with Gemini 2.5 fallback).
    6. Return: Answer, structured sources list, and metrics.
    """
    # Step 1: Ensure NICE Guidelines are indexed
    index_nice_guidelines()
    
    # Step 2: Extract Intent and Filters
    intent = parse_clinical_intent(question)
    is_clinical = intent.get("is_clinical_recommendation", False)
    patient_id = intent.get("patient_id")
    nice_filters = intent.get("nice_guideline_filters", [])
    patient_filters = intent.get("patient_metadata_filters")
    dense_query = intent.get("dense_query", question)
    sparse_keywords = intent.get("sparse_keywords", [])
    
    if not sparse_keywords:
        sparse_keywords = [w for w in question.split() if len(w) > 4]

    # Step 3: Document Budget Allocation
    try:
        from embeddings import collection_count
        patient_count = collection_count()
    except Exception:
        patient_count = 0

    if patient_count == 0:
        n_nice = n_results
        n_patient = 0
    elif is_clinical:
        # Prioritize Guidelines
        n_nice = max(5, int(n_results * 0.65))
        n_patient = max(2, n_results - n_nice)
    else:
        # Prioritize Patients
        n_patient = max(5, int(n_results * 0.65))
        n_nice = max(2, n_results - n_patient)

    # Step 4: Perform Retrievals
    nice_candidates = get_nice_candidates(
        dense_query=dense_query,
        sparse_terms=sparse_keywords,
        guideline_filters=nice_filters,
        top_k=n_nice * 2
    )
    
    patient_candidates = get_patient_candidates(
        dense_query=dense_query,
        sparse_terms=sparse_keywords + ([str(patient_id)] if patient_id else []),
        metadata_filter=patient_filters,
        top_k=n_patient * 2
    )
    
    # Exact lookup supplement if a patient ID is specified
    if patient_id:
        from embeddings import query_similar
        direct_hits = query_similar(f"patient {patient_id}", n_results=2)
        for hit in direct_hits:
            hit["type"] = "patient"
            hit["id"] = hit.get("id", f"exact_{patient_id}")
            if not any(c["document"] == hit["document"] for c in patient_candidates):
                patient_candidates.insert(0, hit)
                
    selected_nice = nice_candidates[:n_nice]
    selected_patient = patient_candidates[:n_patient]
    
    # Step 5: Synthesize and Call LLM
    context = build_context_text(selected_nice, selected_patient)
    
    prompt = f"""Here is the retrieved clinical evidence and dataset context:
    
{context}

User question: {question}

Provide a clinical, fact-grounded synthesis answer based on this context. Ensure that all claims, targets, and guidelines are cited like [NICE NG28, p. 5], and all patient data is cited like [Patient ID: 104]. Let's think step by step to ensure absolute accuracy."""

    answer = ""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[rag] Groq LLaMA synthesis failed: {e}. Trying Gemini...")
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=SYSTEM_PROMPT
            )
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as ex:
            answer = f"RAG LLM synthesis error: {str(ex)}"

    # Step 6: Prepare Sources Output
    sources = []
    for doc in selected_nice:
        meta = doc["metadata"]
        sources.append({
            "type": "nice",
            "guideline": meta.get("guideline", "NICE Guideline"),
            "source": meta.get("source", "NICE Document"),
            "page": meta.get("page", "Unknown"),
            "text": doc["document"]
        })
        
    for doc in selected_patient:
        meta = doc["metadata"]
        patient_id_val = meta.get("PatientID", meta.get("patient_id", doc.get("id", "Unknown")))
        sources.append({
            "type": "patient",
            "patient_id": patient_id_val,
            "text": doc["document"]
        })

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_count": len(selected_nice) + len(selected_patient),
        "nice_count": len(selected_nice),
        "patient_count": len(selected_patient)
    }


def format_answer_html(result: dict) -> str:
    """Converts rag_query result into highly polished HTML for the chat UI."""
    answer_text = result["answer"]
    
    # Formatting conversions
    answer_html = answer_text.replace("\n", "<br>")
    answer_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', answer_html)
    answer_html = re.sub(r'(?:^|<br>)\s*[-*]\s+(.*?)(?=$|<br>)', r'<li>\1</li>', answer_html)
    
    sources = result.get("sources", [])
    sources_html = ""
    
    if sources:
        nice_sources = [s for s in sources if s.get("type") == "nice"]
        patient_sources = [s for s in sources if s.get("type") == "patient"]
        
        nice_items = []
        for s in nice_sources:
            nice_items.append(
                f"""<li>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span class="badge badge-ok">📘 {s.get('guideline', 'NICE Guideline')}</span>
                        <span style="font-size: 10px; color: var(--accent); font-weight: bold;">Page {s.get('page', 'N')}</span>
                    </div>
                    <div style="font-style: italic; font-size: 11px; margin-top: 4px; color: var(--text-muted); line-height: 1.4;">
                        "{s.get('text', '')[:160]}..."
                    </div>
                </li>"""
            )
            
        patient_items = []
        for s in patient_sources:
            patient_items.append(
                f"""<li>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span class="badge badge-warn">👤 Patient Record</span>
                        <span style="font-size: 10px; color: var(--yellow); font-weight: bold;">ID: {s.get('patient_id', 'Unknown')}</span>
                    </div>
                    <div style="font-family: monospace; font-size: 11px; margin-top: 4px; color: var(--text-muted); line-height: 1.4;">
                        {s.get('text', '')}
                    </div>
                </li>"""
            )
            
        nice_section = ""
        if nice_items:
            nice_section = f"""
            <div style="margin-bottom: 12px;">
                <div class="section-title" style="margin-bottom: 6px; font-size: 11px; color: var(--accent); font-weight: bold; border-bottom: 1px solid var(--border); padding-bottom: 2px;">
                    📘 NICE Clinical Guidelines ({len(nice_items)} segments)
                </div>
                <ul class="source-list" style="margin-top: 4px;">{ "".join(nice_items) }</ul>
            </div>"""
            
        patient_section = ""
        if patient_items:
            patient_section = f"""
            <div>
                <div class="section-title" style="margin-bottom: 6px; font-size: 11px; color: var(--yellow); font-weight: bold; border-bottom: 1px solid var(--border); padding-bottom: 2px;">
                    📊 Hospital Patient Records ({len(patient_items)} rows)
                </div>
                <ul class="source-list" style="margin-top: 4px;">{ "".join(patient_items) }</ul>
            </div>"""
            
        total_count = result.get("retrieved_count", len(sources))
        sources_html = f"""
        <details class="source-details" style="margin-top: 14px;">
            <summary style="font-weight: bold; font-size: 11px; outline: none; transition: color 0.15s;">📎 Hybrid Retrieval Sources ({total_count} records referenced)</summary>
            <div style="padding: 12px; background: var(--bg3); border-radius: 8px; border: 1px solid var(--border); margin-top: 8px; max-height: 350px; overflow-y: auto;">
                {nice_section}
                {patient_section}
            </div>
        </details>"""
        
    return f"""
    <div class="rag-answer">
        <div class="rag-text" style="color: var(--text);">{answer_html}</div>
        {sources_html}
    </div>"""
