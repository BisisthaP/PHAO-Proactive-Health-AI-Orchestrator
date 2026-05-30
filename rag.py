from groq import Groq
from embeddings import query_similar
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a medical data analyst assistant. You have access to a hospital patient dataset.
You will be given relevant patient records retrieved from the dataset, and a user question.

Your job:
- Answer the question using ONLY the provided patient records as context
- Be specific, cite numbers and patterns you observe
- If the context doesn't have enough info, say so honestly
- Never make up patient data
- Keep answers concise but complete (3-6 sentences max unless a list is needed)
- If asked to count or aggregate, do your best with the sample provided and note it's a sample
"""


def build_context(similar_docs: list) -> str:
    parts = []
    for i, doc in enumerate(similar_docs, 1):
        parts.append(f"[Record {i}]: {doc['document']}")
    return "\n\n".join(parts)


def rag_query(question: str, n_results: int = 8) -> dict:
    """
    Full RAG pipeline:
    1. Embed question → retrieve similar patient records from ChromaDB
    2. Build context from retrieved docs
    3. Send to Groq with system prompt
    4. Return answer + source snippets
    """
    # Step 1: Retrieve
    similar_docs = query_similar(question, n_results=n_results)

    if not similar_docs:
        return {
            "answer": "No patient data found. Please upload and preprocess a CSV file first.",
            "sources": [],
            "retrieved_count": 0
        }

    # Step 2: Build context
    context = build_context(similar_docs)

    # Step 3: Groq call
    prompt = f"""Here are {len(similar_docs)} relevant patient records from the dataset:

{context}

User question: {question}

Answer based on these records:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"LLM error: {str(e)}"

    # Step 4: Return
    source_snippets = []
    for doc in similar_docs[:3]:  # show top 3 sources in UI
        snippet = doc["document"][:120] + "..." if len(doc["document"]) > 120 else doc["document"]
        source_snippets.append(snippet)

    return {
        "answer": answer,
        "sources": source_snippets,
        "retrieved_count": len(similar_docs)
    }


def format_answer_html(result: dict) -> str:
    """Convert rag_query result into HTML for the chat bubble."""
    answer_text = result["answer"].replace("\n", "<br>")

    sources_html = ""
    if result["sources"]:
        items = "".join(f"<li>{s}</li>" for s in result["sources"])
        sources_html = f"""
        <details class="source-details">
            <summary>📎 {result['retrieved_count']} records retrieved — show top 3</summary>
            <ul class="source-list">{items}</ul>
        </details>"""

    return f"""
    <div class="rag-answer">
        <div class="rag-text">{answer_text}</div>
        {sources_html}
    </div>"""
