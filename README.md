# ⚕️ MediRAG — Hospital Patient RAG System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-orange?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-blue?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> An end-to-end **Retrieval-Augmented Generation (RAG)** system for hospital patient data — upload a CSV, ask questions in natural language, view live metrics, and generate AI-powered patient risk assessments.

---

## 📌 Project Overview

**MediRAG** is a local web application that turns raw hospital CSV data into an intelligent, queryable knowledge base. It combines vector embeddings, a persistent vector database, and large language models to deliver:

- 🔍 **Semantic search** over patient records
- 💬 **Natural language Q&A** grounded in real data
- 📊 **Auto-generated dashboards** from any hospital CSV
- 🚨 **AI risk assessment** for individual patients

The entire pipeline runs locally — no cloud database, no data leaves your machine except for LLM API calls.

---

## 🚀 Features

### 📁 Smart CSV Upload & Preprocessing
- Upload any hospital patient CSV file
- **Auto-drops columns with ≥50% null values** using pandas
- Calls **Gemini 2.5 Flash** to identify the patient ID column, clinically important columns, and generate descriptions for every column
- Cleaned data saved locally for reuse

### 🧠 Embeddings + Vector Database (RAG Core)
- Each patient row is converted into a readable text passage
- Passages are embedded using **`sentence-transformers/all-MiniLM-L6-v2`** (runs fully local)
- Stored in **ChromaDB** (persistent, cosine similarity index)
- Supports datasets from hundreds to tens of thousands of rows

### 📊 Auto Dashboard
- Key stat cards: total patients, average age, length of stay, medications, glucose, BMI
- Pure CSS bar charts — no Chart.js, no external libraries
- Auto-detects column names across different hospital dataset formats
- Covers: age distribution, gender split, top diagnoses, outcomes, admission types
- Refresh button to reload without re-uploading

### 💬 RAG Chatbot
- Ask any question about your patient data in plain English
- Pipeline: embed query → ChromaDB similarity search (top 8 records) → **Groq LLaMA 3.3 70B** → grounded answer
- Expandable source panel shows which patient records were used to answer
- Animated typing indicator, full chat history in session
- Low temperature (0.2) for factual, data-grounded responses

### 🔴 Patient Risk Detection
- Select any patient from a dropdown (up to 200 IDs)
- AI generates a structured risk report:
  - **Risk Level**: Low 🟢 / Medium 🟡 / High 🔴 / Critical 🚨
  - **Risk Score**: 0–100 with animated progress bar
  - **Risk Factors** grounded in actual patient values
  - **Protective Factors**
  - **Clinical Recommendations**
  - **Similar patient pattern** from ChromaDB context
- Expandable full patient record table

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Backend | **FastAPI** | API routes, file handling, session state |
| Frontend | **HTML + HTMX** | Dynamic UI without JS frameworks |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Local, free, fast row embeddings |
| Vector DB | **ChromaDB** | Persistent cosine-similarity vector store |
| LLM — Chat & Risk | **Groq** (`llama-3.3-70b-versatile`) | Fast free inference |
| LLM — Preprocessing | **Gemini 2.5 Flash** | Column analysis and descriptions |
| Data | **pandas** | CSV processing and metrics |
| Styling | **Pure CSS** | No Tailwind, no React, no Chart.js |

---

## 📁 File Structure

```
hospital-rag/
├── main.py              # FastAPI app — all routes and HTML builders
├── preprocessing.py     # Null-drop + Gemini column analysis
├── embeddings.py        # sentence-transformers + ChromaDB store/query
├── rag.py               # RAG pipeline — ChromaDB retrieval + Groq chat
├── dashboard.py         # Pandas metrics + pure CSS chart generation
├── risk.py              # Patient risk assessment via Groq
├── templates/
│   └── index.html       # Single-page HTML + HTMX frontend
├── static/
│   └── style.css        # Full dark-theme CSS
├── data/                # Uploaded and cleaned CSVs stored here
├── chroma_db/           # ChromaDB persistent vector store
└── requirements.txt
```

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the main UI |
| `POST` | `/upload` | Accepts CSV, validates, stores, previews columns |
| `POST` | `/preprocess` | Drops nulls, runs Gemini, embeds rows into ChromaDB |
| `GET` | `/dashboard` | Returns pandas metrics as rendered HTML |
| `POST` | `/chat` | RAG query → Groq → returns answer + sources |
| `GET` | `/patients` | Returns patient ID list for the dropdown |
| `GET` | `/risk?patient_id=X` | Full risk assessment for one patient |
| `GET` | `/status` | Current session info for the sidebar |

---

## 🔄 How It Works

```
CSV Upload
    │
    ▼
Null Column Drop (pandas ≥50%)
    │
    ▼
Gemini 2.5 Flash — Column Analysis
    │   (patient ID col, important cols, descriptions)
    ▼
Row → Text Passage Conversion
    │   "Age: 45 | Diagnosis: Hypertension | ..."
    ▼
sentence-transformers Embedding (local)
    │
    ▼
ChromaDB Vector Store (persistent)
    │
    ├──→ Dashboard (pandas metrics → CSS charts)
    │
    ├──→ RAG Chat
    │       Query → embed → ChromaDB top-8 → Groq LLaMA → Answer
    │
    └──→ Risk Detection
            Patient record + ChromaDB similar cases → Groq → Risk JSON
```

---

## 🧪 Recommended Test Datasets

| Dataset | Rows | Best For |
|---------|------|----------|
| [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | Quick testing, clean columns |
| [Heart Disease UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | Risk detection (binary outcome) |
| [Hospital Readmissions](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) | 101K | Full stress test |
| [Synthea Synthetic EHR](https://synthea.mitre.org/downloads) | 1K–100K | Most realistic hospital data |

---

## 🛠️ Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/hospital-rag.git
cd hospital-rag
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add Your API Keys

In `preprocessing.py`:
```python
GEMINI_API_KEY = "your-gemini-api-key"
```

In `rag.py` and `risk.py`:
```python
GROQ_API_KEY = "your-groq-api-key"
```

Get free keys at:
- Gemini: [aistudio.google.com](https://aistudio.google.com)
- Groq: [console.groq.com](https://console.groq.com)

### 4️⃣ Run the App
```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## 🖥️ User Flow

```
1. Upload Tab    →  Upload your hospital CSV
2. Click         →  "Run AI Preprocessing" — cleans + embeds data
3. Dashboard Tab →  View auto-generated metrics and charts
4. Chat Tab      →  Ask questions: "How many diabetic patients are over 60?"
5. Risk Tab      →  Click "Load Patients" → select one → "Analyze Risk"
```

---

## 📦 requirements.txt

```
fastapi
uvicorn
python-multipart
jinja2
pandas
sentence-transformers
chromadb
google-generativeai
groq
```

---

## 🔮 Future Enhancements

- 🔐 Multi-user sessions with file isolation
- 📤 Export risk reports as PDF
- 📈 Time-series trend charts for longitudinal data
- 🔁 Re-embedding on partial data updates
- 🌐 Support for FHIR / HL7 formats
- 📝 Chat history persistence across sessions

---

## 🤝 Contributions

Contributions are welcome. Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

**Author:** Bisistha Patra
**Contact:** patrabisistha@gmail.com
**GitHub:** [github.com/BisisthaP](https://github.com/BisisthaP)

---

💡 *If you find this project helpful, give it a ⭐️ and share it with others!*
