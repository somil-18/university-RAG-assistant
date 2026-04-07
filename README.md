# 🎓 University RAG: Conversational AI for IIT Documents

This project implements a **retrieval-augmented generation (RAG) system** tailored for IIT university documents containing dense rules and fee tables (fees, academic regulations, hostel policies, etc.).

This system prioritizes:
1. High-fidelity table preservation (HTML-aware)
2. Rule-based answer grounding
3. Semantic retrieval with reranking
4. Strict anti-hallucination prompting

---

## 🚀 Core Capabilities

* Parses complex university PDFs into Markdown using LlamaCloud (agentic tier)
* Structure-aware chunking — HTML tables are never split
* Semantic vector retrieval with neural reranking
* Strict context-only answer generation
* Clean streaming Streamlit interface

---

## 🧠 System Architecture

### Phase 1: High-Fidelity PDF Parsing (`parser.py`)

* Uses `LlamaCloud` sync client to convert PDFs into Markdown
* Parses files one-by-one to avoid metadata loss
* Each file's content is prefixed with `# filename.pdf` as a top-level header for downstream source tracking
* All pages joined with `---` separators
* Output written to `parsed_data.md`

---

### Phase 2: Structure-Aware Chunking & Storage (`chunk_store.py`)

This phase is deliberately asymmetric.

**Chunking Strategy**

Pass 1 — Semantic Split:
* `MarkdownHeaderTextSplitter` splits by `#`, `##`, `###`
* Preserves logical document sections

Pass 2 — Safety Split (text only):
* `RecursiveCharacterTextSplitter` with `chunk_size=4000`, `chunk_overlap=200`
* Only applied to non-table, oversized text chunks

**Table Handling**
* Tables detected via HTML: `"<table" in text.lower()`
* Tables are **never split** — stored as atomic documents
* Oversized tables (>4000 chars) are truncated with a warning log

**Metadata**
* All metadata stripped to just `source` (filename) to stay under Pinecone's 40KB per vector limit

**Embedding & Storage**
* Embedding Model: `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048 dims)
* `truncate="END"` to safely handle edge cases
* Vector DB: Pinecone (Dense, `us-east-1`, cosine metric, 2048 dims)

---

### Phase 3: Retrieval (`retriever.py`)

* Connects to Pinecone vectorstore
* Fetches top 20 chunks via vector similarity search
* Filters empty chunks before reranking
* Truncates chunks >6000 chars before passing to reranker
* Reranker: `nvidia/llama-3.2-nv-rerankqa-1b-v2` → returns top 5 most relevant chunks

---

### Phase 4: Answer Generation (`rag.py`)

* LLM: `llama-3.3-70b-versatile` via Groq
* Streams response chunk by chunk

**System Prompt Guarantees**
* Use ONLY provided context
* If answer not in context → "I don't have that information." and nothing else
* Table-first answers for structured data
* Concise, to-the-point responses

---

### Phase 5: Streamlit Frontend (`app.py`)

* Streaming responses via `st.write_stream()`
* Full chat history with `st.session_state`
* Clean error handling per message

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Frontend | Streamlit |
| Language | Python 3.10+ |
| Parsing | LlamaCloud (agentic tier) |
| Orchestration | LangChain |
| Embeddings | NVIDIA AI Endpoints (`llama-3.2-nv-embedqa-1b-v2`) |
| Reranker | NVIDIA AI Endpoints (`llama-3.2-nv-rerankqa-1b-v2`) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Vector DB | Pinecone (Dense, 2048 dims) |
| Utilities | `python-dotenv`, `langchain`, `langchain-pinecone` |

---

## 📂 Project Structure

```
.
├── data/                   # Raw PDF files
├── src/
│   ├── parser.py           # Step 1: Parse PDFs → parsed_data.md
│   ├── chunk_store.py      # Step 2: Chunk + Embed + Upload to Pinecone
│   ├── retriever.py        # Step 3: Vector search + Reranking
│   └── rag.py              # Step 4: Prompt + LLM generation (streaming)
├── parsed_data.md          # Cached parsed output
├── app.py                  # Streamlit frontend
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🎥 Demo
Check out the hybrid search in action

https://github.com/user-attachments/assets/aa66ae53-359f-4d66-9bff-8952d61e4695

---

## ⚙️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Parse PDFs (one-time)
```bash
python src/parser.py
```

### 3. Chunk & Upload to Pinecone (one-time)
```bash
python src/chunk_store.py
```

### 4. Launch the Web Interface
```bash
streamlit run app.py
```

