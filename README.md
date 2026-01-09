# 🎓 University RAG: Conversational AI for University Documents

This project implements a **retrieval-augmented generation (RAG) system** tailored for university documents containing dense rules and fee tables (fees, academic regulations, hostel policies, etc.).

This system prioritizes:
1. Table preservation
2. Rule-based answer grounding
3. Hybrid retrieval (semantic + keyword)
4. Strict anti-hallucination prompting

This is **not** a toy RAG. The system is built for documents where tables are the main source of truth and calculations should not be altered

---

## 🚀 Core Capailities

* Parses complex university PDFs into Markdown
* Separates tables and text at ingestion time
* Applies structure-aware chunking only where safe
* Routes queries intelligently (fees vs rules vs text)
* Uses hybrid retrieval with routing logic
* Enforces table-first, context-only answers
* Provides a clean terminal + Streamlit interface

---

## 🧠 System Architecture

## 🧠 System Architecture

### Phase 1: High-Fidelity PDF Ingestion (`ingest.py`)

**PDF Parsing**
* Uses `LlamaParse` to convert PDFs into Markdown.
* Tables are preserved using `|`-based Markdown structure.

**Text vs Table Separation**
* Content is classified using a simple heuristic:
    ```python
    text.count("|") > 10
    ```
* Output is stored as:
    * `texts[]`
    * `tables[]`

**Manual Metadata Preservation**
* Files are processed one-by-one
* Source filename is manually injected to avoid metadata loss

**Caching**
* Parsed output is written to `parsed_data.json`
* Prevents repeated parsing API calls

---

### Phase 2: Structure-Aware Chunking (`chunk.py`)

This phase is deliberately asymmetric.

**Text Documents**
* Text content goes through two-pass chunking:
    1.  **Pass 1 — Semantic Split**
        * `MarkdownHeaderTextSplitter`
        * Splits by: `#`, `##`, `###`
        * Preserves logical sections (rules, eligibility, procedures)
    2.  **Pass 2 — Safety Split**
        * `RecursiveCharacterTextSplitter`
        * `chunk_size = 2000`
        * `chunk_overlap = 200`

* This prevents:
    * Section headers getting detached
    * Tables being cut mid-structure

**Table Documents**
* **Never chunked**
* Stored as full, atomic documents
* Treated as authoritative sources

**Final output:**
```python
final_chunks = text_chunks + table_docs
```

### Phase 3: Embedding & Storage (`embedding_store.py`)

* **Embedding Model:** `BAAI/bge-small-en-v1.5`
* **Vector Store:** ChromaDB (persistent, local)
* **Stored under:** `./vector_store`
* **Collection name:** `collec_1`

**Large chunk sizes are supported due to:**

1.  512-token embedding window
2.  Lower semantic drift compared to MiniLM

---

### Phase 4: Hybrid Retrieval + Routing (`rag.py`)

This is where your project becomes non-trivial.

#### 1. Query Classification

A lightweight router detects fee-related queries:

```python
["fee", "fees", "tuition", "admission", "cost", "payment"]
```

## 2. Retrieval Strategies

### Fee Queries
**BM25 over TABLES only**

**Guarantees:**
* Exact numeric matching
* No semantic distortion
* No recomputation

### Non-Fee Queries
Uses hybrid retrieval:
* **Dense:** `MultiQueryRetriever` (Generates multiple semantic variants using the LLM)
* **Sparse:** `BM25Retriever` over text chunks

**Combined using:**
```python
weights = [0.6 (dense), 0.4 (sparse)]
```

This avoids:
* Missing policy language
* Over-semanticizing rules

### Phase 5: Rule-Aware Answer Generation
* **LLM:** Qwen/Qwen2.5-7B-Instruct
* **Temperature:** 0.2 (low creativity)
* **Max tokens:** 1024

#### Strict System Prompt Guarantees
The model is forced to:
1. Use **ONLY** provided context
2. Treat tables as authoritative
3. Quote Grand Total rows verbatim
4. Never recompute fees
5. Refuse answers if data is missing
6. End every response with an official disclaimer

> This dramatically reduces hallucinations — especially for fee queries.
---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Language:** Python 3.10+
* **Parsing:** LlamaParse (LlamaIndex)
* **Orchestration:** LangChain
* **Retrievers:** `MultiQueryRetriever`, `BM25Retriever`, `EnsembleRetriever`
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **LLM:** HuggingFace (`Qwen/Qwen2.5-7B-Instruct`)
* **Vector DB:** ChromaDB
* **Utilities:** `python-dotenv`, `rank_bm25`

---

## 📂 Project Structure

```bash
.
├── data/                       # Raw PDF files
├── src/
│   ├── ingest.py               # Step 1: Parse PDFs -> JSON
│   ├── chunk.py                # Step 2: JSON -> Document Objects (Double-Pass Logic)
│   ├── embedding_store.py      # Step 3: Embed Chunks -> ChromaDB
│   └── rag.py                  # Step 4: Backend Logic (Hybrid Retrieval + LLM Chain)
├── vectorstore/                # Created automatically (The Local Database)
├── parsed_data.json            # Cached output (Markdown text + Metadata)
├── app.py                      # Step 5: Streamlit Frontend UI
├── requirements.txt            # Dependencies
├── .env                        # API Keys (Not committed)
└── README.md
```   
---

## 🔄 Future Roadmap

I will keep updating this repository with new features and improvements. Planned updates include:
- [ ] **Chat Memory:** Enabling the bot to remember previous messages for a natural conversation.
- [ ] **Easy Deployment:** Making the application simpler to install and run on any computer.
- [ ] ...and many more!

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

### 2. Ingest & Embed (one time setup)
#### Run these scripts to parse the PDFs and build the vector database.
```bash
python src/ingest.py
python src/chunk.py
python -m src.embedding_score
```

### 3. Run the RAG pipeline
#### Ask questions directly in the terminal using the new retrieval engine.
```bash
python src/rag.py

```

### 4. Launch the Web Interface
#### Starts the Streamlit server and opens the interactive chat in your default browser.
```bash
streamlit run app.py
```
