# 🎓 University RAG: Conversational AI for University Documents

> **Status:** 🚧 Work in Progress (Active Development) 🚧

This project is a Retrieval-Augmented Generation (RAG) pipeline designed to ingest, parse, and allow querying of complex IIT Bombay documents (Fee structures, Curriculum, Rules, Hostel). 

---

## 🚀 Current Workflow

The pipeline has moved beyond simple ingestion. It now includes a robust **"Double-Pass" Chunking Strategy** and **Vector Storage**.

### Phase 1: High-Fidelity Ingestion
1.  **Data Collection:** PDFs (e.g., Fee Circulars, Academic Rules) are collected in a local `data/` directory.
2.  **Advanced Parsing:** Uses **LlamaParse** to convert PDFs into structured Markdown.
    * *Why?* Standard parsers break tables. LlamaParse preserves table structure (Rows/Columns) using Markdown syntax.
3.  **Smart Caching:** Parsed data is serialized and saved to `parsed_data.json`.
    * *Benefit:* Eliminates redundant API calls to LlamaCloud, saving costs and speeding up experimentation.

### Phase 2: Structure-Aware Chunking & Embedding
4.  **Double-Pass Chunking:** We do not blind-split text. We use a two-step logic to keep tables intact:
    * **Pass 1 (Semantic):** `MarkdownHeaderTextSplitter` splits documents by Headers (`#`, `##`). This keeps logical sections (like "Hostel Rules") together.
    * **Pass 2 (Safety):** `RecursiveCharacterTextSplitter` ensures chunks fit context windows (Chunk Size: **2000 chars**).
    * *Why 2000?* A standard 500-char chunk would cut wide fee tables in half. 2000 chars ensures the Header and the Table Data stay in the same chunk.
5.  **Vector Storage:**
    * **Embedding Model:** `BAAI/bge-small-en-v1.5` (HuggingFace). Selected for its 512-token window which fits our larger chunks better than standard MiniLM models.
    * **Database:** **ChromaDB** (Local). Stores the vectors on disk for fast retrieval.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Parsing:** LlamaParse (LlamaIndex)
* **Orchestration:** LangChain
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **Vector DB:** ChromaDB
* **Utilities:** `python-dotenv`, `json`

---

## 📂 Project Structure

```bash
.
├── data/                   # Raw PDF files
├── src/
│   ├── ingest.py           # Step 1: Parse PDFs -> JSON
│   ├── chunk.py         # Step 2: JSON -> Document Objects (Double-Pass Logic)
│   └── embedding_store.py            # Step 3: Embed Chunks -> ChromaDB
├── vectorstore/            # Created automatically (The Local Database)
├── parsed_data.json        # Cached output (Markdown text + Metadata)
├── requirements.txt        # Dependencies
├── .env                    # API Keys (Not committed)
└── README.md
```

---
## ⚙️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
