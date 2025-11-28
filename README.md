# 🎓 university RAG: Conversational AI for University Documents

> **Status:** 🚧 Work in Progress (Active Development) 🚧

This project is a Retrieval-Augmented Generation (RAG) pipeline designed to ingest, parse, and allow querying of complex IIT Bombay documents (Fee structures, Curriculum, Rules, Hostel). 

---

## 🚀 Current Workflow

Right now, the pipeline focuses on **High-Fidelity Data Ingestion**:

1.  **Data Collection:** PDFs (e.g., Fee Circulars, Academic Rules) are collected in a local `data/` directory.
2.  **Advanced Parsing:** Uses **LlamaParse** to convert PDFs into structured Markdown.
    * *Why?* Standard parsers break tables. LlamaParse preserves table structure (Rows/Columns) using Markdown syntax.
3.  **Smart Caching:** Parsed data is serialized and saved to `parsed_data.json`.
    * *Benefit:* Eliminates redundant API calls to LlamaCloud, saving costs and speeding up experimentation during the chunking phase.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Parsing:** LlamaParse (LlamaIndex)
* **Orchestration:** LangChain
* **Utilities:** `python-dotenv`, `json`

---

## 📂 Project Structure

```bash
.
├── data/                   # Raw PDF files
├── src/
│   └── ingest.py           # Script to parse PDFs and save to JSON
├── parsed_data.json        # Cached output (Markdown text + Metadata)
├── .env                    # API Keys (Not committed)
└── README.md
```

---

## ⚙️ Setup & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
